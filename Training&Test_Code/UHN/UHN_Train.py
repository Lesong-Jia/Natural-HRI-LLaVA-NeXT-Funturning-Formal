#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict


from tqdm.auto import tqdm

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# 基本配置
# =========================
os.environ["WANDB_DISABLED"] = "true"

BASE_MODEL_PATH = "/home/lesong_llava/Natural-HRI-LLaVA-NeXT-Funturning-Formal/Output_Models/llava-next-interleave-qwen-7b"
DATA_PATH = "/home/lesong_llava/Natural-HRI-LLaVA-NeXT-Funturning-Formal/Training&Test_Code/UHN/llava_lora_dataset_UHN.jsonl"

BASE_OUTPUT_DIR = "/home/lesong_llava/Natural-HRI-LLaVA-NeXT-Funturning-Formal/Output_Models/UHN"
TEST_DATA_DIR = "/home/lesong_llava/Natural-HRI-LLaVA-NeXT-Funturning-Formal/Training&Test_Code/UHN/Test_Data"

SEED = 42
MAX_IMAGES = 15
N_FOLDS = 5

# =========================
# Gaze weighting 超参数
# =========================
GAZE_MIN_W = 0.6   # 全黑 gaze 时，所有 patch 权重=GAZE_MIN_W
GAZE_STRENGTH = 2
GAZE_GAMMA = 0.75

# =========================
# LoRA 目标（你要试的）
# =========================
LORA_KEYS = ["q_proj", "v_proj", "o_proj"]
INCLUDE_MM_PROJECTOR_LORA = True


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def strip_image_tokens(user_text: str) -> str:
    lines = user_text.splitlines()
    kept = [ln for ln in lines if "<image>" not in ln]
    return "\n".join(kept).strip()


def load_images(base_dir: str, image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in image_paths:
        img_path = p if p.startswith("/") else os.path.join(base_dir, p)
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    return images


def build_lora_targets(model, keys, include_mm_projector=True):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # ✅ 只允许 language_model 里的层（Qwen）
        in_lm = name.startswith("model.language_model.")

        # ✅ projector 允许全部 Linear（如果你要）
        in_proj = include_mm_projector and ("model.multi_modal_projector" in name)

        if not (in_lm or in_proj):
            continue

        if in_proj:
            targets.append(name)
            continue

        # in_lm: 只挑你想要的那几个关键词
        if any(k in name for k in keys):
            targets.append(name)

    return sorted(set(targets))



# =========================
# Gaze heatmap：路径映射 + 读取
# =========================
def map_image_to_gaze_relpath(img_relpath: str) -> str:
    """
    Image/xxx/1.png  ->  Gaze/xxx/1.png
    兼容 /Image/ 出现在路径中间的情况。
    """
    p = img_relpath.replace("\\", "/")

    if p.startswith("Image/"):
        return "Gaze/" + p[len("Image/"):]
    if "/Image/" in p:
        return p.replace("/Image/", "/Gaze/")
    if "Image" in p:
        return p.replace("Image", "Gaze", 1)
    return p


def load_gaze_map(base_dir: str, img_relpath: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    读取灰度 gaze heatmap (黑=0, 白=1)，返回 (1,H,W) float32 in [0,1]
    - 不存在/读取失败 => 全 0
    """
    H, W = target_hw
    gaze_rel = map_image_to_gaze_relpath(img_relpath)
    gaze_path = gaze_rel if gaze_rel.startswith("/") else os.path.join(base_dir, gaze_rel)

    if (not gaze_rel) or (not os.path.exists(gaze_path)):
        return torch.zeros(1, H, W, dtype=torch.float32)

    try:
        g = Image.open(gaze_path).convert("L")
        if g.size != (W, H):
            g = g.resize((W, H), resample=Image.BILINEAR)
        arr = np.array(g, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        tqdm.write(f"[WARN] Failed to load gaze map {gaze_path}: {e}")
        return torch.zeros(1, H, W, dtype=torch.float32)


# =========================
# Dataset
# =========================
class LLaVANextChatDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], data_path: str):
        self.data = data
        self.base_dir = os.path.dirname(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.data[idx]
        sid = s.get("id", str(idx))
        conv = s["conversations"]
        image_paths = s["images"]

        if len(conv) < 2:
            raise ValueError(f"Sample {sid} has <2 conversations")

        user_text_raw = conv[0]["value"]
        assistant_text = conv[1]["value"]

        num_imgs = len(image_paths)
        num_image_tokens = user_text_raw.count("<image>")
        if num_image_tokens != num_imgs:
            raise ValueError(
                f"[Sample {sid}] num_images={num_imgs} but found {num_image_tokens} '<image>' tokens."
            )

        user_text_clean = strip_image_tokens(user_text_raw)

        # ✅ 不再 open 图片，只传相对路径
        return {
            "id": sid,
            "user_text": user_text_clean,
            "assistant_text": assistant_text,
            "image_paths": image_paths,
        }


# =========================
# DataCollator：chat_template + labels（prefix/full同体系）+ gaze_maps (兼容4D/5D pixel_values)
# =========================
@dataclass
class DataCollatorForLLaVANextChat:
    processor: AutoProcessor
    base_dir: str
    img_cache_size: int = 256      # 你可以按内存调：128/256/512
    gaze_cache_size: int = 512

    def __post_init__(self):
        self._img_cache = OrderedDict()
        self._gaze_cache = OrderedDict()

    def _cache_get(self, cache: OrderedDict, key: str):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_put(self, cache: OrderedDict, key: str, value, max_size: int):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _load_rgb_cached(self, rel_or_abs: str) -> Image.Image:
        path = rel_or_abs if rel_or_abs.startswith("/") else os.path.join(self.base_dir, rel_or_abs)
        hit = self._cache_get(self._img_cache, path)
        if hit is not None:
            # PIL Image 复用一般没问题；如果你担心线程安全，可以 hit.copy()
            return hit
        img = Image.open(path).convert("RGB")
        self._cache_put(self._img_cache, path, img, self.img_cache_size)
        return img

    def _load_gaze_cached(self, img_relpath: str, target_hw: Tuple[int,int]) -> torch.Tensor:
        H, W = target_hw
        gaze_rel = map_image_to_gaze_relpath(img_relpath)
        gaze_path = gaze_rel if gaze_rel.startswith("/") else os.path.join(self.base_dir, gaze_rel)

        key = f"{gaze_path}|{H}x{W}"
        hit = self._cache_get(self._gaze_cache, key)
        if hit is not None:
            # 返回 clone，避免后面意外 inplace
            return hit.clone()

        if (not gaze_rel) or (not os.path.exists(gaze_path)):
            t = torch.zeros(1, H, W, dtype=torch.float32)
            self._cache_put(self._gaze_cache, key, t, self.gaze_cache_size)
            return t.clone()

        try:
            g = Image.open(gaze_path).convert("L")
            if g.size != (W, H):
                g = g.resize((W, H), resample=Image.BILINEAR)
            arr = np.array(g, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
            self._cache_put(self._gaze_cache, key, t, self.gaze_cache_size)
            return t.clone()
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load gaze map {gaze_path}: {e}")
            t = torch.zeros(1, H, W, dtype=torch.float32)
            self._cache_put(self._gaze_cache, key, t, self.gaze_cache_size)
            return t.clone()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages_batch = []
        images_batch = []
        img_paths_list = []

        for f in features:
            user_text = f["user_text"]
            assistant_text = f["assistant_text"]
            img_paths = f.get("image_paths", [])

            # ✅ 在 collator 里加载图片（带缓存）
            imgs = [self._load_rgb_cached(p) for p in img_paths]

            user_content = [{"type": "text", "text": user_text}] + [{"type": "image"} for _ in imgs]
            assistant_content = [{"type": "text", "text": assistant_text}]

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            messages_batch.append(messages)
            images_batch.append(imgs)
            img_paths_list.append(img_paths)

        full_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_batch
        ]

        prefix_messages_batch = []
        for m in messages_batch:
            prefix_messages_batch.append([
                m[0],
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ])
        prefix_texts = [
            self.processor.apply_chat_template(pm, tokenize=False, add_generation_prompt=False)
            for pm in prefix_messages_batch
        ]

        batch = self.processor(
            text=full_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is None:
            raise ValueError("attention_mask is required for labeling.")

        prefix_batch = self.processor(
            text=prefix_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_attn = prefix_batch.get("attention_mask", None)
        if prefix_attn is None:
            raise ValueError("prefix_batch has no attention_mask.")

        labels = torch.full_like(input_ids, -100)
        B = input_ids.size(0)
        for i in range(B):
            prefix_len = int(prefix_attn[i].sum().item())
            full_len = int(attention_mask[i].sum().item())
            if prefix_len < full_len:
                labels[i, prefix_len:full_len] = input_ids[i, prefix_len:full_len]
        batch["labels"] = labels

        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("processor output has no pixel_values; cannot apply gaze weighting.")

        # ✅ gaze：用 cached loader
        if pixel_values.dim() == 5:
            B2, M, _, H, W = pixel_values.shape
            gaze_maps = torch.zeros((B2, M, 1, H, W), dtype=torch.float32)
            for i in range(B2):
                n_imgs = len(images_batch[i])
                img_paths = img_paths_list[i]
                for j in range(min(n_imgs, M)):
                    relp = img_paths[j] if j < len(img_paths) else ""
                    gaze_maps[i, j, 0] = self._load_gaze_cached(relp, target_hw=(H, W))
            batch["gaze_maps"] = gaze_maps
            return batch

        if pixel_values.dim() == 4:
            T, _, H, W = pixel_values.shape
            flat_gaze = []
            for i in range(len(images_batch)):
                img_paths = img_paths_list[i]
                n_imgs = len(images_batch[i])
                for j in range(n_imgs):
                    relp = img_paths[j] if j < len(img_paths) else ""
                    flat_gaze.append(self._load_gaze_cached(relp, target_hw=(H, W)))
            if len(flat_gaze) != T:
                raise ValueError(
                    f"pixel_values has T={T} images, but built gaze for {len(flat_gaze)} images."
                )
            batch["gaze_maps"] = torch.stack(flat_gaze, dim=0)  # (T,1,H,W)
            return batch

        raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")


# =========================
# Trainer：把 gaze_maps 临时挂到 model 上
# =========================
class GazeWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)  # 避免原地 pop 影响别的逻辑
        gaze_maps = inputs.pop("gaze_maps", None)
        model._gaze_maps = gaze_maps
        try:
            outputs = model(**inputs)
            loss = outputs.loss
        finally:
            model._gaze_maps = None
        return (loss, outputs) if return_outputs else loss


# =========================
# Patch-level weighting hook（兼容 feats 3D/4D + gaze 4D/5D）
# =========================
def install_gaze_weighting_hook(model):
    """
    最稳的 gaze 注入点：multi_modal_projector 的 forward_pre_hook
    - 不依赖 get_image_features 是否被 forward 调用
    - 支持 gaze_maps 为 (T,1,H,W) 或 (B,M,1,H,W)
    """
    # PeftModel 兼容：Trainer 里传进来的 model 可能是 PeftModel
    base = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model

    # LlavaForConditionalGeneration 通常主体在 base.model
    core = base.model if hasattr(base, "model") else base
    projector = core.multi_modal_projector

    def _infer_grid(n: int):
        if n <= 0:
            return None, None, False
        s = int(n ** 0.5)
        if s * s == n:
            return s, s, False
        s = int((n - 1) ** 0.5)
        if s * s == (n - 1):
            return s, s, True
        return None, None, False

    def _weight_feats(feats: torch.Tensor, gaze_maps: torch.Tensor) -> torch.Tensor:
        """
        feats: (T,N,D) 或 (B,M,N,D)
        gaze_maps: (T,1,H,W) 或 (B,M,1,H,W)
        """
        if gaze_maps is None:
            return feats

        # ---- gaze 统一成 (T,1,H,W) ----
        if gaze_maps.dim() == 5:
            B, M, _, H, W = gaze_maps.shape
            gaze_flat = gaze_maps.view(B * M, 1, H, W)
        elif gaze_maps.dim() == 4:
            gaze_flat = gaze_maps
        else:
            return feats

        # ---- feats 统一成 (T,N,D) ----
        if feats.dim() == 4:
            B, M, N, D = feats.shape
            feats_flat = feats.view(B * M, N, D)
            need_reshape_back = ("BM", B, M, N, D)
        elif feats.dim() == 3:
            feats_flat = feats
            N = feats.shape[1]
            D = feats.shape[2]
            need_reshape_back = ("T",)
        else:
            return feats

        T = feats_flat.shape[0]
        if gaze_flat.shape[0] < T:
            # 不够就不加权（避免错配）
            return feats

        gaze_flat = gaze_flat[:T].to(device=feats_flat.device, dtype=torch.float32)  # (T,1,H,W)

        gh, gw, has_cls = _infer_grid(N)

        if gh is None:
            # fallback：2D gaze -> 1D 权重
            gaze_1d = gaze_flat[:, 0].reshape(T, -1).unsqueeze(1)  # (T,1,HW)
            w = F.interpolate(gaze_1d, size=N, mode="linear", align_corners=False).squeeze(1)  # (T,N)
            w = w / (w.amax(dim=-1, keepdim=True) + 1e-6)
            w = GAZE_MIN_W + (1.0 - GAZE_MIN_W) * (GAZE_STRENGTH * w).clamp(0, 1) ** GAZE_GAMMA
        else:
            gm = F.interpolate(gaze_flat, size=(gh, gw), mode="bilinear", align_corners=False)[:, 0]  # (T,gh,gw)
            gm = gm / (gm.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            w_patch = GAZE_MIN_W + (1.0 - GAZE_MIN_W) * (GAZE_STRENGTH * gm).clamp(0, 1) ** GAZE_GAMMA
            w_patch = w_patch.reshape(T, gh * gw)  # (T,N') where N'=gh*gw
            if has_cls:
                ones = torch.ones((T, 1), device=w_patch.device, dtype=w_patch.dtype)
                w = torch.cat([ones, w_patch], dim=-1)
            else:
                w = w_patch

        feats_w = feats_flat * w.unsqueeze(-1)  # (T,N,D)

        # ---- 还原形状 ----
        if need_reshape_back[0] == "BM":
            _, B, M, N, D = need_reshape_back
            return feats_w.view(B, M, N, D)
        return feats_w

    def pre_hook(module, inputs):
        if not inputs:
            return inputs
        feats = inputs[0]
        gaze_maps = getattr(model, "_gaze_maps", None)

        feats = _weight_feats(feats, gaze_maps)
        return (feats,) + tuple(inputs[1:])

    projector.register_forward_pre_hook(pre_hook)


# =========================
# 5折交叉验证训练函数
# =========================
def train_fold(
    fold_idx: int,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    output_dir: str,
    test_data_path: str,
    data_path: str,
):
    torch.cuda.empty_cache()
    set_seed(SEED)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Fold {fold_idx + 1}/{N_FOLDS}")
    tqdm.write(f"{'='*60}")
    tqdm.write(f"Train samples: {len(train_data)}")
    tqdm.write(f"Test samples: {len(test_data)}")

    save_jsonl(test_data_path, test_data)
    tqdm.write(f"Saved test split -> {test_data_path}")

    tqdm.write("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tqdm.write("Loading base model in 4-bit...")
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 先做 k-bit 训练准备（包含一些 dtype / grad 设置）
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ====== 关键：构造 LoRA target（含 mm projector）======
    target_modules = build_lora_targets(
        model,
        keys=LORA_KEYS,
        include_mm_projector=INCLUDE_MM_PROJECTOR_LORA,
    )

    if len(target_modules) == 0:
        raise RuntimeError(
            "No LoRA target modules found. Please check LORA_KEYS / model module names."
        )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,   # ✅ transformer + multi_modal_projector 全部在一个 adapter 里
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # gaze weighting hook（仍然挂在 projector 入口）
    install_gaze_weighting_hook(model)

    train_dataset = LLaVANextChatDataset(train_data, data_path)
    data_collator = DataCollatorForLLaVANextChat(
        processor=processor,
        base_dir=os.path.dirname(data_path),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # 你如果发现不稳/过拟合，可以试 5e-5 或 2e-5
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,        # 你可以试 4 / 8
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,  # workers 常驻，减少反复 fork 开销
    )

    trainer = GazeWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    tqdm.write(f"===== TRAIN Fold {fold_idx + 1}/{N_FOLDS} =====")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)   # ✅ 仍然是一个 adapter 目录
    processor.save_pretrained(output_dir)

    tqdm.write(f"\n✅ Fold {fold_idx + 1} Done. LoRA adapter saved to: {output_dir}")
    tqdm.write(f"   Test split saved to: {test_data_path}")
    tqdm.write("   Base model unchanged.")

    del model, trainer, processor, train_dataset, data_collator, training_args, lora_config, bnb_config
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# =========================
# 主逻辑：5-fold
# =========================
def main():
    set_seed(SEED)

    tqdm.write(f"Loading data: {DATA_PATH}")
    all_data = load_jsonl(DATA_PATH)
    tqdm.write(f"Loaded {len(all_data)} samples")

    filtered = []
    dropped = 0
    for s in all_data:
        img_count = len(s.get("images", []))
        if 0 < img_count <= MAX_IMAGES:
            filtered.append(s)
        else:
            dropped += 1
    all_data = filtered
    tqdm.write(f"After filtering (0 < img_count <= {MAX_IMAGES}): kept={len(all_data)}, dropped={dropped}")

    if len(all_data) == 0:
        raise ValueError("No samples after filtering.")

    random.shuffle(all_data)

    n_samples = len(all_data)
    fold_size = n_samples // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start_idx = i * fold_size
        end_idx = n_samples if i == N_FOLDS - 1 else (i + 1) * fold_size
        folds.append(all_data[start_idx:end_idx])

    tqdm.write("\n5-fold split:")
    for i, fold in enumerate(folds):
        tqdm.write(f"  Fold {i+1}: {len(fold)} samples")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for fold_idx in range(N_FOLDS):
        test_data = folds[fold_idx]
        train_data = []
        for j in range(N_FOLDS):
            if j != fold_idx:
                train_data.extend(folds[j])

        output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_idx + 1}")
        test_data_path = os.path.join(TEST_DATA_DIR, f"test_fold_{fold_idx + 1}.jsonl")

        train_fold(
            fold_idx=fold_idx,
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            test_data_path=test_data_path,
            data_path=DATA_PATH,
        )

    tqdm.write(f"\n{'='*60}")
    tqdm.write("✅ All 5 folds completed!")
    tqdm.write(f"   Models saved to: {BASE_OUTPUT_DIR}")
    tqdm.write(f"   Test data saved to: {TEST_DATA_DIR}")
    tqdm.write(f"{'='*60}")


if __name__ == "__main__":
    main()
