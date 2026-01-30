#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5-fold eval for LLaVA-NeXT Interleave Qwen-7B with gaze-weighted patch features.

✅ Requirement (your latest): FULLY INDEPENDENT base vs lora
- BASE pass uses a fresh, standalone base model instance
- LORA pass uses another fresh base model instance + LoRA adapter injected
- BOTH BASE and LORA use the SAME gaze weighting hook
- Hook installed ONCE per model instance (anti-duplicate)
- Gaze maps are attached per forward via model._gaze_maps (cleared in finally)

Note:
- This is the most reliable way to guarantee "base is not polluted by LoRA injection".
"""

import os
import json
import csv
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


# =========================
# Config
# =========================
BASE_MODEL_PATH = "/home/lesong_llava/models/llava-next-interleave-qwen-7b"

# 5-fold CV paths
BASE_OUTPUT_DIR = "/home/lesong_llava/models/UHN_V2"
TEST_DATA_DIR = "/home/lesong_llava/LLaVA-NeXT/Lesong_Model/Formal/UHN/Test_Data_V2"
OUT_CSV_PATH = "/home/lesong_llava/LLaVA-NeXT/Lesong_Model/Formal/UHN/eval_results_5fold_UHN_V3.csv"

# Original dataset jsonl (for base_dir of images/gaze relpaths)
DATA_PATH = "/home/lesong_llava/LLaVA-NeXT/Lesong_Model/Formal/UHN/llava_lora_dataset_UHN.jsonl"

N_FOLDS = 5

# Generation
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 42

# Filter
MAX_IMAGES = 15

# =========================
# [GAZE] MUST match training
# =========================
USE_GAZE = True
GAZE_MIN_W = 1
GAZE_STRENGTH = 1.0
GAZE_GAMMA = 1.0

# Debug
DEBUG_PRINT = False         # True => debug for every sample
DEBUG_PRINT_FIRST_K = 1     # debug first K samples per pass (0 disables)


# =========================
# Utils
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def strip_image_tokens(user_text: str) -> str:
    """
    Only remove <image> tokens; keep other text lines.
    """
    t = user_text.replace("<image>", "")
    lines = [" ".join(ln.split()).strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def load_images(base_dir: str, image_paths: List[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in image_paths:
        img_path = p if p.startswith("/") else os.path.join(base_dir, p)
        img = Image.open(img_path).convert("RGB")
        imgs.append(img)
    return imgs


def build_prompt(processor: AutoProcessor, user_text: str, num_images: int) -> str:
    user_content = [{"type": "text", "text": user_text}] + [{"type": "image"} for _ in range(num_images)]
    messages = [{"role": "user", "content": user_content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def clean_model_output(text: Optional[str]) -> str:
    if text is None:
        return ""
    t = text.strip()
    lower = t.lower()
    for key in ["assistant\n", "assistant:", "<assistant>", "### assistant", "role: assistant"]:
        idx = lower.rfind(key)
        if idx != -1:
            t = t[idx + len(key):].strip()
            lower = t.lower()
    for end_tok in ["</s>", "<|eot_id|>", "<|endoftext|>"]:
        t = t.replace(end_tok, "").strip()
    return t


# =========================
# [GAZE] mapping + loading
# =========================
def map_image_to_gaze_relpath(img_relpath: str) -> str:
    """
    Image/xxx/1.png  ->  Gaze/xxx/1.png
    Compatible with '/Image/' in the middle.
    """
    p = (img_relpath or "").replace("\\", "/")
    if p.startswith("Image/"):
        return "Gaze/" + p[len("Image/"):]
    if "/Image/" in p:
        return p.replace("/Image/", "/Gaze/")
    if "Image" in p:
        return p.replace("Image", "Gaze", 1)
    return p


def load_gaze_map(base_dir: str, img_relpath: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Read grayscale gaze heatmap (0..1), return (1,H,W) float32
    Missing => zeros
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
        return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
    except Exception as e:
        print(f"[WARN] Failed to load gaze map {gaze_path}: {e}")
        return torch.zeros(1, H, W, dtype=torch.float32)


# =========================
# [GAZE] hook install (ONCE)
# =========================
def _get_core_projector(model):
    """
    Find multi_modal_projector robustly for both:
    - LlavaForConditionalGeneration
    - PeftModel wrapping base model
    """
    base = model
    # unwrap peft -> underlying llava
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model

    core = base.model if hasattr(base, "model") else base
    if not hasattr(core, "multi_modal_projector"):
        raise AttributeError("Cannot find multi_modal_projector on this model structure.")
    return core.multi_modal_projector


def install_gaze_weighting_hook(model):
    """
    Install once per model instance. Hook reads model._gaze_maps.
    """
    if not USE_GAZE:
        return

    if getattr(model, "_gaze_hook_installed", False):
        return
    model._gaze_hook_installed = True

    projector = _get_core_projector(model)

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

    def _weight_feats(feats: torch.Tensor, gaze_maps: Optional[torch.Tensor]) -> torch.Tensor:
        if (gaze_maps is None) or (not isinstance(feats, torch.Tensor)):
            return feats

        # gaze -> (T,1,H,W) or (B,M,1,H,W)
        if gaze_maps.dim() == 5:
            B, M, _, H, W = gaze_maps.shape
            gaze_flat = gaze_maps.view(B * M, 1, H, W)
        elif gaze_maps.dim() == 4:
            gaze_flat = gaze_maps
        else:
            return feats

        # feats -> (T,N,D) or (B,M,N,D)
        if feats.dim() == 4:
            B, M, N, D = feats.shape
            feats_flat = feats.view(B * M, N, D)
            reshape_back = ("BM", B, M, N, D)
        elif feats.dim() == 3:
            feats_flat = feats
            N = feats.shape[1]
            D = feats.shape[2]
            reshape_back = ("T",)
        else:
            return feats

        T = feats_flat.shape[0]
        if gaze_flat.shape[0] < T:
            return feats  # mismatch => no weighting

        gaze_flat = gaze_flat[:T].to(device=feats_flat.device, dtype=torch.float32)

        gh, gw, has_cls = _infer_grid(N)

        if gh is None:
            gaze_1d = gaze_flat[:, 0].reshape(T, -1).unsqueeze(1)  # (T,1,HW)
            w = F.interpolate(gaze_1d, size=N, mode="linear", align_corners=False).squeeze(1)  # (T,N)
            w = w / (w.amax(dim=-1, keepdim=True) + 1e-6)
            w = GAZE_MIN_W + (1.0 - GAZE_MIN_W) * (GAZE_STRENGTH * w).clamp(0, 1) ** GAZE_GAMMA
        else:
            gm = F.interpolate(gaze_flat, size=(gh, gw), mode="bilinear", align_corners=False)[:, 0]  # (T,gh,gw)
            gm = gm / (gm.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            w_patch = GAZE_MIN_W + (1.0 - GAZE_MIN_W) * (GAZE_STRENGTH * gm).clamp(0, 1) ** GAZE_GAMMA
            w_patch = w_patch.reshape(T, gh * gw)
            if has_cls:
                ones = torch.ones((T, 1), device=w_patch.device, dtype=w_patch.dtype)
                w = torch.cat([ones, w_patch], dim=-1)
            else:
                w = w_patch

        feats_w = feats_flat * w.unsqueeze(-1)

        if reshape_back[0] == "BM":
            _, B, M, N, D = reshape_back
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
# [GAZE] build gaze_maps aligned with pixel_values
# =========================
def build_gaze_maps_from_inputs(
    inputs: Dict[str, Any],
    base_dir: str,
    image_paths: List[str],
    num_imgs: int,
) -> torch.Tensor:
    pv = inputs.get("pixel_values", None)
    if pv is None:
        raise ValueError("processor output has no pixel_values; cannot build gaze_maps.")

    if pv.dim() == 5:
        # (B=1, M, C, H, W)
        B, M, _, H, W = pv.shape
        gaze_maps = torch.zeros((B, M, 1, H, W), dtype=torch.float32)
        for j in range(min(num_imgs, M)):
            gaze_maps[0, j, 0] = load_gaze_map(base_dir, image_paths[j], target_hw=(H, W))
        return gaze_maps

    if pv.dim() == 4:
        # (T, C, H, W)
        T, _, H, W = pv.shape
        flat = []
        for j in range(num_imgs):
            flat.append(load_gaze_map(base_dir, image_paths[j], target_hw=(H, W)))  # (1,H,W)
        if len(flat) != T:
            raise ValueError(
                f"pixel_values has T={T}, but built gaze for {len(flat)} images. "
                f"Check image flatten order."
            )
        return torch.stack(flat, dim=0)  # (T,1,H,W)

    raise ValueError(f"Unexpected pixel_values shape: {tuple(pv.shape)} (expect 4D or 5D)")


# =========================
# Generation
# =========================
@torch.inference_mode()
def generate_answer(
    model,
    processor: AutoProcessor,
    user_text: str,
    imgs: List[Image.Image],
    image_paths: List[str],
    base_dir: str,
    device: torch.device,
    debug: bool = False,
) -> str:
    prompt = build_prompt(processor, user_text, num_images=len(imgs))

    inputs = processor(
        text=[prompt],
        images=[imgs],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    gaze_maps = None
    if USE_GAZE:
        gaze_maps = build_gaze_maps_from_inputs(
            inputs=inputs,
            base_dir=base_dir,
            image_paths=image_paths,
            num_imgs=len(imgs),
        )

    # Move inputs to a "primary device" (works with device_map="auto" in most cases)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE)
    if DO_SAMPLE:
        gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P))

    # Attach gaze_maps per call
    model._gaze_maps = gaze_maps if USE_GAZE else None

    try:
        output_ids = model.generate(**inputs, **gen_kwargs)
    finally:
        model._gaze_maps = None

    input_len = inputs["input_ids"].shape[1]
    gen_only = output_ids[0, input_len:]

    if debug:
        gen_len = output_ids.shape[1] - input_len
        raw_text = processor.tokenizer.decode(gen_only, skip_special_tokens=False)
        text_skip = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
        print(f"[DEBUG] gen_len={gen_len} (output_len={output_ids.shape[1]}, input_len={input_len})")
        print("[DEBUG] raw(no-skip) =", repr(raw_text[:200]))
        print("[DEBUG] dec(skip)   =", repr(text_skip[:200]))
        try:
            print("[DEBUG] ids head   =", gen_only[:20].tolist())
        except Exception:
            pass
        if USE_GAZE and gaze_maps is not None:
            print("[DEBUG] gaze mean/max =", float(gaze_maps.mean()), float(gaze_maps.max()))

    text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
    return clean_model_output(text)


def prepare_samples(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, s in enumerate(raw_data):
        sid = s.get("id", str(idx))
        conv = s.get("conversations", [])
        image_paths = s.get("images", [])

        if len(conv) < 2:
            print(f"[WARN] {sid}: conversations < 2, skip")
            continue

        img_count = len(image_paths)
        if not (0 < img_count <= MAX_IMAGES):
            print(f"[WARN] {sid}: img_count={img_count} out of range, skip")
            continue

        user_text_raw = conv[0].get("value", "")
        gt = conv[1].get("value", "")

        # strict alignment check
        if user_text_raw.count("<image>") != img_count:
            print(f"[WARN] {sid}: <image> count mismatch (found {user_text_raw.count('<image>')} vs {img_count}), skip")
            continue

        user_text = strip_image_tokens(user_text_raw)

        samples.append({
            "id": sid,
            "user_text": user_text,
            "gt": gt,
            "image_paths": image_paths,
        })
    return samples


# =========================
# Independent model loading helpers
# =========================
def load_base_model(bnb_config: BitsAndBytesConfig):
    """
    Load a FRESH base model instance (4-bit) with device_map="auto".
    This is used to guarantee independence between BASE and LORA passes.
    """
    m = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def get_primary_device(model) -> torch.device:
    """
    Pick a primary device for inputs.to(device).
    With device_map="auto", parameters can be on multiple GPUs,
    but picking the first parameter device usually works for inputs.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Test one fold (FULLY INDEPENDENT: base_model vs lora_model, both gaze-weighted)
# =========================
def test_fold(
    fold_idx: int,
    processor: AutoProcessor,
    bnb_config: BitsAndBytesConfig,
) -> List[Dict[str, Any]]:

    print(f"\n{'='*60}")
    print(f"Testing Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*60}")

    lora_adapter_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_idx + 1}")
    test_data_path = os.path.join(TEST_DATA_DIR, f"test_fold_{fold_idx + 1}.jsonl")

    if not os.path.exists(lora_adapter_dir):
        raise FileNotFoundError(f"LoRA adapter not found: {lora_adapter_dir}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    print(f"Loading test data: {test_data_path}")
    raw_data = load_jsonl(test_data_path)
    base_dir = os.path.dirname(DATA_PATH)

    samples = prepare_samples(raw_data)
    n = len(samples)
    print(f"Prepared {n} valid samples")
    if n == 0:
        print(f"[WARN] Fold {fold_idx + 1}: No valid samples, skipping")
        return []

    base_preds = [""] * n
    lora_preds = [""] * n

    # ==========================================================
    # PASS 1/2: BASE (fresh base model instance)
    # ==========================================================
    print(f"\n  [Fold {fold_idx + 1}] PASS 1/2: BASE (fresh base, gaze={USE_GAZE})")
    base_model = load_base_model(bnb_config)
    install_gaze_weighting_hook(base_model)
    device_base = get_primary_device(base_model)
    print(f"    BASE primary device: {device_base}")

    for i, s in enumerate(samples):
        imgs = load_images(base_dir, s["image_paths"])
        debug = DEBUG_PRINT or (DEBUG_PRINT_FIRST_K > 0 and i < DEBUG_PRINT_FIRST_K)

        base_preds[i] = generate_answer(
            model=base_model,
            processor=processor,
            user_text=s["user_text"],
            imgs=imgs,
            image_paths=s["image_paths"],
            base_dir=base_dir,
            device=device_base,
            debug=debug,
        )

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    BASE processed {i+1}/{n}")

    # Free BASE model completely before LORA model is loaded
    del base_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ==========================================================
    # PASS 2/2: LORA (another fresh base model + adapter)
    # ==========================================================
    print(f"\n  [Fold {fold_idx + 1}] PASS 2/2: LORA (fresh base + adapter, gaze={USE_GAZE})")
    base_for_lora = load_base_model(bnb_config)
    print(f"    Loading LoRA adapter: {lora_adapter_dir}")
    lora_model = PeftModel.from_pretrained(base_for_lora, lora_adapter_dir)
    lora_model.eval()

    install_gaze_weighting_hook(lora_model)
    device_lora = get_primary_device(lora_model)
    print(f"    LORA primary device: {device_lora}")

    for i, s in enumerate(samples):
        imgs = load_images(base_dir, s["image_paths"])
        debug = DEBUG_PRINT or (DEBUG_PRINT_FIRST_K > 0 and i < DEBUG_PRINT_FIRST_K)

        lora_preds[i] = generate_answer(
            model=lora_model,
            processor=processor,
            user_text=s["user_text"],
            imgs=imgs,
            image_paths=s["image_paths"],
            base_dir=base_dir,
            device=device_lora,
            debug=debug,
        )

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    LORA processed {i+1}/{n}")

    # Package fold results
    results: List[Dict[str, Any]] = []
    for i in range(n):
        results.append({
            "fold_num": fold_idx + 1,
            "id": samples[i]["id"],
            "gt": samples[i]["gt"],
            "base_pred": base_preds[i],
            "lora_pred": lora_preds[i],
        })

    del lora_model
    del base_for_lora
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"  [Fold {fold_idx + 1}] Done. {n} samples processed.")
    return results


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

    print(f"Loading processor: {BASE_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"USE_GAZE={USE_GAZE} (min_w={GAZE_MIN_W}, strength={GAZE_STRENGTH}, gamma={GAZE_GAMMA})")
    all_results: List[Dict[str, Any]] = []

    for fold_idx in range(N_FOLDS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        fold_results = test_fold(
            fold_idx=fold_idx,
            processor=processor,
            bnb_config=bnb_config,
        )
        all_results.extend(fold_results)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"Writing all results to CSV -> {OUT_CSV_PATH}")
    print(f"Total samples: {len(all_results)}")

    with open(OUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold_num", "id", "gt", "base_pred", "lora_pred"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n{'='*60}")
    print("✅ All folds completed!")
    print(f"   Total samples: {len(all_results)}")
    for i in range(N_FOLDS):
        c = sum(1 for r in all_results if r["fold_num"] == i + 1)
        print(f"   Fold {i + 1}: {c} samples")
    print(f"   Results saved to: {OUT_CSV_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
