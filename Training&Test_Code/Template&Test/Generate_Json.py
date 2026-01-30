#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建 LLaVA LoRA 训练用数据集的脚本。

输入：
- Instruction.csv （包含 interval_id, Participant_ID, Distance, Position, Density, Task,
                   ObjGroup, ObjPosition, Transcript, Start_Time, End_Time 等）
- Generated_Image/ 下面若干子文件夹，例如：
    P1_D1_P1_High_T3_27_Mid/
        1.png
        2.png
        ...

规则：
- 子文件夹名由以下字段拼成：
  Participant_ID + Distance + Position + Density + Task + ObjGroup + ObjPosition
  例如：P1_D1_P1_High_T3_27_Mid

- Ground truth 六个字段：
  Type, Color, Shape, Local Position, Global Position, Task Type
  全部从 ObjGroup / Distance / Position / Density / ObjPosition / Task 自动推导。

输出：
- llava_lora_dataset.jsonl
  每行是一个样本，包含：
    - "id"
    - "images": 图像相对路径列表
    - "conversations": user + assistant 对话（assistant 为 ground truth JSON）
"""

import os
import csv
import json

# === 路径设置：按你当前的目录结构来 ===
DATA_ROOT = "/home/lesong_llava/LLaVA-NeXT/Lesong_Model/Formal/UHN"
IMAGE_ROOT = os.path.join(DATA_ROOT, "Image")
CSV_PATH = os.path.join(DATA_ROOT, "Instruction.csv")
OUT_PATH = os.path.join(DATA_ROOT, "llava_lora_dataset_UHN.jsonl")

# === System prompt 模板：和你之后推理时保持一致会更好 ===
SYSTEM_PROMPT = (
    "You are a vision assistant that helps a robot interpret the user's instruction.\n"
    "The user gives a natural language instruction, and the robot captures several frames (images) "
    "during the instruction period.\n"
    "The images in this sample are ordered in time from early to late.\n"
    "Note: visual features may be reweighted using gaze-derived saliency.\n"
    "Please infer the most likely single target object that the user refers to, "
    "based on BOTH the instruction and all frames.\n"
    "Answer ONLY in JSON with the following fields (in English):\n"
    "  - Type: object type (e.g., 'cup', 'vase', 'plant').\n"
    "  - Color: main color (e.g., 'white', 'green', 'brown').\n"
    "  - Shape: shape / appearance (e.g., 'cylindrical', 'rectangular', 'irregular-shaped').\n"
    "  - Local Position: local position relative to nearby objects (e.g., 'the left one among the three objects').\n"
    "  - Global Position: global position in the environment (e.g., 'on the wall-mounted shelf').\n"
    "  - Task Type: the type of task (e.g., 'Move', 'Pour water', 'Clean').\n"
    "Do not output any extra text outside the JSON."
)


# ===================== 映射函数区域 =====================

def get_type_color_shape(obj_group: int):
    """
    根据 ObjGroup (1~27) 返回 (Type, Color, Shape)，全英文。
    Type: vase / plant / cup
    Color: green / brown / white
    Shape: cylindrical / rectangular / irregular-shaped
    """
    if obj_group < 1 or obj_group > 27:
        raise ValueError(f"ObjGroup out of range: {obj_group} (should be 1~27)")

    obj_index = obj_group - 1  # 0 ~ 26

    colors = ["green", "brown", "white"]
    types = ["vase", "plant", "cup"]

    color_index = obj_index // 9          # 0~2
    type_index = (obj_index % 9) // 3     # 0~2
    shape_index = (obj_index % 3) + 1     # 1~3

    color = colors[color_index]
    obj_type = types[type_index]

    if shape_index == 1:
        shape = "cylindrical"
    elif shape_index == 2:
        shape = "rectangular"
    else:  # 3
        shape = "irregular-shaped"

    return obj_type, color, shape


def get_task_type(task_code: str):
    """
    Task 列: T1/T2/T3 -> Task Type 英文
    """
    task_code = (task_code or "").strip()
    mapping = {
        "T1": "Move",
        "T2": "Pour water",
        "T3": "Clean",
    }
    return mapping.get(task_code, "Unknown")


def get_local_position(density: str, obj_position: str):
    """
    Local Position:
    - Density == 'Low'  -> only one object
    - else + ObjPosition(Left/Mid/Right)
    """
    density = (density or "").strip()
    obj_position = (obj_position or "").strip()

    if density == "Low":
        return "the only object in this area"

    pos_map = {
        "Left": "the left one among the three objects",
        "Mid": "the middle one among the three objects",
        "Right": "the right one among the three objects",
    }
    return pos_map.get(obj_position, "one of the three objects")


def get_global_position(distance: str, position: str):
    """
    Global Position:
    按 Distance (D1/D2/D3) + Position (P1~P9) 映射到英文描述
    """
    distance = (distance or "").strip()
    position = (position or "").strip()

    # D1
    if distance == "D1":
        if position in ["P1", "P2", "P3", "P4", "P5", "P6"]:
            return "on the wall-mounted shelf"
        elif position in ["P7", "P8", "P9"]:
            return "inside the storage box"
        else:
            return "unknown position for D1"

    # D2
    if distance == "D2":
        if position == "P1":
            return "on top of the cabinet"
        elif position == "P2":
            return "on the wall-mounted shelf"
        elif position == "P3":
            return "on top of the wall-mounted shelf"
        elif position == "P4":
            return "on the middle shelf inside the cabinet"
        elif position == "P5":
            return "on the upper shelf inside the cabinet"
        elif position == "P6":
            return "on the very top shelf inside the cabinet"
        elif position in ["P7", "P8", "P9"]:
            return "on the bottom shelf inside the cabinet"
        else:
            return "unknown position for D2"

    # D3
    if distance == "D3":
        if position == "P1":
            return "on top of the desk-top shelf"
        elif position == "P2":
            return "on top of the open shelf unit"
        elif position == "P3":
            return "on top of the cabinet"
        elif position == "P4":
            return "on the desk surface"
        elif position == "P5":
            return "on the middle level of the open shelf unit"
        elif position == "P6":
            return "on the middle shelf inside the cabinet"
        elif position == "P7":
            return "under the desk"
        elif position == "P8":
            return "under the open shelf unit"
        elif position == "P9":
            return "on the bottom shelf inside the cabinet"
        else:
            return "unknown position for D3"

    # 未知 Distance
    return "unknown global position"


def build_folder_name(row):
    """
    根据 CSV 里的字段拼出子文件夹名字：
    Participant_ID Distance Position Density Task ObjGroup ObjPosition
    -> P1_D1_P1_High_T3_27_Mid
    """
    # 注意：ObjGroup 可能是数字或字符串，这里统一转成字符串即可
    participant_id = row["Participant_ID"].strip()
    distance = row["Distance"].strip()
    position = row["Position"].strip()
    density = row["Density"].strip()
    task = row["Task"].strip()
    obj_group = str(row["ObjGroup"]).strip()
    obj_position = row["ObjPosition"].strip()

    return f"{participant_id}_{distance}_{position}_{density}_{task}_{obj_group}_{obj_position}"


def load_images_for_folder(folder_name):
    """
    读取子文件夹里所有图片路径，并按文件名的数字排序。
    返回相对路径列表，例如：
        ["Image/P1_D1_P1_High_T3_27_Mid/1.png", ...]
    """
    folder_path = os.path.join(IMAGE_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    ]
    if not files:
        raise ValueError(f"No image files found in folder: {folder_path}")

    def parse_numeric(name):
        stem = os.path.splitext(name)[0]
        try:
            return float(stem)
        except ValueError:
            # 如果不是纯数字，就按字符串排序
            return stem

    files = sorted(files, key=parse_numeric)

    # 计算相对于 DATA_ROOT 的相对路径
    image_folder_name = os.path.basename(IMAGE_ROOT)  # "Image"
    rel_paths = [
        os.path.join(image_folder_name, folder_name, f)
        for f in files
    ]
    return rel_paths


def detect_dialect(csv_path):
    """
    自动检测 CSV 分隔符（支持 , 和 \t）。
    如果检测失败，就默认用 Tab 分隔（因为你的示例是 TSV）。
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            # fallback：默认 Tab
            class SimpleDialect(csv.Dialect):
                delimiter = "\t"
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            dialect = SimpleDialect()
    return dialect


# ===================== 主流程 =====================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Instruction.csv not found at: {CSV_PATH}")
    if not os.path.isdir(IMAGE_ROOT):
        raise FileNotFoundError(f"Generated_Image folder not found at: {IMAGE_ROOT}")

    dialect = detect_dialect(CSV_PATH)

    samples = []
    missing_folders = 0
    error_rows = 0
    total_rows = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            total_rows += 1

            try:
                folder_name = build_folder_name(row)
            except KeyError as e:
                print(f"[跳过] 第 {total_rows} 行缺少字段: {e}")
                error_rows += 1
                continue

            try:
                image_paths = load_images_for_folder(folder_name)
            except FileNotFoundError:
                print(f"[警告] 对应图片文件夹不存在: {folder_name}")
                missing_folders += 1
                continue
            except Exception as e:
                print(f"[跳过] 读取图片失败 ({folder_name}): {e}")
                error_rows += 1
                continue

            # 指令文本
            instruction = (row.get("Transcript") or "").strip()

            # 时间范围（可选）
            start_t = (row.get("Start_Time") or "").strip()
            end_t = (row.get("End_Time") or "").strip()
            if start_t and end_t:
                time_info = f"The frames roughly correspond to the speech interval from {start_t} seconds to {end_t} seconds."
            else:
                time_info = ""

            # 构造 user 段文本
            image_placeholders = " ".join(["<image>"] * len(image_paths))
            user_value = (
                f"{image_placeholders}\n"
                f"The user's spoken instruction is: \"{instruction}\"\n"
                f"{time_info}"
            ).strip()

            # ---- Ground truth 六个字段，从 CSV 自动推导 ----
            try:
                obj_group = int(row["ObjGroup"])
            except Exception:
                print(f"[跳过] ObjGroup 不是有效整数 (row {total_rows}, value={row.get('ObjGroup')})")
                error_rows += 1
                continue

            distance = row.get("Distance", "")
            position = row.get("Position", "")
            density = row.get("Density", "")
            obj_position = row.get("ObjPosition", "")
            task_code = row.get("Task", "")

            obj_type, color, shape = get_type_color_shape(obj_group)
            local_pos = get_local_position(density, obj_position)
            global_pos = get_global_position(distance, position)
            task_type = get_task_type(task_code)

            answer_json = {
                "Type": obj_type,
                "Color": color,
                "Shape": shape,
                "Local Position": local_pos,
                "Global Position": global_pos,
                "Task Type": task_type,
            }

            # conversations：user 里加入 system prompt
            conversations = [
                {
                    "from": "user",
                    "value": f"{SYSTEM_PROMPT}\n\n{user_value}"
                },
                {
                    "from": "assistant",
                    "value": json.dumps(answer_json, ensure_ascii=False)
                },
            ]

            sample = {
                "id": folder_name,
                "images": image_paths,
                "conversations": conversations,
            }
            samples.append(sample)

    # 写出 jsonl
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("====================================")
    print(f"✅ Done. Output dataset: {OUT_PATH}")
    print(f"   Total CSV rows:     {total_rows}")
    print(f"   Valid samples:      {len(samples)}")
    print(f"   Missing folders:    {missing_folders}")
    print(f"   Error rows skipped: {error_rows}")
    print("====================================")


if __name__ == "__main__":
    main()
