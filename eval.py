import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import sys
from typing import Optional

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config):
    qs = line["prompt"]
    qs += f"\n{args.question_extension}"

    if line["image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if line["image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = line["image"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


def evaluate_model(
        model_path: str,
        dataset_path: str,
        output_path: str,
        model_base: Optional[str] = None,
        question_extension: str = "Answer with the option's letter from the given choices directly.",
        conv_mode: str = "phi3",
        num_chunks: int = 1,
        chunk_idx: int = 0,
        temperature: float = 0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        seed: int = 42
):
    """封装后的评估函数

    Args:
        model_path: 模型文件路径
        dataset_path: 数据集路径
        output_path: 结果输出路径
        model_base: 基础模型路径（可选）
        question_extension: 问题扩展提示语
        conv_mode: 对话模式
        num_chunks: 数据分块数
        chunk_idx: 当前处理的数据块索引
        temperature: 生成温度
        top_p: 核采样参数
        num_beams: 束搜索宽度
        max_new_tokens: 最大生成token数
        seed: 随机种子
    """

    # 参数封装
    class Args:
        pass

    args = Args()

    # 核心路径参数
    args.model_path = model_path
    args.coco_dataset_path = dataset_path
    args.answers_file = output_path

    # 模型参数
    args.model_base = model_base

    # 生成参数
    args.question_extension = question_extension
    args.conv_mode = conv_mode
    args.num_chunks = num_chunks
    args.chunk_idx = chunk_idx
    args.temperature = temperature
    args.top_p = top_p
    args.num_beams = num_beams
    args.max_new_tokens = max_new_tokens
    args.seed = seed

    # 固定参数（原代码默认值）
    args.noise = None  # 原代码未实际使用

    # 执行评估
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载模型
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)

    # 加载数据集
    questions = load_dataset(args.coco_dataset_path, split="train")

    # 准备输出文件
    if not args.answers_file.endswith(".jsonl"):
        raise ValueError("输出文件必须是.jsonl格式")
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    ans_file = open(args.answers_file, "w")

    # 数据处理
    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    for line in tqdm(questions, total=len(questions)):
        idx += 1
        if idx < valid_chunk[0] or idx > valid_chunk[1]:
            continue

        input_ids, image_tensor, image_sizes, prompt = process(line, args, tokenizer, image_processor, model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "questionId": idx,
            "image": line["img_name"],
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": line["answer"],
            "category": line["sub_task"],
            "options": line["choices"],
            "image_id": line["image_id"],
            "model_id": model_name
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./cambrian-phi3-3b")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="phi3")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed
    )