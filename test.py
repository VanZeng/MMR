import os
import shutil
import random
import pandas as pd
import ast
from GaussianNoise import apply_noise
from accuracy import compute_metrics
import torch
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image

# 初始化数据容器
data = {
    'strengths': [],
    'overall': [],
    'count': [],
    'location': []
}

# 初始化基线数据
def calculate_random_baseline():
    df = pd.read_csv('metadata.csv', engine='python', quotechar='"')
    df['choices'] = df['choices'].apply(ast.literal_eval)
    df['correct_answer'] = df['answer'].str.strip('()')

    def random_select_answer(choices_list):
        return random.choice([chr(65 + i) for i in range(len(choices_list))])

    df['predicted_answer'] = df['choices'].apply(random_select_answer)
    df['is_correct'] = df['predicted_answer'] == df['correct_answer']

    return {
        'overall': df['is_correct'].mean(),
        'count': df.loc[df['sub_task'] == 'Count', 'is_correct'].mean(),
        'location': df.loc[df['sub_task'] == 'Relative location', 'is_correct'].mean()
    }

random_baseline = calculate_random_baseline()

def process(line, tokenizer, image_processor, model_config):
    qs = line["prompt"] + "\nAnswer with the option's letter from the given choices directly."
    
    if line["image"] is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # 处理图像
    if line["image"] is not None:
        image_path = os.path.join("./benchmark/benchmark_coco_filtered/images", line["image_id"])
        print(f"尝试加载图像: {image_path}")  # 调试路径
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"图像尺寸: {image.size}")  # 调试图像尺寸
            image_tensor = process_images([image], image_processor, model_config)
            print(f"图像张量形状: {image_tensor.shape}")  # 调试张量形状
            image_size = [image.size]
        except Exception as e:
            print(f"加载图像失败: {str(e)}")
            raise
    else:
        image_tensor = image_size = None

    # Tokenization
    input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f"input_ids 形状: {input_ids.shape}")  # 调试输入IDs
    print(f"input_ids 内容示例: {input_ids[0, :10]}")  # 显示前10个token

    return input_ids, image_tensor, image_size, qs

def evaluate_model(model_path, dataset_path, output_path, temperature=0.7, max_new_tokens=512):
    disable_torch_init()
    
    # 加载模型
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base=None, model_name=model_name)
    
    # 修改点1：只加载前5个样本
    dataset = load_dataset(dataset_path, split="train").select(range(5))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as ans_file:
        for idx, line in enumerate(tqdm(dataset)):
            try:
                # 修改点2：添加原始问题显示
                print(f"\n原始问题 {idx+1}:")
                print(line["prompt"])
                print("选项:", line["choices"])
                print("正确答案:", line["answer"])
                
                # 处理流程保持不变
                input_ids, image_tensor, image_size, prompt = process(line, tokenizer, image_processor, model.config)
                input_ids = input_ids.to(device='cuda', non_blocking=True)
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_size,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        use_cache=True
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # 修改点3：实时打印模型输出
                print("模型回答:", outputs)
                
                ans_file.write(json.dumps({
                    "questionId": idx,
                    "image": line["img_name"],
                    "prompt": prompt,
                    "answer": outputs,
                    "gt_answer": line["answer"],
                    "category": line["sub_task"],
                    "options": line["choices"], 
                    "model_id": model_name
                }) + "\n")
                ans_file.flush()
                
            except Exception as e:
                print(f"处理样本 {idx} 时发生错误: {str(e)}")
                continue

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()

# 修改点4：简化主流程为单次测试
try:
    current_strength = 0.1
    noise_dir = f"./benchmark/GaussianNoise_{current_strength:.1f}"
    
    # 创建必要目录
    os.makedirs(noise_dir, exist_ok=True)
    shutil.copy("./metadata.csv", os.path.join(noise_dir, "metadata.csv"))
    
    # 应用噪声（保持原逻辑）
    apply_noise(
        strength_factor=current_strength,
        input_image_path="./benchmark/benchmark_coco_filtered/images",
        output_image_path=os.path.join(noise_dir, "images"),
        example_image_path=f"examples/GaussianNoise_{current_strength:.1f}",
        base_sigma=20,
        max_sigma=150,
        num_workers=8,
        png_compression=0
    )

    # 执行测试
    answers_file = f"./answers/TEST_TOP5_answers.jsonl"
    evaluate_model(
        model_path="./cambrian-phi3-3b",
        dataset_path=noise_dir,
        output_path=answers_file,
        temperature=0.7,
        max_new_tokens=2048
    )

except Exception as e:
    print(f"测试失败: {str(e)}")
finally:
    torch.cuda.empty_cache()
