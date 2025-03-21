import os
import shutil
import random
import pandas as pd
import ast
import json
import torch
from GaussianNoise import apply_noise
from accuracy import compute_metrics
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token
)
from cambrian.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from cambrian.conversation import conv_templates
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import glob

############################ 参数设置 ############################
# 实验核心参数
MODEL_PATH = "./cambrian-phi3-3b"          
CLEAN_IMAGE_DIR = "./benchmark/benchmark_coco_filtered/images"  
INIT_STRENGTH = 1.2                        
MAX_STRENGTH = 3.0                         
STEP_SIZE = 0.2                            
RESULT_CSV = "./results/robustness.csv"    

# 模型生成参数
GENERATION_TEMP = 0.2     
TOP_P = 0.9               
NUM_BEAMS = 1             
MAX_TOKENS = 512          

# 实验控制参数
CONV_MODE = "phi3"        
TOLERANCE = 0.05          
RANDOM_SEED = 42          

# 噪声生成参数
BASE_SIGMA = 20           
MAX_SIGMA = 150           
SIGMA_VARIATION = 0.15    
GAMMA = 1.5               
NUM_EXAMPLES = 5          
#################################################################

# 数据收集结构
experiment_data = {
    'strengths': [],
    'overall_acc': [],
    'count_acc': [],
    'location_acc': []
}

def init_baseline():
    """计算随机基准性能"""
    df = pd.read_csv('metadata.csv', engine='python', quotechar='"')
    df['choices'] = df['choices'].apply(ast.literal_eval)
    df['correct'] = df['answer'].str.strip('()')
    
    random.seed(RANDOM_SEED)
    df['pred'] = df['choices'].apply(lambda x: random.choice([chr(65+i) for i in range(len(x))]))
    df['correct'] = df['pred'] == df['correct']
    
    return {
        'overall': df['correct'].mean(),
        'count': df[df.sub_task == 'Count']['correct'].mean(),
        'location': df[df.sub_task == 'Relative location']['correct'].mean()
    }

random_baseline = init_baseline()

def build_prompt(line, model_config):
    """构建符合对话模板的提示"""
    qs = f"{line['prompt']}\nAnswer with the option's letter from the given choices directly."
    
    # 处理图像标记
    if line["image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"
    
    # 应用对话模板
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def process_sample(line, dataset_path, tokenizer, processor, model_config):
    """处理单个样本（修复图像路径）"""
    # 构建提示
    prompt = build_prompt(line, model_config)
    
    # 图像处理（关键修改：从噪声数据集加载图像）
    image_tensor = image_size = None
    if line["image"] is not None:
        # 从当前噪声数据集目录加载图像
        image_dir = os.path.join(dataset_path, "images")
        image_path = os.path.join(image_dir, line["image_id"])
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], processor, model_config)
        image_size = [image.size]
    
    # Tokenization
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    return input_ids, image_tensor, image_size, prompt

def evaluate(dataset_path):
    """执行模型评估"""
    # 模型初始化
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, 
        model_base=None,
        model_name=model_name
    )
    
    # 数据加载
    dataset = load_dataset(dataset_path, split="train")
    output_file = f"./results/answers_{os.path.basename(dataset_path)}.jsonl"
    
    # 结果收集（修复字段）
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            inputs, images, img_sizes, prompt = process_sample(
                sample, dataset_path,  # 传递当前数据集路径
                tokenizer, processor, model.config
            )
            
            with torch.inference_mode():
                outputs = model.generate(
                    inputs,
                    images=images,
                    image_sizes=img_sizes,
                    do_sample=GENERATION_TEMP > 0,
                    temperature=GENERATION_TEMP,
                    top_p=TOP_P,
                    num_beams=NUM_BEAMS,
                    max_new_tokens=MAX_TOKENS,
                    use_cache=True
                )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            results.append({
                "model_id": model_name,
                "image_id": sample["image_id"],
                "answer": answer,
                "gt_answer": sample["answer"],
                "category": sample["sub_task"],
                "options": sample["choices"]
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    # 资源清理
    del model, tokenizer, processor
    torch.cuda.empty_cache()
    return output_file

def noise_experiment():
    """噪声鲁棒性实验主流程（修复版本）"""
    current_strength = INIT_STRENGTH
    torch.manual_seed(RANDOM_SEED)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        while current_strength <= MAX_STRENGTH:
            # 创建噪声数据集目录
            noise_dir = f"./benchmark/Gaussian_{current_strength:.1f}"
            example_dir = f"./examples/Gaussian_{current_strength:.1f}"
            os.makedirs(noise_dir, exist_ok=True)
            os.makedirs(example_dir, exist_ok=True)
            
            # 复制元数据（确保指向噪声图像）
            shutil.copy('metadata.csv', os.path.join(noise_dir, 'metadata.csv'))
            
            # 应用噪声到新目录（关键路径修正）
            apply_noise(
                strength_factor=current_strength,
                input_image_path=CLEAN_IMAGE_DIR,
                output_image_path=os.path.join(noise_dir, "images"),  # 正确输出路径
                example_image_path=example_dir,
                base_sigma=BASE_SIGMA,
                max_sigma=MAX_SIGMA,
                sigma_variation=SIGMA_VARIATION,
                gamma=GAMMA,
                random_seed=RANDOM_SEED,
                num_examples=NUM_EXAMPLES,
                num_workers=8,
                png_compression=0
            )
            
            # 执行评估（使用带噪声的数据集）
            result_file = evaluate(noise_dir)
            
            # ============== 指标计算 ==============
            strength_str = f"{current_strength:.1f}".replace('.', '_')
            metrics_csv = f"./results/metrics/Gaussian_{strength_str}_{start_time}.csv"
            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            
            # 计算指标
            compute_metrics(
                noise=f"Gaussian_{current_strength:.1f}",
                answers_file=result_file,
                output_file=f"./incorrect/incorrect_Gaussian_{strength_str}.csv",
                csv_file=metrics_csv,
                image_path_field='image_id',
                extra_outdir="./backup/metrics"
            )
            
            # 读取指标数据
            try:
                metrics_df = pd.read_csv(metrics_csv)
                latest = metrics_df.iloc[-1]
                metrics = {
                    'overall': latest['总体准确率'],
                    'count': latest.get('Count_准确率', 0.0),
                    'location': latest.get('Relative location_准确率', 0.0)
                }
            except Exception as e:
                print(f"⚠️ 指标读取失败: {str(e)}")
                metrics = {'overall': 0.0, 'count': 0.0, 'location': 0.0}
            
            # ============== 数据记录 ==============
            experiment_data['strengths'].append(current_strength)
            experiment_data['overall_acc'].append(metrics['overall'])
            experiment_data['count_acc'].append(metrics['count'])
            experiment_data['location_acc'].append(metrics['location'])
            
            # 打印状态
            print(f"\n{' NOISE LEVEL ':=^60}")
            print(f"| {'参数':<20} | {'值':<36} |")
            print(f"| {'-'*18} | {'-'*34} |")
            print(f"| 噪声强度        | {current_strength:<10.1f} ({current_strength/MAX_STRENGTH:.0%}) |")
            print(f"| 总体准确率      | {metrics['overall']:>10.2%} |")
            print(f"| 计数准确率      | {metrics['count']:>10.2%} |")
            print(f"| 定位准确率      | {metrics['location']:>10.2%} |")
            print(f"| 随机基准线      | {random_baseline['overall']:>10.2%} |")
            print(f"{'':=^60}")
            
            # 终止条件
            if metrics['overall'] <= (random_baseline['overall'] + TOLERANCE):
                print(f"\n🔴 实验终止：当前准确率 {metrics['overall']:.2%} ≤ 基准线 {random_baseline['overall']:.2%} + 容忍度 {TOLERANCE:.0%}")
                break
                
            current_strength = round(current_strength + STEP_SIZE, 1)
            
    except KeyboardInterrupt:
        print("\n🟡 实验被用户中断")
    except Exception as e:
        print(f"\n🔴 严重错误: {str(e)}")
        raise
    finally:
        # 保存最终结果
        os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
        pd.DataFrame(experiment_data).to_csv(RESULT_CSV, index=False)
        
        # 生成带时间戳的备份
        backup_dir = f"./results/backups/{start_time}"
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy(RESULT_CSV, os.path.join(backup_dir, os.path.basename(RESULT_CSV)))
        
        print(f"\n✅ 实验数据已保存至: {os.path.abspath(RESULT_CSV)}")
        print(f"🔄 已创建备份副本: {os.path.abspath(backup_dir)}")
        torch.cuda.empty_cache()

def clear_previous_noise_data():
    """安全清理之前的噪声数据（保留原始数据）"""
    # 仅删除Gaussian_开头的目录
    noise_dirs = glob.glob(os.path.join("./benchmark", "Gaussian_*"))
    for dir_path in noise_dirs:
        if os.path.isdir(dir_path):
            print(f"清理旧数据: {os.path.abspath(dir_path)}")
            shutil.rmtree(dir_path)

if __name__ == "__main__":
    # 前置检查
    required = {
        'metadata.csv': "元数据文件",
        CLEAN_IMAGE_DIR: "干净图像目录"
    }
    for path, desc in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc}不存在: {os.path.abspath(path)}")

    # 初始化环境
    torch.set_float32_matmul_precision('high')
    disable_torch_init()
    
    # 清理历史数据
    clear_previous_noise_data()
    
    # 执行实验
    noise_experiment()