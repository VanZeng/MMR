import os
import json
import csv
from datetime import datetime

def add_data_to_csv(file_path, data):
    """追加数据到CSV文件，自动处理表头"""
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def extract_mcq_answer(text):
    """从文本中提取单选题答案"""
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for keyword in answer_keywords:
        if keyword in text:
            text = text.split(keyword)[-1].strip(' .')
    
    # 处理带括号的情况
    text = text.strip('()[]{}')
    return text[0] if text else ''

def compute_metrics(noise, 
                   answers_file=None, 
                   output_file=None, 
                   csv_file=None, 
                   extra_outdir=None,
                   image_path_field='image'):  # 默认字段改为image_id
    """
    准确率计算函数（字段兼容版）
    
    参数：
    noise : str - 噪声类型标识
    answers_file : str - 答案文件路径（JSONL格式）
    output_file : str - 错误记录保存路径（CSV格式）
    csv_file : str - 评估结果保存路径（CSV格式）
    extra_outdir : str - 备份目录路径
    image_path_field : str - 图像路径字段名（根据数据实际情况配置）
    """
    # 设置默认路径
    answers_file = answers_file or f"./answers/answers_{noise}_0.jsonl"
    output_file = output_file or f"./incorrect/incorrect_{noise}.csv"
    csv_file = csv_file or "./experiments.csv"
    extra_outdir = extra_outdir or "./backup_results"

    # 数据预验证
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("答案文件为空")
            
            sample_data = json.loads(first_line)
            required_fields = ['model_id', 'category', 'answer', 'gt_answer', image_path_field]
            
            # 字段存在性检查
            missing_fields = [field for field in required_fields if field not in sample_data]
            if missing_fields:
                available_fields = ", ".join(sample_data.keys())
                raise KeyError(
                    f"缺少必需字段: {', '.join(missing_fields)}\n"
                    f"当前可用字段: {available_fields}\n"
                    f"请通过 image_path_field 参数指定正确的图像路径字段名"
                )
    except Exception as e:
        print(f"数据验证失败: {str(e)}")
        return

    # 初始化元数据
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        "model": "",
        "categories": set(),
        "category_metrics": {}
    }

    # 处理错误记录
    with open(answers_file, 'r', encoding='utf-8') as f_in:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            # 动态生成CSV列名
            fieldnames = ['model_id', 'category', 'answer', 'gt_answer', image_path_field]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for line in f_in:
                data = json.loads(line.strip())
                
                # 获取图像路径（兼容不同字段名）
                try:
                    img_path = data[image_path_field]
                except KeyError:
                    img_path = "路径缺失"
                
                # 收集元数据
                metrics["model"] = data.get('model_id', metrics["model"])
                category = data.get('category', '未知分类')
                metrics["categories"].add(category)

                # 初始化分类指标
                if category not in metrics["category_metrics"]:
                    metrics["category_metrics"][category] = {'matches': 0, 'total': 0}
                metrics["category_metrics"][category]['total'] += 1

                # 答案比对
                answer = extract_mcq_answer(data.get('answer', ''))
                gt_answer = data.get('gt_answer', '').lower().strip('()')[0]  # 处理括号格式
                
                if answer == gt_answer:
                    metrics["category_metrics"][category]['matches'] += 1
                else:
                    # 写入错误记录
                    writer.writerow({
                        'model_id': data.get('model_id'),
                        'category': category,
                        'answer': data.get('answer'),
                        'gt_answer': data.get('gt_answer'),
                        image_path_field: img_path
                    })

    # 计算结果指标
    total_matches = sum(m['matches'] for m in metrics["category_metrics"].values())
    total_count = sum(m['total'] for m in metrics["category_metrics"].values())
    
    combined_data = {
        "模型": metrics["model"],
        "时间": time_string,
        "噪声类型": noise,
        "总体准确率": round(total_matches / total_count, 4),
        "总题数": total_count
    }

    # 添加分类指标
    for category in metrics["categories"]:
        m = metrics["category_metrics"][category]
        combined_data[f"{category}_准确率"] = round(m['matches'] / m['total'], 4)
        combined_data[f"{category}_题数"] = m['total']

    # 保存结果
    add_data_to_csv(csv_file, combined_data)
    print(f"✅ 评估结果已保存至: {csv_file}")

    # 备份处理
    if extra_outdir:
        backup_name = f"accuracy_{noise}_{current_time.strftime('%Y%m%d')}.csv"
        backup_path = os.path.join(extra_outdir, backup_name)
        os.makedirs(extra_outdir, exist_ok=True)
        add_data_to_csv(backup_path, combined_data)
        print(f"🔒 备份文件已生成: {backup_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="通用准确率计算工具")
    parser.add_argument("--noise", required=True, help="噪声类型标识")
    parser.add_argument("--answers_file", help="答案文件路径")
    parser.add_argument("--output_file", help="错误记录输出路径")
    parser.add_argument("--csv_file", help="结果文件路径")
    parser.add_argument("--extra_outdir", help="备份目录路径")
    parser.add_argument("--image_field", default="image", 
                      help="图像路径字段名（默认：file_name）")
    
    args = parser.parse_args()
    
    compute_metrics(
        noise=args.noise,
        answers_file=args.answers_file,
        output_file=args.output_file,
        csv_file=args.csv_file,
        extra_outdir=args.extra_outdir,
        image_path_field=args.image_field
    )