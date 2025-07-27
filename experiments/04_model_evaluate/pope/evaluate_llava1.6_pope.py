# evaluate_llava1.6_pope.py

# 导入所需的库
import argparse
import json
import os
import time
import torch
import sys
import csv
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime

# ### 核心改动 ###: 导入 LLaVA 1.6 相关模块
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="LLaVA 1.6 在POPE数据集上的评估")
    
    # ### 核心改动 ###: 模型参数适配 LLaVA 1.6
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLaVA 1.6 模型路径")
    
    # 数据集相关参数
    parser.add_argument("--question_file", type=str, required=True,
                        help="POPE数据集问题JSON文件路径")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图像文件夹路径")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="生成温度参数 (评估时建议使用较低值)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p采样参数")
    
    # 输出参数
    parser.add_argument("--output_file", type=str, default="llava1.6_pope_results.json",
                        help="输出详细结果的JSON文件路径")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="用于记录评估结果摘要的CSV文件路径 (可选)")
    
    return parser.parse_args()

def load_questions(question_file):
    """
    加载POPE数据集问题
    """
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    if isinstance(questions, list):
        return questions
    # 兼容一些可能包含元数据的文件格式
    elif isinstance(questions, dict) and "results" in questions:
        return questions["results"]
    else:
        print("警告: 未知的数据格式")
        return []

# ### 核心改动 ###: 新的模型初始化函数 for LLaVA 1.6
def initialize_llava_model(args):
    """
    初始化 LLaVA 1.6 模型
    """
    model_name = os.path.basename(args.model_path.strip('/'))
    processor = LlavaNextProcessor.from_pretrained(args.model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    return processor, model, model_name

# ### 核心改动 ###: 适配 LLaVA 1.6 的问题处理函数
def process_yes_no_question(question_item, args, processor, model):
    """
    处理判断题（是/否问题） - 适配 LLaVA 1.6
    """
    image_file = question_item["image"]
    question_text = question_item.get("text", question_item.get("question", ""))
    label = question_item["label"]
    
    image_path = os.path.join(args.image_dir, image_file)
    
    if not os.path.exists(image_path):
        print(f"警告: 图像文件不存在: {image_path}")
        return {**question_item, "prediction": "ERROR: 图像文件不存在", "correct": False, "predicted_answer": "ERROR"}

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图像 {image_path}: {e}")
        return {**question_item, "prediction": f"ERROR: 无法加载图像: {str(e)}", "correct": False, "predicted_answer": "ERROR"}

    # 为 LLaVA 1.6 构建对话式 Prompt，硬编码 system_prompt 以保证评估一致性
    system_prompt = 'You are an assistant who must answer the question with only "YES" or "NO".'
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": question_text}, {"type": "image"}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # 准备模型输入
    inputs = processor(text=prompt, images=image, return_tensors='pt').to(model.device)

    generate_kwargs = {
        'do_sample': args.temperature > 0,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': 10, # For Yes/No, a small number is sufficient
    }

    with torch.inference_mode():
        res = model.generate(**inputs, **generate_kwargs)
    
    # 关键改动：只解码模型生成的部分
    input_token_len = inputs['input_ids'].shape[1]
    generated_tokens = res[0][input_token_len:]
    output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 解析答案的逻辑可以复用 (非常鲁棒)
    prediction = output.lower()
    predicted_answer = ""
    
    if "yes" in prediction and "no" not in prediction:
        predicted_answer = "yes"
    elif "no" in prediction:
        predicted_answer = "no"
    else:
        # 如果答案不明确，采取保守策略，视为未看见（no）
        predicted_answer = "no"
    
    correct = predicted_answer.lower() == label.lower()
    
    return {**question_item, "prediction": output, "predicted_answer": predicted_answer, "correct": correct}


# --- 统计和CSV日志函数 (与原版完全相同) ---
def calculate_statistics(results):
    """计算评估统计信息"""
    stats = {
        "basic_questions": {
            "total_questions": len(results.get("basic_questions", [])),
            "positive_questions_total": 0, "negative_questions_total": 0,
            "negative_questions_triggered": 0, "negative_trigger_rate": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "error_count": 0,
        }
    }
    # 别名，方便访问
    basic_stats = stats["basic_questions"]
    tp, fp, tn, fn = 0, 0, 0, 0

    for q in results.get("basic_questions", []):
        if q.get("predicted_answer") == "ERROR": 
            basic_stats["error_count"] += 1
            continue
        true_label = q.get("label", "").lower()
        predicted_answer = q.get("predicted_answer", "").lower()
        if true_label == "yes" and predicted_answer == "yes": tp += 1
        elif true_label == "no" and predicted_answer == "yes": fp += 1
        elif true_label == "no" and predicted_answer == "no": tn += 1
        elif true_label == "yes" and predicted_answer == "no": fn += 1

    basic_stats["positive_questions_total"] = tp + fn
    basic_stats["negative_questions_total"] = tn + fp
    basic_stats["negative_questions_triggered"] = fp
    
    if basic_stats["negative_questions_total"] > 0: 
        basic_stats["negative_trigger_rate"] = fp / basic_stats["negative_questions_total"]
    
    valid_total = tp + fp + tn + fn
    if valid_total > 0: 
        basic_stats["accuracy"] = (tp + tn) / valid_total
    
    if (tp + fp) > 0: 
        basic_stats["precision"] = tp / (tp + fp)
    
    if (tp + fn) > 0: 
        basic_stats["recall"] = tp / (tp + fn)
    
    if (basic_stats["precision"] + basic_stats["recall"]) > 0:
        basic_stats["f1_score"] = 2 * (basic_stats["precision"] * basic_stats["recall"]) / (basic_stats["precision"] + basic_stats["recall"])
    
    return {"basic_questions": basic_stats}

def log_summary_to_csv(summary_file, stats, model_name):
    """将评估摘要记录到指定的CSV文件中"""
    try:
        row_data = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for key, value in stats.get('basic_questions', {}).items():
            row_data[f'pope_{key}'] = f"{value:.4f}" if isinstance(value, float) else value
        header = list(row_data.keys())
        file_exists = os.path.isfile(summary_file)
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists: writer.writeheader()
            writer.writerow(row_data)
        print(f"评估摘要已成功追加到: {summary_file}")
    except Exception as e:
        print(f"错误: 无法写入CSV摘要文件 {summary_file}: {e}")

def main():
    """
    主函数 - 负责完整的测试、指标计算和日志记录
    """
    args = parse_arguments()
    
    questions = load_questions(args.question_file)
    print(f"加载了 {len(questions)} 个问题从: {args.question_file}")
    
    print("正在初始化 LLaVA 1.6 模型...")
    # ### 核心改动 ###: 调用新的模型初始化函数
    processor, model, model_name = initialize_llava_model(args)
    print(f"模型 '{model_name}' 初始化完成")
    
    results = []
    print(f"开始处理问题 ({len(questions)}个)...")
    for item in tqdm(questions, desc=f"处理POPE数据集 ({model_name})"):
        # ### 核心改动 ###: 调用新的问题处理函数
        result = process_yes_no_question(item, args, processor, model)
        results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 将结果保存为JSON文件
    output_data = {"basic_questions": results, "enhanced_questions": []}
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细测试结果已保存到: {args.output_file}")

    # 计算并打印统计指标
    print("\n正在计算评估指标...")
    stats = calculate_statistics(output_data)
    basic_stats = stats.get("basic_questions", {})

    print("\n================== POPE 评 估 统 计 结 果 (LLaVA 1.6) ==================")
    print(f" 模型名称: {model_name}")
    print("---------------------------------------------------------")
    print(f" - 总评估问题数: {basic_stats.get('total_questions', 0)}")
    print(f" - 正例问题数 ('yes' questions): {basic_stats.get('positive_questions_total', 0)}")
    print(f" - 负例问题数 ('no' questions): {basic_stats.get('negative_questions_total', 0)} (用于测试幻觉)")
    print(f" - 负例问题触发数 (Hallucinations): {basic_stats.get('negative_questions_triggered', 0)}")
    print(f" - 负例问题触发率 (Hallucination Rate): {basic_stats.get('negative_trigger_rate', 0):.2%}")
    print("---------------------------------------------------------")
    print(f" - Accuracy: {basic_stats.get('accuracy', 0):.2%}")
    print(f" - Precision: {basic_stats.get('precision', 0):.2%}")
    print(f" - Recall: {basic_stats.get('recall', 0):.2%}")
    print(f" - F1 Score: {basic_stats.get('f1_score', 0):.2%}")
    print(f" - 出错问题数: {basic_stats.get('error_count', 0)}")
    print("=====================================================================")

    # 如果指定，则记录到CSV
    if args.summary_csv:
        log_summary_to_csv(args.summary_csv, stats, model_name)

if __name__ == "__main__":
    main()