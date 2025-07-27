# evaluate_llava_1.6.py
import argparse
import time
import torch
import os
import sys
import json
import yaml
import logging
import re
import csv
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ### 核心改动 ###: 导入 LLaVA 1.6 相关模块
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def parse_args():
    """
    解析命令行参数，支持CLI覆盖YAML配置
    """
    parser = argparse.ArgumentParser(description="在TIDE数据集上评估 LLaVA 1.6 模型")
    
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--tide_dataset_file", type=str, default=None, help="TIDE数据集文件路径 (覆盖YAML)")
    parser.add_argument("--image_dir", type=str, default=None, help="图像目录路径 (覆盖YAML)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录路径 (覆盖YAML)")
    parser.add_argument("--output_filename", type=str, default=None, help="输出文件名 (覆盖YAML)")
    parser.add_argument("--summary_csv", type=str, default=None, help="用于记录评估结果摘要的CSV文件路径 (覆盖YAML)")
    
    # ### 核心改动 ###: 模型参数适配 LLaVA 1.6
    parser.add_argument("--model_path", type=str, default=None, help="模型路径 (覆盖YAML)")
    
    args = parser.parse_args()
    
    if args.config:
        args = load_and_merge_config(args)
    else:
        raise ValueError("请提供YAML配置文件路径，使用 --config 参数")
    
    return args

def load_and_merge_config(cli_args):
    """
    从YAML配置文件加载参数，并与CLI参数合并（CLI优先）- 适配 LLaVA 1.6
    """
    config_path = cli_args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 假设评估配置在 '04_model_evaluate' 键下
    config = config_data.get('04_model_evaluate')
    if config is None:
        print(f"FATAL: '04_model_evaluate' not found in config '{config_path}'."); sys.exit(1)

    args = argparse.Namespace(**vars(cli_args))

    # 模型配置
    model_config = config.get('model', {})
    args.model_path = cli_args.model_path if cli_args.model_path else model_config.get('path')
    args.model_name = model_config.get('name', os.path.basename(args.model_path) if args.model_path else 'llava_1.6_model')
    
    # 数据配置
    data_config = config.get('data', {})
    args.tide_dataset_file = cli_args.tide_dataset_file if cli_args.tide_dataset_file else data_config.get('tide_dataset_file')
    args.image_dir = cli_args.image_dir if cli_args.image_dir else data_config.get('image_dir')
    args.output_dir = cli_args.output_dir if cli_args.output_dir else data_config.get('output_dir')
    args.output_filename = cli_args.output_filename if cli_args.output_filename else data_config.get('output_filename', 'tide_evaluation_results.json')
    
    # 推理配置
    inference_config = config.get('inference', {})
    args.temperature = inference_config.get('temperature', 0.2) # LLaVA 默认值通常较低
    args.top_p = inference_config.get('top_p', 1.0)
    args.max_tokens = inference_config.get('max_tokens', 128)
    
    # 其他配置
    other_config = config.get('other', {})
    args.device = other_config.get('device', 'cuda')
    args.seed = other_config.get('seed', 42)
    args.log_level = other_config.get('log_level', 'INFO')
    args.use_timestamp = other_config.get('use_timestamp', True)
    args.summary_csv = cli_args.summary_csv if cli_args.summary_csv else other_config.get('summary_csv', None)
    
    args.config_file = config_path
    return args

def validate_args(args, logger=None):
    """验证必需的参数"""
    required_fields = {
        'model_path': '模型路径', 'tide_dataset_file': 'TIDE数据集文件路径', 
        'image_dir': '图像目录路径', 'output_dir': '输出目录路径'
    }
    missing_fields = [f"{desc} ({field})" for field, desc in required_fields.items() if not getattr(args, field, None)]
    
    if missing_fields:
        error_msg = f"以下必需参数未指定: {', '.join(missing_fields)}"
        if logger: logger.critical(error_msg)
        else: print(f"ERROR: {error_msg}")
        return False
    return True

# --- 日志和目录创建函数 ---
def setup_logger(output_dir, log_level='INFO'):
    logger = logging.getLogger("tide_llava_1_6_evaluator") # 改名
    if logger.hasHandlers(): logger.handlers.clear()
    
    try:
        numeric_level = getattr(logging, log_level.upper())
        logger.setLevel(numeric_level)
    except AttributeError:
        logger.setLevel(logging.INFO)
        print(f"WARNING: 无效的日志级别 '{log_level}'，使用INFO级别")
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "tide_evaluation_llava_1_6.log") # 改名
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def create_output_directory(base_dir, model_name, use_timestamp=True, logger=None):
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(base_dir, f"tide_eval_{model_name}_{timestamp}")
    else:
        result_dir = os.path.join(base_dir, f"tide_eval_{model_name}")
    os.makedirs(result_dir, exist_ok=True)
    if logger: logger.info(f"创建输出目录: {result_dir}")
    return result_dir

def load_tide_dataset(tide_file_path, logger=None):
    """加载TIDE数据集"""
    if not os.path.exists(tide_file_path):
        raise FileNotFoundError(f"TIDE数据集文件不存在: {tide_file_path}")
    
    with open(tide_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    basic_questions = data.get("basic_questions", [])
    enhanced_questions = data.get("enhanced_questions", [])
    
    if logger:
        logger.info(f"从 {tide_file_path} 加载了 {len(basic_questions)} 个基础问题, {len(enhanced_questions)} 个增强问题")
        logger.info(f"总计: {len(basic_questions) + len(enhanced_questions)} 个问题")
    return basic_questions, enhanced_questions

# --- 核心改动：新的推理函数 for LLaVA 1.6 ---
def ask_llava_safe(prompt_text, image, model, processor, args, logger=None):
    """
    安全地调用 LLaVA 1.6 模型进行推理
    """
    try:
        inputs = processor(text=prompt_text, images=image, return_tensors='pt').to(model.device)

        generate_kwargs = {
            'do_sample': args.temperature > 0,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_new_tokens': args.max_tokens,
        }

        with torch.inference_mode():
            res = model.generate(**inputs, **generate_kwargs)
        
        # 关键改动: 只解码模型生成的部分，而非全部
        input_token_len = inputs['input_ids'].shape[1]
        generated_tokens = res[0][input_token_len:]
        raw_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return raw_output
        
    except Exception as e:
        if logger: logger.error(f"LLaVA 1.6 推理过程中出错: {e}", exc_info=True)
        return f"ERROR: {str(e)}"

# --- 核心改动：适配 LLaVA 1.6 的问题处理函数 ---
def process_basic_question(question_item, image_dir, model, processor, args, logger=None):
    """处理基础Yes/No问题 (适配 LLaVA 1.6)"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: 缺少图像文件名", "predicted_answer": "ERROR", "correct": False}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: 图像文件不存在", "predicted_answer": "ERROR", "correct": False}
    
    try:
        image = Image.open(image_path).convert('RGB')
        question_text = question_item.get("text", "")
        label = question_item.get("label", "").lower()
        
        # 为 LLaVA 1.6 构建对话式 Prompt
        system_prompt = "You are an assistant who answers questions about an image. Respond with only the single word 'YES' or 'NO'."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": question_text + "Respond with only the single word 'YES' or 'NO'."}, {"type": "image"}]}
        ]
        prompt_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        raw_output = ask_llava_safe(prompt_for_model, image, model, processor, args, logger)
        
        # 答案解析逻辑 (与原版相同，因为它是模型无关的)
        if raw_output.startswith("ERROR:"):
            predicted_answer = "ERROR"
        else:
            output_lower = raw_output.lower().strip()
            if output_lower.startswith("yes"):
                predicted_answer = "yes"
            elif output_lower.startswith("no"):
                predicted_answer = "no"
            else:
                # 关键修改：如果无法解析，则视为回答错误。
                predicted_answer = "no" if label == "yes" else "yes"

        correct = (predicted_answer == label) if predicted_answer != "ERROR" else False
        return {**question_item, "prediction": raw_output, "predicted_answer": predicted_answer, "correct": correct}
        
    except Exception as e:
        if logger: logger.error(f"处理基础问题时出错 (图像: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": "ERROR", "correct": False}

def process_enhanced_question(question_item, image_dir, model, processor, args, logger=None):
    """处理增强选择题问题 (适配 LLaVA 1.6)"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: 缺少图像文件名", "predicted_answer": "ERROR", "correct": False}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: 图像文件不存在", "predicted_answer": "ERROR", "correct": False}
    
    try:
        image = Image.open(image_path).convert('RGB')
        question_text = question_item.get("text", "")
        options = question_item.get("options", {})
        label = question_item.get("label", "").upper()
        
        # 修改1：从 ABC 改为 ABCD
        option_str = "\n".join([f"{key}. {value}" for key, value in options.items() if key in ['A', 'B', 'C', 'D']])
        full_question_text = f"{question_text}\nOptions:\n{option_str}"
        
        # 修改2：系统提示词中也要改为 ABCD
        system_prompt = "You are an assistant who answers multiple-choice questions about an image. Respond with only the letter of the correct option (A, B, C, or D)."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            # 修改3：用户提示词中也要改为 ABCD
            {"role": "user", "content": [{"type": "text", "text": full_question_text + "\n\nAnswer with only the letter (A, B, C, or D)."}, {"type": "image"}]}
        ]
        prompt_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        raw_output = ask_llava_safe(prompt_for_model, image, model, processor, args, logger)
        
        # 答案解析逻辑
        if raw_output.startswith("ERROR:"):
            predicted_option = "ERROR"
        else:
            predicted_option = "X"
            output_upper = raw_output.upper().strip()
            # 修改4：正则表达式从 [ABC] 改为 [ABCD]
            first_char_match = re.match(r"^\s*([ABCD])", output_upper)
            if first_char_match:
                predicted_option = first_char_match.group(1)
            else:
                # 修改5：这里也要改为 [ABCD]
                match = re.search(r"\b([ABCD])\b", output_upper)
                if match:
                    predicted_option = match.group(1)

        correct = (predicted_option == label) if predicted_option != "ERROR" else False
        return {**question_item, "prediction": raw_output, "predicted_answer": predicted_option, "correct": correct}
        
    except Exception as e:
        if logger: logger.error(f"处理增强问题时出错 (图像: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": "ERROR", "correct": False}

# ### 核心改动 ###: 更新函数签名以传递 LLaVA 1.6 相关对象
def process_questions(basic_questions, enhanced_questions, image_dir, model, processor, args, logger=None):
    """处理所有问题"""
    start_time = time.time()
    all_results = {"basic_questions": [], "enhanced_questions": []}
    
    if basic_questions:
        if logger: logger.info(f"开始处理 {len(basic_questions)} 个基础问题...")
        for item in tqdm(basic_questions, desc="处理基础问题"):
            result = process_basic_question(item, image_dir, model, processor, args, logger)
            all_results["basic_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if enhanced_questions:
        if logger: logger.info(f"开始处理 {len(enhanced_questions)} 个增强问题...")
        for item in tqdm(enhanced_questions, desc="处理增强问题"):
            result = process_enhanced_question(item, image_dir, model, processor, args, logger)
            all_results["enhanced_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    if logger: logger.info(f"所有问题处理完成，总耗时: {total_time:.2f} 秒")
    return all_results

# --- 统计和CSV日志函数 (与原版完全相同) ---
def calculate_statistics(results, logger=None):
    # (此函数无需修改，逻辑与模型无关)
    stats = {
        "basic_questions": {"total_questions": len(results.get("basic_questions", [])),"positive_questions_total": 0,"negative_questions_total": 0,"negative_questions_triggered": 0,"negative_trigger_rate": 0.0,"accuracy": 0.0,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"error_count": 0,},
        "enhanced_questions": {"total_questions": len(results.get("enhanced_questions", [])),"accuracy": 0.0,"correct_answers": 0,"error_count": 0,}
    }
    basic_stats = stats["basic_questions"]
    tp, fp, tn, fn = 0, 0, 0, 0
    for q in results.get("basic_questions", []):
        if q.get("predicted_answer") == "ERROR": basic_stats["error_count"] += 1; continue
        true_label = q.get("label", "").lower(); predicted_answer = q.get("predicted_answer", "").lower()
        if true_label == "yes" and predicted_answer == "yes": tp += 1
        elif true_label == "no" and predicted_answer == "yes": fp += 1
        elif true_label == "no" and predicted_answer == "no": tn += 1
        elif true_label == "yes" and predicted_answer == "no": fn += 1
    basic_stats["positive_questions_total"] = tp + fn; basic_stats["negative_questions_total"] = tn + fp; basic_stats["negative_questions_triggered"] = fp
    if basic_stats["negative_questions_total"] > 0: basic_stats["negative_trigger_rate"] = fp / basic_stats["negative_questions_total"]
    valid_total = tp + fp + tn + fn
    if valid_total > 0: basic_stats["accuracy"] = (tp + tn) / valid_total
    if (tp + fp) > 0: basic_stats["precision"] = tp / (tp + fp)
    if (tp + fn) > 0: basic_stats["recall"] = tp / (tp + fn)
    if (basic_stats["precision"] + basic_stats["recall"]) > 0: basic_stats["f1_score"] = 2 * (basic_stats["precision"] * basic_stats["recall"]) / (basic_stats["precision"] + basic_stats["recall"])
    enhanced_stats = stats["enhanced_questions"]
    for q in results.get("enhanced_questions", []):
        if q.get("predicted_answer") == "ERROR": enhanced_stats["error_count"] += 1
        elif q.get("correct", False): enhanced_stats["correct_answers"] += 1
    valid_enhanced = enhanced_stats["total_questions"] - enhanced_stats["error_count"]
    if valid_enhanced > 0: enhanced_stats["accuracy"] = enhanced_stats["correct_answers"] / valid_enhanced
    if logger:
        logger.info("================== 评 估 统 计 结 果 =================="); logger.info("--- 基础问题 (Basic Questions) ---")
        logger.info(f"  - 总评估问题数: {basic_stats['total_questions']}"); logger.info(f"  - 正例问题数: {basic_stats['positive_questions_total']}")
        logger.info(f"  - 负例问题数: {basic_stats['negative_questions_total']} (用于测试幻觉)"); logger.info(f"  - 负例问题触发数: {basic_stats['negative_questions_triggered']} (模型产生幻觉的次数)")
        logger.info(f"  - 负例问题触发率: {basic_stats['negative_trigger_rate']:.2%}"); logger.info(f"  ----------------------------------------------------")
        logger.info(f"  - Accuracy: {basic_stats['accuracy']:.2%}"); logger.info(f"  - Precision: {basic_stats['precision']:.2%}"); logger.info(f"  - Recall: {basic_stats['recall']:.2%}"); logger.info(f"  - F1 Score: {basic_stats['f1_score']:.2%}"); logger.info(f"  - 出错问题数: {basic_stats['error_count']}")
        logger.info("--- 增强问题 (Enhanced Questions) ---"); logger.info(f"  - 总评估问题数: {enhanced_stats['total_questions']}")
        logger.info(f"  - Accuracy: {enhanced_stats['accuracy']:.2%}"); logger.info(f"  - 出错问题数: {enhanced_stats['error_count']}"); logger.info("==========================================================")
    return stats

def log_summary_to_csv(summary_file, stats, model_name, logger=None):
    # (此函数无需修改，逻辑与模型无关)
    try:
        row_data = {'model_name': model_name, 'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for key, value in stats.get('basic_questions', {}).items(): row_data[f'basic_{key}'] = f"{value:.4f}" if isinstance(value, float) else value
        for key, value in stats.get('enhanced_questions', {}).items(): row_data[f'enhanced_{key}'] = f"{value:.4f}" if isinstance(value, float) else value
        header = list(row_data.keys())
        file_exists = os.path.isfile(summary_file)
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists: writer.writeheader()
            writer.writerow(row_data)
        if logger: logger.info(f"评估摘要已成功追加到: {summary_file}")
    except Exception as e:
        if logger: logger.error(f"写入CSV摘要文件时发生错误: {e}", exc_info=True)

def main():
    """主函数"""
    script_start_time = time.time()
    
    try:
        args = parse_args()
        if not validate_args(args): sys.exit(1)
        
        output_dir = create_output_directory(args.output_dir, args.model_name, use_timestamp=args.use_timestamp)
        logger = setup_logger(output_dir, args.log_level)
        
        logger.info("TIDE数据集 LLaVA 1.6 评估开始")
        logger.info(f"配置文件: {args.config_file}")
        logger.info(f"输出目录: {output_dir}")
        for arg, value in sorted(vars(args).items()): logger.debug(f"参数 {arg}: {value}")
        
        torch.cuda.empty_cache()
        
        # ### 核心改动：加载 LLaVA 1.6 模型 ###
        logger.info("开始加载 LLaVA 1.6 模型...")
        model_load_start = time.time()
        
        processor = LlavaNextProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to(args.device)
        logger.info(f"模型加载完成，耗时: {time.time() - model_load_start:.2f} 秒。")
        
        basic_questions, enhanced_questions = load_tide_dataset(args.tide_dataset_file, logger)
        
        config_to_save = {k: v for k, v in vars(args).items()}
        config_to_save["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(output_dir, "evaluation_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        
        # ### 核心改动 ###: 传递 LLaVA 1.6 相关对象
        results = process_questions(
            basic_questions, enhanced_questions, args.image_dir,
            model, processor, args, logger
        )
        
        stats = calculate_statistics(results, logger)
        
        if args.summary_csv:
            log_summary_to_csv(args.summary_csv, stats, args.model_name, logger)

        final_results = {
            "statistics": stats,
            "model_info": {"model_path": args.model_path, "model_name": args.model_name, "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "detailed_results": results
        }
        
        output_file = os.path.join(output_dir, args.output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_file}")
        
    except Exception as e:
        logging.error(f"评估主流程出错: {str(e)}", exc_info=True)
        print(f"评估主流程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        total_script_time = time.time() - script_start_time
        logging.info(f"TIDE评估流程结束，总耗时: {total_script_time:.2f} 秒")
        if 'output_dir' in locals():
            print(f"OUTPUT_DIR={output_dir}")

if __name__ == "__main__":
    main()