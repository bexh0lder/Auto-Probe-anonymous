# evaluate_instructblip.py
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
import copy # --- NEW/MODIFIED --- (导入copy模块以支持深度复制)

# 添加必要的路径
sys.path.append("/root/Auto-Probe")
sys.path.append(os.path.dirname(os.path.abspath(".")))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("."))))

from lavis.models import load_model_and_preprocess


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在TIDE数据集上评估InstructBLIP模型")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_instructblip_evaluation.yaml",
        help="YAML配置文件路径"
    )
    # 其余CLI参数
    parser.add_argument("--tide_dataset_file", type=str, default=None, help="TIDE数据集文件路径 (覆盖YAML配置)")
    parser.add_argument("--image_dir", type=str, default=None, help="图像目录路径 (覆盖YAML配置)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录路径 (覆盖YAML配置)")
    parser.add_argument("--output_filename", type=str, default=None, help="输出文件名 (覆盖YAML配置)")
    parser.add_argument("--model_name_cli", type=str, default=None, help="模型名称 (覆盖YAML配置)")
    parser.add_argument("--model_type_cli", type=str, default=None, help="模型类型 (覆盖YAML配置)")

    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="用于记录评估结果摘要的CSV文件路径 (覆盖YAML配置)"
    )

    # --- NEW/MODIFIED START ---
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="评估轮数，用于多轮评估以获取更稳定的幻觉率。默认为1轮。此参数只能通过命令行设置。"
    )
    # --- NEW/MODIFIED END ---

    args = parser.parse_args()
    
    if args.config:
        args = load_and_merge_config(args)
    else:
        raise ValueError("请提供YAML配置文件路径，使用 --config 参数")
    
    return args

def load_and_merge_config(cli_args):
    """从YAML加载并合并配置，CLI优先"""
    config_path = cli_args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config.get('04_model_evaluate') is None:
        print(f"FATAL: '04_model_evaluate' not found in config '{config_path}'."); sys.exit(1)
    config = config.get('04_model_evaluate')

    args = argparse.Namespace(**vars(cli_args))

    # 模型配置
    model_config = config.get('model', {})
    args.model_name = cli_args.model_name_cli if cli_args.model_name_cli else model_config.get('name', 'blip2_vicuna_instruct')
    args.model_type = cli_args.model_type_cli if cli_args.model_type_cli else model_config.get('type', 'vicuna7b')

    # 数据配置
    data_config = config.get('data', {})
    args.tide_dataset_file = cli_args.tide_dataset_file if cli_args.tide_dataset_file else data_config.get('tide_dataset_file')
    args.image_dir = cli_args.image_dir if cli_args.image_dir else data_config.get('image_dir')
    args.output_dir = cli_args.output_dir if cli_args.output_dir else data_config.get('output_dir')
    args.output_filename = cli_args.output_filename if cli_args.output_filename else data_config.get('output_filename', 'tide_evaluation_results.json')
    
    # 推理配置
    inference_config = config.get('inference', {})
    args.system_prompt = inference_config.get('system_prompt', '')
    args.temperature = inference_config.get('temperature', 0.1)
    args.top_p = inference_config.get('top_p', 0.9)
    args.max_tokens = inference_config.get('max_tokens', 128)
    
    # 其他配置
    other_config = config.get('other', {})
    args.device = other_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    args.seed = other_config.get('seed', 42)
    args.log_level = other_config.get('log_level', 'INFO')
    args.use_timestamp = other_config.get('use_timestamp', True)
    
    args.summary_csv = cli_args.summary_csv if cli_args.summary_csv else other_config.get('summary_csv', None)
    
    args.config_file = config_path
    
    return args

def validate_args(args, logger=None):
    """验证必需的参数"""
    required_fields = {
        'model_name': '模型名称 (name)',
        'model_type': '模型类型 (type)',
        'tide_dataset_file': 'TIDE数据集文件路径', 
        'image_dir': '图像目录路径',
        'output_dir': '输出目录路径'
    }
    
    missing_fields = [f"{field} ({desc})" for field, desc in required_fields.items() if not getattr(args, field, None)]
    
    if missing_fields:
        error_msg = f"以下必需参数未指定: {', '.join(missing_fields)}"
        if logger: logger.critical(error_msg)
        else: print(f"ERROR: {error_msg}")
        return False
    
    return True

def setup_logger(output_dir, log_level='INFO'):
    """设置日志记录器"""
    logger = logging.getLogger("tide_instructblip_evaluator")
    if logger.hasHandlers(): logger.handlers.clear()
    
    try:
        numeric_level = getattr(logging, log_level.upper())
    except AttributeError:
        numeric_level = logging.INFO
        print(f"WARNING: 无效的日志级别 '{log_level}'，使用INFO级别")
    logger.setLevel(numeric_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "tide_evaluation_instructblip.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_output_directory(base_dir, model_name, use_timestamp=True, logger=None):
    """创建输出目录"""
    safe_model_name = model_name.replace("/", "_")
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(base_dir, f"tide_eval_{safe_model_name}_{timestamp}")
    else:
        result_dir = os.path.join(base_dir, f"tide_eval_{safe_model_name}")
    
    os.makedirs(result_dir, exist_ok=True)
    if logger: logger.info(f"创建输出目录: {result_dir}")
    
    return result_dir

def load_tide_dataset(tide_file_path, logger=None):
    """加载TIDE数据集"""
    if not os.path.exists(tide_file_path):
        raise FileNotFoundError(f"TIDE数据集文件不存在: {tide_file_path}")
    
    with open(tide_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    basic_questions = data.get("basic_questions", [])
    enhanced_questions = data.get("enhanced_questions", [])
    
    if logger:
        logger.info(f"从 {tide_file_path} 加载了 {len(basic_questions)} 个基础问题, {len(enhanced_questions)} 个增强问题")
    
    return basic_questions, enhanced_questions

def ask_instructblip_safe(prompt_text, image_tensor, model, args, logger=None):
    """安全地调用InstructBLIP模型进行推理"""
    try:
        inputs = {"image": image_tensor, "prompt": prompt_text}
        
        generate_kwargs = {
            "use_nucleus_sampling": args.top_p is not None,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "max_length": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
        
        with torch.inference_mode():
            outputs = model.generate(inputs, **generate_kwargs)
        
        raw_output = outputs[0] if isinstance(outputs, list) and len(outputs) > 0 else ""
        return raw_output.strip()
        
    except Exception as e:
        if logger:
            logger.error(f"InstructBLIP推理过程中出错: {e}", exc_info=True)
        return f"ERROR: {str(e)}"

def process_basic_question(question_item, image_dir, model, vis_processors, args, logger=None):
    """处理基础Yes/No问题 (适配InstructBLIP)"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: 缺少图像文件名", "predicted_answer": "ERROR", "correct": False}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: 图像文件不存在", "predicted_answer": "ERROR", "correct": False}
    
    try:
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)
        
        question_text = question_item.get("text", "")
        instruction = 'Answer with only "YES" or "NO".'
        full_prompt = f"{args.system_prompt} {question_text} {instruction}".strip()
        
        raw_output = ask_instructblip_safe(full_prompt, image_tensor, model, args, logger)
        
        label = question_item.get("label", "").lower()
        
        if raw_output.startswith("ERROR:"):
            predicted_answer = "ERROR"
        else:
            output_lower = raw_output.lower().strip()
            if output_lower.startswith("yes"):
                predicted_answer = "yes"
            elif output_lower.startswith("no"):
                predicted_answer = "no"
            else:
                predicted_answer = "no" if label == "yes" else "yes"

        correct = (predicted_answer == label) if predicted_answer != "ERROR" else False
        
        return {**question_item, "prediction": raw_output, "predicted_answer": predicted_answer, "correct": correct}
        
    except Exception as e:
        if logger:
            logger.error(f"处理基础问题时出错 (图像: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": "ERROR", "correct": False}

def process_enhanced_question(question_item, image_dir, model, vis_processors, args, logger=None):
    """处理增强选择题问题 (适配InstructBLIP) - 开放式回答版本"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: 缺少图像文件名", "predicted_answer": None, "correct": None}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: 图像文件不存在", "predicted_answer": None, "correct": None}
    
    try:
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)
        
        question_text = question_item.get("text", "")
        # 开放式提示词
        full_prompt = f"{args.system_prompt} {question_text}".strip()

        raw_output = ask_instructblip_safe(full_prompt, image_tensor, model, args, logger)

        if raw_output.startswith("ERROR:"):
            return {**question_item, "prediction": raw_output, "predicted_answer": None, "correct": None}
        else:
            return {**question_item, "prediction": raw_output, "predicted_answer": None, "correct": None}
        
    except Exception as e:
        if logger:
            logger.error(f"处理增强问题时出错 (图像: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": None, "correct": None}

def process_questions(basic_questions, enhanced_questions, image_dir, model, vis_processors, args, logger=None):
    """处理所有问题"""
    start_time = time.time()
    
    all_results = {"basic_questions": [], "enhanced_questions": []}
    
    if basic_questions:
        logger.info(f"开始处理 {len(basic_questions)} 个基础问题...")
        for item in tqdm(basic_questions, desc="处理基础问题"):
            result = process_basic_question(item, image_dir, model, vis_processors, args, logger)
            all_results["basic_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if enhanced_questions:
        logger.info(f"开始处理 {len(enhanced_questions)} 个增强问题...")
        for item in tqdm(enhanced_questions, desc="处理增强问题"):
            result = process_enhanced_question(item, image_dir, model, vis_processors, args, logger)
            all_results["enhanced_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    logger.info(f"单轮问题处理完成，耗时: {total_time:.2f} 秒")
    
    return all_results

def calculate_statistics(results, logger=None):
    """计算评估统计信息"""
    stats = {
        "basic_questions": {
            "total_questions": len(results.get("basic_questions", [])),
            "positive_questions_total": 0, "negative_questions_total": 0,
            "negative_questions_triggered": 0, "negative_trigger_rate": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "error_count": 0,
            "_true_positives": 0, "_false_positives": 0, "_true_negatives": 0, "_false_negatives": 0,
        },
        "enhanced_questions": {
            "total_questions": len(results.get("enhanced_questions", [])),
            "accuracy": 0.0, "correct_answers": 0, "error_count": 0,
        }
    }

    basic_stats = stats["basic_questions"]
    # --- NEW/MODIFIED START ---
    # 为了保证多轮评估的统计正确性，必须基于最终聚合的 'correct' 字段来计算指标，
    # 而不是依赖最后一轮的 'predicted_answer'。
    for q in results.get("basic_questions", []):
        if q.get("predicted_answer") == "ERROR":
            basic_stats["error_count"] += 1
            continue
            
        true_label = q.get("label", "").lower()
        is_correct = q.get("correct", False) # 使用聚合后的最终正确性状态

        if true_label == 'yes':
            if is_correct:
                basic_stats["_true_positives"] += 1 # 正确回答 Yes (TP)
            else:
                basic_stats["_false_negatives"] += 1 # 错误回答 No (FN)
        else: # true_label == 'no'
            if is_correct:
                basic_stats["_true_negatives"] += 1 # 正确回答 No (TN)
            else:
                basic_stats["_false_positives"] += 1 # 错误回答 Yes (FP, 幻觉)
    # --- NEW/MODIFIED END ---
    
    tp = basic_stats["_true_positives"]; fp = basic_stats["_false_positives"]
    tn = basic_stats["_true_negatives"]; fn = basic_stats["_false_negatives"]
    basic_stats["positive_questions_total"] = tp + fn
    basic_stats["negative_questions_total"] = tn + fp
    basic_stats["negative_questions_triggered"] = fp
    if basic_stats["negative_questions_total"] > 0:
        basic_stats["negative_trigger_rate"] = fp / basic_stats["negative_questions_total"]
    valid_total = tp + fp + tn + fn
    if valid_total > 0: basic_stats["accuracy"] = (tp + tn) / valid_total
    if (tp + fp) > 0: basic_stats["precision"] = tp / (tp + fp)
    if (tp + fn) > 0: basic_stats["recall"] = tp / (tp + fn)
    if (basic_stats["precision"] + basic_stats["recall"]) > 0:
        basic_stats["f1_score"] = 2 * (basic_stats["precision"] * basic_stats["recall"]) / (basic_stats["precision"] + basic_stats["recall"])

    enhanced_stats = stats["enhanced_questions"]
    for q in results.get("enhanced_questions", []):
        if q.get("predicted_answer") == "ERROR": enhanced_stats["error_count"] += 1
        elif q.get("correct", False): enhanced_stats["correct_answers"] += 1
    valid_enhanced = enhanced_stats["total_questions"] - enhanced_stats["error_count"]
    if valid_enhanced > 0: enhanced_stats["accuracy"] = enhanced_stats["correct_answers"] / valid_enhanced

    if logger:
        # --- NEW/MODIFIED START ---
        logger.info("================== 最终聚合评估统计结果 ==================")
        # --- NEW/MODIFIED END ---
        logger.info("--- 基础问题 (Basic Questions) ---")
        logger.info(f"  - 总评估问题数: {basic_stats['total_questions']}")
        logger.info(f"  - 正例问题数: {basic_stats['positive_questions_total']}")
        logger.info(f"  - 负例问题数: {basic_stats['negative_questions_total']} (用于测试幻觉)")
        logger.info(f"  - 负例问题触发数 (幻觉次数): {basic_stats['negative_questions_triggered']}")
        logger.info(f"  - 负例问题触发率 (幻觉率): {basic_stats['negative_trigger_rate']:.2%}")
        logger.info(f"  ----------------------------------------------------")
        logger.info(f"  - Accuracy: {basic_stats['accuracy']:.2%}")
        logger.info(f"  - Precision: {basic_stats['precision']:.2%}")
        logger.info(f"  - Recall: {basic_stats['recall']:.2%}")
        logger.info(f"  - F1 Score: {basic_stats['f1_score']:.2%}")
        logger.info(f"  - 出错问题数: {basic_stats['error_count']}")
        logger.info("--- 增强问题 (Enhanced Questions) ---")
        logger.info(f"  - 总评估问题数: {enhanced_stats['total_questions']}")
        logger.info(f"  - Accuracy: {enhanced_stats['accuracy']:.2%}")
        logger.info(f"  - 出错问题数: {enhanced_stats['error_count']}")
        logger.info("==========================================================")
    
    for key in ["_true_positives", "_false_positives", "_true_negatives", "_false_negatives"]:
        del basic_stats[key]
        
    return stats

def log_summary_to_csv(summary_file, stats, model_name, logger=None):
    """将评估摘要记录到指定的CSV文件中"""
    try:
        row_data = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in stats.get('basic_questions', {}).items():
            row_data[f'basic_{key}'] = f"{value:.4f}" if isinstance(value, float) else value
            
        for key, value in stats.get('enhanced_questions', {}).items():
            row_data[f'enhanced_{key}'] = f"{value:.4f}" if isinstance(value, float) else value

        header = list(row_data.keys())
        
        file_exists = os.path.isfile(summary_file)
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(row_data)
            
        if logger:
            logger.info(f"评估摘要已成功追加到: {summary_file}")

    except IOError as e:
        if logger:
            logger.error(f"无法写入CSV摘要文件 {summary_file}: {e}", exc_info=True)
    except Exception as e:
        if logger:
            logger.error(f"记录CSV摘要时发生未知错误: {e}", exc_info=True)


def main():
    """主函数"""
    script_start_time = time.time()
    logger = None
    output_dir = "tide_evaluation_output"
    
    try:
        args = parse_args()
        
        if not validate_args(args):
            sys.exit(1)
        
        model_identifier = f"{args.model_name}-{args.model_type}"
        output_dir = create_output_directory(args.output_dir, model_identifier, args.use_timestamp)
        
        logger = setup_logger(output_dir, args.log_level)
        
        logger.info("TIDE数据集 InstructBLIP 评估开始")
        logger.info(f"配置文件: {args.config_file}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"评估模型: {args.model_name} (类型: {args.model_type})")
        # --- NEW/MODIFIED START ---
        logger.info(f"评估轮数 (Epochs): {args.epoch}")
        # --- NEW/MODIFIED END ---
        
        logger.info("开始加载InstructBLIP模型...")
        model_load_start = time.time()
        
        try:
            import subprocess
            result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
            output = result.stdout
            for line in output.splitlines():
                if '=' in line:
                    var, value = line.split('=', 1)
                    os.environ[var] = value
        except Exception:
            pass
            
        model, vis_processors, _ = load_model_and_preprocess(
            name=args.model_name,
            model_type=args.model_type,
            is_eval=True,
            device=args.device
        )
        logger.info(f"模型加载完成，耗时: {time.time() - model_load_start:.2f} 秒。")

        basic_questions, enhanced_questions = load_tide_dataset(args.tide_dataset_file, logger)
        
        config_to_save = {k: v for k, v in vars(args).items()}
        with open(os.path.join(output_dir, "evaluation_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        
        # --- NEW/MODIFIED START ---
        final_aggregated_results = None

        for epoch_num in range(1, args.epoch + 1):
            logger.info(f"--- 开始第 {epoch_num}/{args.epoch} 轮评估 ---")
            
            current_epoch_results = process_questions(
                basic_questions, enhanced_questions, args.image_dir,
                model, vis_processors, args, logger
            )
            
            if final_aggregated_results is None:
                final_aggregated_results = copy.deepcopy(current_epoch_results)
            else:
                # 聚合基础问题
                for i, res in enumerate(current_epoch_results["basic_questions"]):
                    if not res["correct"]:
                        final_aggregated_results["basic_questions"][i]["correct"] = False

                # 聚合增强问题
                for i, res in enumerate(current_epoch_results["enhanced_questions"]):
                    if not res["correct"]:
                        final_aggregated_results["enhanced_questions"][i]["correct"] = False
        
        logger.info(f"--- 所有 {args.epoch} 轮评估完成，开始计算最终聚合统计数据 ---")

        stats = calculate_statistics(final_aggregated_results, logger)
        
        if args.summary_csv:
            log_summary_to_csv(args.summary_csv, stats, model_identifier, logger)
        
        final_results = {
            "statistics": stats,
            "model_info": {
                "model_name": args.model_name,
                "model_type": args.model_type,
                "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_run": args.epoch,
            },
            "detailed_results": final_aggregated_results
        }
        # --- NEW/MODIFIED END ---
        
        output_file = os.path.join(output_dir, args.output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最终聚合评估结果已保存到: {output_file}")
        
    except Exception as e:
        if logger: logger.error(f"评估主流程出错: {str(e)}", exc_info=True)
        else: print(f"评估主流程出错: {str(e)}")
        import traceback; traceback.print_exc()
        
    finally:
        total_script_time = time.time() - script_start_time
        if logger: logger.info(f"TIDE评估流程结束，总耗时: {total_script_time:.2f} 秒")
        if 'output_dir' in locals() and os.path.exists(output_dir):
            print(f"OUTPUT_DIR={output_dir}")

if __name__ == "__main__":
    main()