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

# Import LLaVA 1.6 related modules
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def parse_args():
    """Parse command line arguments with support for CLI override of YAML config."""
    parser = argparse.ArgumentParser(description="Evaluate LLaVA 1.6 model on TIDE dataset")
    
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--tide_dataset_file", type=str, default=None, help="TIDE dataset file path (overrides YAML)")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory path (overrides YAML)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path (overrides YAML)")
    parser.add_argument("--output_filename", type=str, default=None, help="Output filename (overrides YAML)")
    parser.add_argument("--summary_csv", type=str, default=None, help="CSV file path for evaluation result summary (overrides YAML)")
    
    # Model parameters adapted for LLaVA 1.6
    parser.add_argument("--model_path", type=str, default=None, help="Model path (overrides YAML)")
    
    args = parser.parse_args()
    
    if args.config:
        args = load_and_merge_config(args)
    else:
        raise ValueError("Please provide YAML configuration file path using --config parameter")
    
    return args

def load_and_merge_config(cli_args):
    """Load parameters from YAML config file and merge with CLI args (CLI takes priority) - adapted for LLaVA 1.6."""
    config_path = cli_args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Assume evaluation configuration is under '04_model_evaluate' key
    config = config_data.get('04_model_evaluate')
    if config is None:
        print(f"FATAL: '04_model_evaluate' not found in config '{config_path}'."); sys.exit(1)

    args = argparse.Namespace(**vars(cli_args))

    # Model configuration
    model_config = config.get('model', {})
    args.model_path = cli_args.model_path if cli_args.model_path else model_config.get('path')
    args.model_name = model_config.get('name', os.path.basename(args.model_path) if args.model_path else 'llava_1.6_model')
    
    # Data configuration
    data_config = config.get('data', {})
    args.tide_dataset_file = cli_args.tide_dataset_file if cli_args.tide_dataset_file else data_config.get('tide_dataset_file')
    args.image_dir = cli_args.image_dir if cli_args.image_dir else data_config.get('image_dir')
    args.output_dir = cli_args.output_dir if cli_args.output_dir else data_config.get('output_dir')
    args.output_filename = cli_args.output_filename if cli_args.output_filename else data_config.get('output_filename', 'tide_evaluation_results.json')
    
    # Inference configuration
    inference_config = config.get('inference', {})
    args.temperature = inference_config.get('temperature', 0.2) # LLaVA default values are usually lower
    args.top_p = inference_config.get('top_p', 1.0)
    args.max_tokens = inference_config.get('max_tokens', 128)
    
    # Other configuration
    other_config = config.get('other', {})
    args.device = other_config.get('device', 'cuda')
    args.seed = other_config.get('seed', 42)
    args.log_level = other_config.get('log_level', 'INFO')
    args.use_timestamp = other_config.get('use_timestamp', True)
    args.summary_csv = cli_args.summary_csv if cli_args.summary_csv else other_config.get('summary_csv', None)
    
    args.config_file = config_path
    return args

def validate_args(args, logger=None):
    """Validate required parameters"""
    required_fields = {
        'model_path': 'Model path', 'tide_dataset_file': 'TIDE dataset file path', 
        'image_dir': 'Image directory path', 'output_dir': 'Output directory path'
    }
    missing_fields = [f"{desc} ({field})" for field, desc in required_fields.items() if not getattr(args, field, None)]
    
    if missing_fields:
        error_msg = f"The following required parameters are not specified: {', '.join(missing_fields)}"
        if logger: logger.critical(error_msg)
        else: print(f"ERROR: {error_msg}")
        return False
    return True

# --- Logging and directory creation functions ---
def setup_logger(output_dir, log_level='INFO'):
    logger = logging.getLogger("tide_llava_1_6_evaluator") # renamed
    if logger.hasHandlers(): logger.handlers.clear()
    
    try:
        numeric_level = getattr(logging, log_level.upper())
        logger.setLevel(numeric_level)
    except AttributeError:
        logger.setLevel(logging.INFO)
        print(f"WARNING: Invalid log level '{log_level}', using INFO level")
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "tide_evaluation_llava_1_6.log") # renamed
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
    if logger: logger.info(f"Created output directory: {result_dir}")
    return result_dir

def load_tide_dataset(tide_file_path, logger=None):
    """Load TIDE dataset"""
    if not os.path.exists(tide_file_path):
        raise FileNotFoundError(f"TIDE dataset file does not exist: {tide_file_path}")
    
    with open(tide_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    basic_questions = data.get("basic_questions", [])
    enhanced_questions = data.get("enhanced_questions", [])
    
    if logger:
        logger.info(f"Loaded {len(basic_questions)} basic questions and {len(enhanced_questions)} enhanced questions from {tide_file_path}")
        logger.info(f"Total: {len(basic_questions) + len(enhanced_questions)} questions")
    return basic_questions, enhanced_questions

# --- Core modification: New inference function for LLaVA 1.6 ---
def ask_llava_safe(prompt_text, image, model, processor, args, logger=None):
    """
    Safely call LLaVA 1.6 model for inference
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
        
        # Key modification: Only decode the model-generated part, not the entire output
        input_token_len = inputs['input_ids'].shape[1]
        generated_tokens = res[0][input_token_len:]
        raw_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return raw_output
        
    except Exception as e:
        if logger: logger.error(f"Error during LLaVA 1.6 inference: {e}", exc_info=True)
        return f"ERROR: {str(e)}"

# --- Core modification: Question processing functions adapted for LLaVA 1.6 ---
def process_basic_question(question_item, image_dir, model, processor, args, logger=None):
    """Process basic Yes/No questions (adapted for LLaVA 1.6)"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: Missing image filename", "predicted_answer": "ERROR", "correct": False}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: Image file does not exist", "predicted_answer": "ERROR", "correct": False}
    
    try:
        image = Image.open(image_path).convert('RGB')
        question_text = question_item.get("text", "")
        label = question_item.get("label", "").lower()
        
        # Build conversational prompt for LLaVA 1.6
        system_prompt = "You are an assistant who answers questions about an image. Respond with only the single word 'YES' or 'NO'."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": question_text + "Respond with only the single word 'YES' or 'NO'."}, {"type": "image"}]}
        ]
        prompt_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        raw_output = ask_llava_safe(prompt_for_model, image, model, processor, args, logger)
        
        # Answer parsing logic (same as original version, as it's model-agnostic)
        if raw_output.startswith("ERROR:"):
            predicted_answer = "ERROR"
        else:
            output_lower = raw_output.lower().strip()
            if output_lower.startswith("yes"):
                predicted_answer = "yes"
            elif output_lower.startswith("no"):
                predicted_answer = "no"
            else:
                # Key modification: If unable to parse, treat as wrong answer.
                predicted_answer = "no" if label == "yes" else "yes"

        correct = (predicted_answer == label) if predicted_answer != "ERROR" else False
        return {**question_item, "prediction": raw_output, "predicted_answer": predicted_answer, "correct": correct}
        
    except Exception as e:
        if logger: logger.error(f"Error processing basic question (image: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": "ERROR", "correct": False}

def process_enhanced_question(question_item, image_dir, model, processor, args, logger=None):
    """Process enhanced multiple-choice questions (adapted for LLaVA 1.6)"""
    image_file = question_item.get("image")
    if not image_file:
        return {**question_item, "prediction": "ERROR: Missing image filename", "predicted_answer": "ERROR", "correct": False}
    
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        return {**question_item, "prediction": "ERROR: Image file does not exist", "predicted_answer": "ERROR", "correct": False}
    
    try:
        image = Image.open(image_path).convert('RGB')
        question_text = question_item.get("text", "")
        options = question_item.get("options", {})
        label = question_item.get("label", "").upper()
        
        # Modification 1: Change from ABC to ABCD
        option_str = "\n".join([f"{key}. {value}" for key, value in options.items() if key in ['A', 'B', 'C', 'D']])
        full_question_text = f"{question_text}\nOptions:\n{option_str}"
        
        # Modification 2: Also change system prompt to ABCD
        system_prompt = "You are an assistant who answers multiple-choice questions about an image. Respond with only the letter of the correct option (A, B, C, or D)."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            # Modification 3: Also change user prompt to ABCD
            {"role": "user", "content": [{"type": "text", "text": full_question_text + "\n\nAnswer with only the letter (A, B, C, or D)."}, {"type": "image"}]}
        ]
        prompt_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        raw_output = ask_llava_safe(prompt_for_model, image, model, processor, args, logger)
        
        # Answer parsing logic
        if raw_output.startswith("ERROR:"):
            predicted_option = "ERROR"
        else:
            predicted_option = "X"
            output_upper = raw_output.upper().strip()
            # Modification 4: Change regex from [ABC] to [ABCD]
            first_char_match = re.match(r"^\s*([ABCD])", output_upper)
            if first_char_match:
                predicted_option = first_char_match.group(1)
            else:
                # Modification 5: Also change this to [ABCD]
                match = re.search(r"\b([ABCD])\b", output_upper)
                if match:
                    predicted_option = match.group(1)

        correct = (predicted_option == label) if predicted_option != "ERROR" else False
        return {**question_item, "prediction": raw_output, "predicted_answer": predicted_option, "correct": correct}
        
    except Exception as e:
        if logger: logger.error(f"Error processing enhanced question (image: {image_file}): {e}", exc_info=True)
        return {**question_item, "prediction": f"ERROR: {str(e)}", "predicted_answer": "ERROR", "correct": False}

# ### Core modification ###: Update function signature to pass LLaVA 1.6 related objects
def process_questions(basic_questions, enhanced_questions, image_dir, model, processor, args, logger=None):
    """Process all questions"""
    start_time = time.time()
    all_results = {"basic_questions": [], "enhanced_questions": []}
    
    if basic_questions:
        if logger: logger.info(f"Starting to process {len(basic_questions)} basic questions...")
        for item in tqdm(basic_questions, desc="Processing basic questions"):
            result = process_basic_question(item, image_dir, model, processor, args, logger)
            all_results["basic_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if enhanced_questions:
        if logger: logger.info(f"Starting to process {len(enhanced_questions)} enhanced questions...")
        for item in tqdm(enhanced_questions, desc="Processing enhanced questions"):
            result = process_enhanced_question(item, image_dir, model, processor, args, logger)
            all_results["enhanced_questions"].append(result)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    if logger: logger.info(f"All questions processed, total time: {total_time:.2f} seconds")
    return all_results

# --- Statistics and CSV logging functions (completely same as original version) ---
def calculate_statistics(results, logger=None):
    # (This function needs no modification, logic is model-agnostic)
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
        logger.info("================== EVALUATION STATISTICS =================="); logger.info("--- Basic Questions ---")
        logger.info(f"  - Total evaluation questions: {basic_stats['total_questions']}"); logger.info(f"  - Positive questions: {basic_stats['positive_questions_total']}")
        logger.info(f"  - Negative questions: {basic_stats['negative_questions_total']} (for testing hallucination)"); logger.info(f"  - Negative questions triggered: {basic_stats['negative_questions_triggered']} (times model hallucinated)")
        logger.info(f"  - Negative trigger rate: {basic_stats['negative_trigger_rate']:.2%}"); logger.info(f"  ----------------------------------------------------")
        logger.info(f"  - Accuracy: {basic_stats['accuracy']:.2%}"); logger.info(f"  - Precision: {basic_stats['precision']:.2%}"); logger.info(f"  - Recall: {basic_stats['recall']:.2%}"); logger.info(f"  - F1 Score: {basic_stats['f1_score']:.2%}"); logger.info(f"  - Error questions: {basic_stats['error_count']}")
        logger.info("--- Enhanced Questions ---"); logger.info(f"  - Total evaluation questions: {enhanced_stats['total_questions']}")
        logger.info(f"  - Accuracy: {enhanced_stats['accuracy']:.2%}"); logger.info(f"  - Error questions: {enhanced_stats['error_count']}"); logger.info("==========================================================")
    return stats

def log_summary_to_csv(summary_file, stats, model_name, logger=None):
    # (This function needs no modification, logic is model-agnostic)
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
        if logger: logger.info(f"Evaluation summary successfully appended to: {summary_file}")
    except Exception as e:
        if logger: logger.error(f"Error writing CSV summary file: {e}", exc_info=True)

def main():
    """Main function"""
    script_start_time = time.time()
    
    try:
        args = parse_args()
        if not validate_args(args): sys.exit(1)
        
        output_dir = create_output_directory(args.output_dir, args.model_name, use_timestamp=args.use_timestamp)
        logger = setup_logger(output_dir, args.log_level)
        
        logger.info("TIDE dataset LLaVA 1.6 evaluation started")
        logger.info(f"Configuration file: {args.config_file}")
        logger.info(f"Output directory: {output_dir}")
        for arg, value in sorted(vars(args).items()): logger.debug(f"Parameter {arg}: {value}")
        
        torch.cuda.empty_cache()
        
        # ### Core modification: Load LLaVA 1.6 model ###
        logger.info("Starting to load LLaVA 1.6 model...")
        model_load_start = time.time()
        
        processor = LlavaNextProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to(args.device)
        logger.info(f"Model loading completed, time taken: {time.time() - model_load_start:.2f} seconds.")
        
        basic_questions, enhanced_questions = load_tide_dataset(args.tide_dataset_file, logger)
        
        config_to_save = {k: v for k, v in vars(args).items()}
        config_to_save["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(output_dir, "evaluation_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        
        # ### Core modification ###: Pass LLaVA 1.6 related objects
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
        logger.info(f"Evaluation results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main evaluation process: {str(e)}", exc_info=True)
        print(f"Error in main evaluation process: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        total_script_time = time.time() - script_start_time
        logging.info(f"TIDE evaluation process completed, total time: {total_script_time:.2f} seconds")
        if 'output_dir' in locals():
            print(f"OUTPUT_DIR={output_dir}")

if __name__ == "__main__":
    main()