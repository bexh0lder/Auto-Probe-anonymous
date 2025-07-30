# evaluate_llava1.6_pope.py

# Import required libraries
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

# ### Core change     print(f" - Positive Questions ('yes' questions): {basic_stats.get('positive_questions_total', 0)}")##: Import LLaVA 1.6 related modules
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="LLaVA 1.6 evaluation on POPE dataset")
    
    # ### Core change ###: Model parameters adapted for LLaVA 1.6
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLaVA 1.6 model path")
    
    # Dataset related parameters
    parser.add_argument("--question_file", type=str, required=True,
                        help="POPE dataset question JSON file path")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Image folder path")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature parameter (recommend lower values for evaluation)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    
    # Output parameters
    parser.add_argument("--output_file", type=str, default="llava1.6_pope_results.json",
                        help="Output detailed results JSON file path")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="CSV file path for recording evaluation result summary (optional)")
    
    return parser.parse_args()

def load_questions(question_file):
    """
    Load POPE dataset questions
    """
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    if isinstance(questions, list):
        return questions
    # Compatible with some file formats that may contain metadata
    elif isinstance(questions, dict) and "results" in questions:
        return questions["results"]
    else:
        print("Warning: Unknown data format")
        return []

# ### Core change ###: New model initialization function for LLaVA 1.6
def initialize_llava_model(args):
    """
    Initialize LLaVA 1.6 model
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

# ### Core change ###: Adapted question processing function for LLaVA 1.6
def process_yes_no_question(question_item, args, processor, model):
    """
    Process yes/no questions - adapted for LLaVA 1.6
    """
    image_file = question_item["image"]
    question_text = question_item.get("text", question_item.get("question", ""))
    label = question_item["label"]
    
    image_path = os.path.join(args.image_dir, image_file)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image file does not exist: {image_path}")
        return {**question_item, "prediction": "ERROR: Image file does not exist", "correct": False, "predicted_answer": "ERROR"}

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Warning: Cannot load image {image_path}: {e}")
        return {**question_item, "prediction": f"ERROR: Cannot load image: {str(e)}", "correct": False, "predicted_answer": "ERROR"}

    # Build conversational prompt for LLaVA 1.6, hardcode system_prompt to ensure evaluation consistency
    system_prompt = 'You are an assistant who must answer the question with only "YES" or "NO".'
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": question_text}, {"type": "image"}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Prepare model input
    inputs = processor(text=prompt, images=image, return_tensors='pt').to(model.device)

    generate_kwargs = {
        'do_sample': args.temperature > 0,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': 10, # For Yes/No, a small number is sufficient
    }

    with torch.inference_mode():
        res = model.generate(**inputs, **generate_kwargs)
    
    # Key change: Only decode the generated part from the model
    input_token_len = inputs['input_ids'].shape[1]
    generated_tokens = res[0][input_token_len:]
    output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Answer parsing logic can be reused (very robust)
    prediction = output.lower()
    predicted_answer = ""
    
    if "yes" in prediction and "no" not in prediction:
        predicted_answer = "yes"
    elif "no" in prediction:
        predicted_answer = "no"
    else:
        # If answer is unclear, take conservative strategy, consider as not seen (no)
        predicted_answer = "no"
    
    correct = predicted_answer.lower() == label.lower()
    
    return {**question_item, "prediction": output, "predicted_answer": predicted_answer, "correct": correct}


# --- Statistics and CSV logging functions (identical to original version) ---
def calculate_statistics(results):
    """Calculate evaluation statistics"""
    stats = {
        "basic_questions": {
            "total_questions": len(results.get("basic_questions", [])),
            "positive_questions_total": 0, "negative_questions_total": 0,
            "negative_questions_triggered": 0, "negative_trigger_rate": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "error_count": 0,
        }
    }
    # Alias for convenient access
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
    """Record evaluation summary to specified CSV file"""
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
        print(f"Evaluation summary successfully appended to: {summary_file}")
    except Exception as e:
        print(f"Error: Unable to write CSV summary file {summary_file}: {e}")

def main():
    """
    Main function - responsible for complete testing, metric calculation and logging
    """
    args = parse_arguments()
    
    questions = load_questions(args.question_file)
    print(f"Loaded {len(questions)} questions from: {args.question_file}")
    
    print("Initializing LLaVA 1.6 model...")
    # Core modification: Call new model initialization function
    processor, model, model_name = initialize_llava_model(args)
    print(f"Model '{model_name}' initialization completed")
    
    results = []
    print(f"Starting to process questions ({len(questions)} total)...")
    for item in tqdm(questions, desc=f"Processing POPE dataset ({model_name})"):
        # Core modification: Call new question processing function
        result = process_yes_no_question(item, args, processor, model)
        results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results as JSON file
    output_data = {"basic_questions": results, "enhanced_questions": []}
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed test results saved to: {args.output_file}")

    # Calculate and print statistics
    print("\nCalculating evaluation metrics...")
    stats = calculate_statistics(output_data)
    basic_stats = stats.get("basic_questions", {})

    print("\n================== POPE Evaluation Results (LLaVA 1.6) ==================")
    print(f" Model Name: {model_name}")
    print("---------------------------------------------------------")
    print(f" - Total Questions: {basic_stats.get('total_questions', 0)}")
    print(f" - Positive Questions ('yes' questions): {basic_stats.get('positive_questions_total', 0)}")
    print(f" - Negative Questions ('no' questions): {basic_stats.get('negative_questions_total', 0)} (for hallucination testing)")
    print(f" - Negative Questions Triggered (Hallucinations): {basic_stats.get('negative_questions_triggered', 0)}")
    print(f" - Negative Questions Triggered Rate (Hallucination Rate): {basic_stats.get('negative_trigger_rate', 0):.2%}")
    print("---------------------------------------------------------")
    print(f" - Accuracy: {basic_stats.get('accuracy', 0):.2%}")
    print(f" - Precision: {basic_stats.get('precision', 0):.2%}")
    print(f" - Recall: {basic_stats.get('recall', 0):.2%}")
    print(f" - F1 Score: {basic_stats.get('f1_score', 0):.2%}")
    print(f" - Error Count: {basic_stats.get('error_count', 0)}")
    print("=====================================================================")

    # If specified, log to CSV
    if args.summary_csv:
        log_summary_to_csv(args.summary_csv, stats, model_name)

if __name__ == "__main__":
    main()