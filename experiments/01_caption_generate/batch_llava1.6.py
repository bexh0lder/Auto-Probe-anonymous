import argparse
import time
import torch
import os
import sys
import json
import yaml
import random
import logging
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ### MODIFIED ###: 引入 LLaVA 1.6 的 transformers 库
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


# --- Helper functions ---

# ========================================================================================
# --- UNMODIFIED FUNCTION START ---
# 该函数大部分保留，仅更新了描述和一些模型特定的默认值
# ========================================================================================
def parse_args_and_config():
    """
    解析命令行参数和YAML配置文件。
    配置参数被拆分为 model, data, generation, inference, other 等部分。
    CLI参数可以覆盖YAML中的配置。
    """
    parser = argparse.ArgumentParser(description="使用 LLaVA 1.6 模型为一批图片生成描述或提取实体")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径。")
    
    # 添加CLI参数以覆盖YAML中的关键设置
    parser.add_argument("--model_path", type=str, default=None, help="覆盖YAML中的模型路径")
    parser.add_argument("--image_dir", type=str, default=None, help="覆盖YAML中的图像目录路径")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖YAML中的输出目录路径")
    
    cli_args = parser.parse_args()

    # --- 加载YAML文件 ---
    config_path = cli_args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 根键名可以保持不变，也可以按需修改
    config = config_data.get('01_caption_generate')
    if config is None:
        raise ValueError(f"在配置文件 '{config_path}' 中未找到 '01_caption_generate' 部分。")

    # --- 合并配置 (CLI优先) ---
    args = argparse.Namespace(**vars(cli_args))

    # --- 模型配置 ---
    model_config = config.get('model', {})
    args.model_path = cli_args.model_path if cli_args.model_path else model_config.get('path')
    args.model_name = model_config.get('name', os.path.basename(args.model_path) if args.model_path else 'unknown_model')

    # --- 数据配置 ---
    data_config = config.get('data', {})
    args.image_dir = cli_args.image_dir if cli_args.image_dir else data_config.get('image_dir')
    args.output_dir = cli_args.output_dir if cli_args.output_dir else data_config.get('output_dir')
    args.selected_images_file = data_config.get('selected_images_file', None)
    
    # --- 生成逻辑配置 ---
    generation_config = config.get('generation', {})
    args.num_images_to_process = generation_config.get('num_images_to_process', 200)
    args.num_epochs_per_prompt_set = generation_config.get('num_epochs_per_prompt_set', 1)
    args.json_parse_retries = generation_config.get('json_parse_retries', 3)
    
    # --- 推理配置 (适配 LLaVA 1.6) ---
    inference_config = config.get('inference', {})
    args.temperature = inference_config.get('temperature', 1.0)
    args.top_p = inference_config.get('top_p', 1.0)
    args.max_tokens = inference_config.get('max_tokens', 1024)
    
    # --- 其他配置 ---
    other_config = config.get('other', {})
    args.device = other_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    args.seed = other_config.get('seed', 42)
    args.log_level = other_config.get('log_level', 'INFO')

    # --- 提示词组合 (脚本特有) ---
    if 'prompt_sets' not in config:
        raise ValueError("YAML配置 '01_caption_generate' 部分必须包含 'prompt_sets'。")
    args.prompt_set_configs_raw = config['prompt_sets']
    
    # --- 记录与验证 ---
    args.config_file_path = config_path

    if not args.output_dir:
        raise ValueError("必须在YAML的'data'部分或通过--output_dir指定'output_dir'。")
    if not args.model_path:
        raise ValueError("必须在YAML的'model'部分或通过--model_path指定'model_path'。")
    if not args.image_dir:
        raise ValueError("必须在YAML的'data'部分或通过--image_dir指定'image_dir'。")

    return args

# ======================================================================================
# --- MODIFIED FUNCTION START ---
# 该函数被修改，不再强制要求 system_prompt 必须存在
# ======================================================================================
def load_and_validate_prompt_sets(raw_prompt_set_configs):
    loaded_prompt_sets = []
    if not isinstance(raw_prompt_set_configs, list): raise ValueError("'prompt_sets' must be a list.")
    
    for i, ps_config in enumerate(raw_prompt_set_configs):
        if not isinstance(ps_config, dict): raise ValueError(f"Item {i+1} in 'prompt_sets' must be a dict.")
        
        name = ps_config.get('name', f"prompt_set_{i+1}")
        # system_prompt 现在是可选的
        system_prompt = ps_config.get('system_prompt')
        user_prompts = ps_config.get('user_prompts')
        parse_json = ps_config.get('parse_json_output', False)

        # 移除了对 system_prompt 的强制检查

        if not user_prompts or not isinstance(user_prompts, list) or not all(isinstance(q, str) for q in user_prompts):
            raise ValueError(f"Prompt set '{name}' 必须包含一个字符串列表类型的 'user_prompts'。")
        if not isinstance(parse_json, bool): raise ValueError(f"'{name}': parse_json_output must be boolean.")
        
        loaded_prompt_sets.append({
            "name": name,
            "system_prompt": system_prompt, # 值为 None 或字符串
            "user_prompts": user_prompts,
            "parse_json_output": parse_json,
        })
    return loaded_prompt_sets


def setup_logger(output_dir_path, log_level_str):
    logger = logging.getLogger("llava_1_6_batch_process")
    if logger.hasHandlers(): logger.handlers.clear()
    try: level = getattr(logging, log_level_str.upper())
    except AttributeError: print(f"Warning: Invalid log_level '{log_level_str}'. Defaulting to INFO."); level = logging.INFO
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_file = os.path.join(output_dir_path, "batch_process_llava_1_6.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8'); file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(); console_handler.setFormatter(formatter)
    logger.addHandler(file_handler); logger.addHandler(console_handler)
    return logger

def select_random_images(image_dir, num_images, seed, logger=None):
    start_time = time.time()
    if logger: logger.info(f"Starting image selection from {image_dir} with seed: {seed}")
    random.seed(seed)
    all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if logger: logger.info(f"Found {len(all_image_files)} images in the directory.")

    if not all_image_files:
        if logger: logger.error(f"No supported image files found in {image_dir}.")
        return [], set()

    if len(all_image_files) > num_images:
        image_files = random.sample(all_image_files, num_images)
        if logger: logger.info(f"Randomly selected {len(image_files)} images out of {len(all_image_files)}.")
    else:
        image_files = all_image_files
        if logger: logger.info(f"Using all {len(image_files)} images from the directory (requested {num_images}).")
            
    selected_images_set = set(image_files)
    end_time = time.time()
    if logger: logger.info(f"Image selection completed in {end_time - start_time:.2f} seconds.")
    return image_files, selected_images_set


# ======================================================================================
# --- MODIFIED FUNCTION START ---
# 该函数被修改，以动态构建对话列表，从而支持可选的 system_prompt
# ======================================================================================
def process_images_for_all_prompt_sets(image_files, image_dir, model, processor, global_args, loaded_prompt_sets, current_output_dir, logger=None):
    master_results_for_individual_files = {ps_config["name"]: [] for ps_config in loaded_prompt_sets}
    
    consolidated_desc_by_image = {}
    consolidated_json_by_image = {}

    images_processed_successfully_count = 0
    images_with_catastrophic_errors_count = 0
    total_script_start_time = time.time()

    if logger: logger.info(f"Beginning processing for {len(image_files)} images across {len(loaded_prompt_sets)} prompt sets.")

    # 准备固定的生成参数
    generate_kwargs = {
        'do_sample': global_args.temperature > 0,
        'temperature': global_args.temperature,
        'top_p': global_args.top_p,
        'max_new_tokens': global_args.max_tokens,
    }
    if logger: logger.info(f"Generation parameters: {generate_kwargs}")

    for img_idx, img_file in enumerate(tqdm(image_files, desc="Processing Images Overall")):
        image_path = os.path.join(image_dir, img_file)
        if logger: logger.info(f"--- Processing Image {img_idx+1}/{len(image_files)}: {img_file} ---")
        
        current_image_had_catastrophic_error = False
        try:
            img_load_start_time = time.time()
            if logger: logger.debug(f"Loading image: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            if logger: logger.debug(f"Image loaded in {time.time() - img_load_start_time:.2f}s")

            for ps_config in loaded_prompt_sets:
                ps_name = ps_config["name"]
                is_json_method_ps = ps_config['parse_json_output']
                
                if logger: logger.info(f"  Processing image '{img_file}' with prompt set: '{ps_name}' (Method: {'entity2entity' if is_json_method_ps else 'caption2entity'})")

                captions_for_this_ps_image = []
                objects_for_this_ps_image_raw = []
                successful_epochs_this_ps = 0

                for epoch in range(global_args.num_epochs_per_prompt_set):
                    if logger: logger.debug(f"      Epoch {epoch+1}/{global_args.num_epochs_per_prompt_set}")
                    
                    for user_prompt in ps_config['user_prompts']:
                        # --- 核心修改：动态构建对话，支持可选的 system_prompt ---
                        conversation = [
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image"}
                            ]}
                        ]
                        system_prompt = ps_config.get('system_prompt')
                        if system_prompt:
                            system_part = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                            conversation.insert(0, system_part)
                        
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        if logger: logger.debug(f"        User Prompt: '{user_prompt}'")
                        
                        output_text_epoch = ""
                        
                        if is_json_method_ps:
                            parsed_successfully_this_epoch = False
                            for attempt in range(global_args.json_parse_retries):
                                with torch.no_grad():
                                    inputs = processor(text=prompt, images=pil_image, return_tensors='pt').to(model.device)
                                    res = model.generate(**inputs, **generate_kwargs)
                                    
                                    input_token_len = inputs['input_ids'].shape[1]
                                    generated_tokens = res[0][input_token_len:]
                                    output_text_epoch = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                                try:
                                    if output_text_epoch.strip().startswith("```json"):
                                        output_text_epoch = output_text_epoch.strip()[7:].strip()
                                        if output_text_epoch.endswith("```"):
                                            output_text_epoch = output_text_epoch[:-3].strip()
                                    
                                    caption_objects_epoch = json.loads(output_text_epoch)
                                    if not isinstance(caption_objects_epoch, list): raise ValueError("Parsed JSON not a list")
                                    if not all(isinstance(item, str) for item in caption_objects_epoch): raise ValueError("Parsed list has non-strings")
                                    
                                    captions_for_this_ps_image.append(output_text_epoch)
                                    objects_for_this_ps_image_raw.extend(caption_objects_epoch)
                                    successful_epochs_this_ps += 1
                                    parsed_successfully_this_epoch = True
                                    break
                                except (json.JSONDecodeError, ValueError) as e:
                                    if logger: logger.warning(f"JSON parse attempt {attempt+1} failed for {ps_name}. Error: {e}. Raw output: '{output_text_epoch}'")
                                    if attempt == global_args.json_parse_retries - 1:
                                        logger.error(f"Max JSON retries reached for {ps_name}, epoch {epoch+1}")
                            
                            if not parsed_successfully_this_epoch:
                                logger.warning(f"All JSON parse attempts failed for {ps_name}, epoch {epoch+1}")
                        
                        else: # caption2entity method
                            with torch.no_grad():
                                inputs = processor(text=prompt, images=pil_image, return_tensors='pt').to(model.device)
                                res = model.generate(**inputs, **generate_kwargs)

                                input_token_len = inputs['input_ids'].shape[1]
                                generated_tokens = res[0][input_token_len:]
                                output_text_epoch = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                            if output_text_epoch or output_text_epoch == "":
                                captions_for_this_ps_image.append(output_text_epoch)
                                successful_epochs_this_ps += 1
                            else:
                                logger.warning(f"No text generated for {ps_name}, ep {epoch+1}, prompt: '{user_prompt}'")

                if successful_epochs_this_ps > 0:
                    individual_result_entry = {
                        "image": img_file, "captions": captions_for_this_ps_image, "model": global_args.model_name,
                        "method": "entity2entity" if is_json_method_ps else "caption2entity",
                        "epochs_successful": successful_epochs_this_ps, "prompt_set": ps_name
                    }
                    if is_json_method_ps and objects_for_this_ps_image_raw:
                        valid_objects = [str(obj).lower().strip() for obj in objects_for_this_ps_image_raw if isinstance(obj, str) and obj.strip()]
                        individual_result_entry["caption_objects"] = sorted(list(set(valid_objects))) if valid_objects else []
                    
                    master_results_for_individual_files[ps_name].append(individual_result_entry)

                    if is_json_method_ps:
                        if img_file not in consolidated_json_by_image:
                            consolidated_json_by_image[img_file] = {
                                "image": img_file, "captions": [], "caption_objects": set(), "model": global_args.model_name,
                                "method": "entity2entity", "epochs": 0, "prompt_sets": set()
                            }
                        consolidated_json_by_image[img_file]["captions"].extend(captions_for_this_ps_image)
                        if "caption_objects" in individual_result_entry:
                            for obj in individual_result_entry["caption_objects"]: consolidated_json_by_image[img_file]["caption_objects"].add(obj)
                        consolidated_json_by_image[img_file]["epochs"] += successful_epochs_this_ps
                        consolidated_json_by_image[img_file]["prompt_sets"].add(ps_name)
                    else:
                        if img_file not in consolidated_desc_by_image:
                            consolidated_desc_by_image[img_file] = {
                                "image": img_file, "captions": [], "model": global_args.model_name, "method": "caption2entity",
                                "epochs": 0, "prompt_sets": set()
                            }
                        consolidated_desc_by_image[img_file]["captions"].extend(captions_for_this_ps_image)
                        consolidated_desc_by_image[img_file]["epochs"] += successful_epochs_this_ps
                        consolidated_desc_by_image[img_file]["prompt_sets"].add(ps_name)
                else:
                    if logger: logger.warning(f"  No successful epochs for image '{img_file}' with PS '{ps_name}'. Not added to results.")

            for ps_name_to_save in master_results_for_individual_files:
                ps_results_file_path = os.path.join(current_output_dir, f"results_{ps_name_to_save.replace(' ', '_')}.json")
                try:
                    with open(ps_results_file_path, "w", encoding="utf-8") as f:
                        json.dump(master_results_for_individual_files[ps_name_to_save], f, ensure_ascii=False, indent=2)
                except IOError as e:
                    if logger: logger.error(f"  Failed to save intermediate results for '{ps_name_to_save}': {e}")
            
            torch.cuda.empty_cache()
            if logger: logger.info(f"  Finished processing all prompt sets for image '{img_file}'.")
        
        except Exception as e:
            current_image_had_catastrophic_error = True
            if logger: logger.error(f"--- Catastrophic error processing image {img_file}. Error: {e}", exc_info=True)
        
        if not current_image_had_catastrophic_error: images_processed_successfully_count +=1
        else: images_with_catastrophic_errors_count +=1

    if logger: logger.info("--- Finalizing and Saving Image-Level Consolidated Results ---")
    
    final_consolidated_descriptions = []
    for img_data in consolidated_desc_by_image.values():
        img_data["prompt_sets"] = sorted(list(img_data["prompt_sets"]))
        final_consolidated_descriptions.append(img_data)

    final_consolidated_json_objects = []
    for img_data in consolidated_json_by_image.values():
        img_data["caption_objects"] = sorted(list(img_data["caption_objects"]))
        img_data["prompt_sets"] = sorted(list(img_data["prompt_sets"]))
        final_consolidated_json_objects.append(img_data)

    consolidated_desc_path = os.path.join(current_output_dir, "c2e.json")
    try:
        with open(consolidated_desc_path, "w", encoding="utf-8") as f:
            json.dump(final_consolidated_descriptions, f, ensure_ascii=False, indent=2)
        if logger: logger.info(f"Consolidated caption2entity results saved to: {consolidated_desc_path} (Entries: {len(final_consolidated_descriptions)})")
    except IOError as e:
        if logger: logger.error(f"Failed to save consolidated caption2entity results: {e}")

    consolidated_json_path = os.path.join(current_output_dir, "e2e.json")
    try:
        with open(consolidated_json_path, "w", encoding="utf-8") as f:
            json.dump(final_consolidated_json_objects, f, ensure_ascii=False, indent=2)
        if logger: logger.info(f"Consolidated JSON object results saved to: {consolidated_json_path} (Entries: {len(final_consolidated_json_objects)})")
    except IOError as e:
        if logger: logger.error(f"Failed to save consolidated JSON object results: {e}")
    
    total_script_run_time = time.time() - total_script_start_time
    if logger:
        logger.info(f"--- Overall Processing Summary ---")
        logger.info(f"Total script run time: {total_script_run_time:.2f} seconds.")
        logger.info(f"Images fully processed: {images_processed_successfully_count}")
        logger.info(f"Images with catastrophic errors: {images_with_catastrophic_errors_count}")

    return master_results_for_individual_files

def main():
    overall_start_time = time.time()
    logger = None
    try:
        args = parse_args_and_config()
        
        final_output_dir = args.output_dir
        os.makedirs(final_output_dir, exist_ok=True)

        logger = setup_logger(final_output_dir, getattr(args, 'log_level', 'INFO'))
        
        logger.info(f"LLaVA 1.6 batch processing started.")
        logger.info(f"Using configuration file: {args.config_file_path}")
        logger.info(f"All output will be saved to directory: {final_output_dir}")
        logger.info(f"Settings: Model='{args.model_name}', Device='{args.device}'")

        loaded_prompt_sets = load_and_validate_prompt_sets(args.prompt_set_configs_raw)
        logger.info(f"Loaded {len(loaded_prompt_sets)} prompt sets:")
        for i, ps_conf in enumerate(loaded_prompt_sets):
            has_system_prompt = "Yes" if ps_conf.get('system_prompt') else "No"
            logger.info(f"  {i+1}. Name:'{ps_conf['name']}', SystemPrompt:{has_system_prompt}, ParseJSON:{ps_conf['parse_json_output']}, UserPrompts:{len(ps_conf['user_prompts'])}")

        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()): logger.info(f"    Dev {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
        
        logger.info("Loading LLaVA 1.6 model..."); model_load_start = time.time()
        
        processor = LlavaNextProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        model.to(args.device)

        logger.info(f"Model loaded in {time.time() - model_load_start:.2f}s.")
        
        image_files_basenames = []
        selected_images_to_save_set = set()

        if args.selected_images_file:
            logger.info(f"Attempting to load selected image list from: {args.selected_images_file}")
            cfg_dir = os.path.dirname(os.path.abspath(args.config_file_path))
            potential_selected_images_path = os.path.join(cfg_dir, args.selected_images_file)
            
            actual_selected_images_path = args.selected_images_file
            if not os.path.isabs(args.selected_images_file) and os.path.exists(potential_selected_images_path):
                actual_selected_images_path = potential_selected_images_path
            elif not os.path.exists(actual_selected_images_path):
                logger.warning(f"Specified 'selected_images_file' ({args.selected_images_file} / {potential_selected_images_path}) not found.")
                actual_selected_images_path = None

            if actual_selected_images_path and os.path.exists(actual_selected_images_path):
                try:
                    with open(actual_selected_images_path, 'r', encoding='utf-8') as f:
                        selected_list_from_file = json.load(f)
                    
                    if not isinstance(selected_list_from_file, list):
                        logger.warning(f"'selected_images_file' ({actual_selected_images_path}) does not contain a JSON list. Falling back to random selection.")
                        raise ValueError("JSON content is not a list")

                    temp_image_files = []
                    for img_name in selected_list_from_file:
                        if not isinstance(img_name, str):
                            logger.warning(f"Skipping non-string entry '{img_name}' in {actual_selected_images_path}.")
                            continue
                        if os.path.exists(os.path.join(args.image_dir, img_name)) and \
                           img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            temp_image_files.append(img_name)
                        else:
                            logger.warning(f"Image '{img_name}' from '{actual_selected_images_path}' not found in '{args.image_dir}' or not a supported type. Skipped.")
                    
                    if not temp_image_files:
                        logger.warning(f"No valid images found from '{actual_selected_images_path}' in '{args.image_dir}'. Falling back to random selection.")
                        raise ValueError("No valid images from file")

                    if len(temp_image_files) > args.num_images_to_process:
                        logger.info(f"Loaded {len(temp_image_files)} images from file, using the first {args.num_images_to_process} as per 'num_images_to_process'.")
                        image_files_basenames = temp_image_files[:args.num_images_to_process]
                    else:
                        image_files_basenames = temp_image_files
                    
                    selected_images_to_save_set = set(image_files_basenames)
                    logger.info(f"Successfully loaded {len(image_files_basenames)} image(s) from '{actual_selected_images_path}'.")

                except (FileNotFoundError, ValueError, json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to load or process 'selected_images_file' ({actual_selected_images_path}): {e}. Falling back to random selection.")
                    image_files_basenames, selected_images_to_save_set = select_random_images(args.image_dir, args.num_images_to_process, args.seed, logger)
            else:
                if args.selected_images_file:
                    logger.warning(f"Specified 'selected_images_file' ('{args.selected_images_file}') could not be found. Falling back to random selection.")
                image_files_basenames, selected_images_to_save_set = select_random_images(args.image_dir, args.num_images_to_process, args.seed, logger)
        else:
            logger.info("No 'selected_images_file' provided in config. Using random image selection.")
            image_files_basenames, selected_images_to_save_set = select_random_images(args.image_dir, args.num_images_to_process, args.seed, logger)
        
        if not image_files_basenames:
            logger.error("No images were selected for processing. Exiting.")
            return

        selected_images_output_path = os.path.join(final_output_dir, "selected_images.json")
        try:
            with open(selected_images_output_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(selected_images_to_save_set)), f, ensure_ascii=False, indent=2)
            logger.info(f"List of selected image basenames saved to: {selected_images_output_path}")
        except IOError as e:
            logger.error(f"Failed to save selected images list to '{selected_images_output_path}': {e}")
        
        process_images_for_all_prompt_sets(
            image_files_basenames, args.image_dir, model, processor, args, loaded_prompt_sets, final_output_dir, logger
        )

        logger.info(f"Batch processing completed. Total time: {time.time() - overall_start_time:.2f} seconds.")
        logger.info(f"All results saved in directory: {final_output_dir}")
        print(f"OUTPUT_DIR={final_output_dir}")

    except Exception as e:
        print(f"An unexpected error occurred at the main level: {e}")
        if logger: logger.error("An unexpected error occurred at the main level.", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()