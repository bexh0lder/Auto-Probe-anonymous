import yaml
import json
import os
import sys
import re
import logging
import argparse
from tqdm import tqdm
import copy

from openai import OpenAI

# -----------------------------------------------------------------------------
# 0. Configuration Loading & Utilities
# -----------------------------------------------------------------------------
def load_config(config_path="config_extractor.yaml"):
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            # This print will be used if logger is not yet set up in main
            print(f"ERROR: Configuration file {config_path} is empty or invalid.")
            return None
        
        if config_data['02_entity_extract_caption_extractor'] is None:
            print("❌02_entity_extract_caption_extractor is not specified in the configuration file. Exiting.")
            sys.exit(1)
        config_data = config_data['02_entity_extract_caption_extractor']

        # Set defaults for expected sections
        config_data.setdefault('io', {})
        config_data.setdefault('logging', {})
        config_data.setdefault('llm', {})

        # Defaults relevant to extractor
        config_data['logging'].setdefault('level', 'INFO')
        config_data['llm'].setdefault('model_name', 'qwen-plus')

        # IO paths are now optional in YAML, will be validated in main() after CLI override
        io_conf = config_data['io']
        io_conf.setdefault('input_json_path', None) 
        io_conf.setdefault('output_json_path', None)
        
        # Prompt paths for LLMExtractor are checked during its initialization
        # No need for strict checks here if they are configured under 'llm' section
        
        return config_data
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading config {config_path}: {e}")
        return None

def setup_logger(name, level=logging.INFO):
    """Sets up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers(): # Avoid adding multiple handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False # Avoid double logging if root logger is also configured
    return logger

# -----------------------------------------------------------------------------
# 1. LLM Entity Extractor
# -----------------------------------------------------------------------------
class LLMExtractor:
    # ... (您的 LLMExtractor 类代码保持不变) ...
    # 确保其内部的 self.logger.xxx() 调用可以正确工作
    def __init__(self, openai_api_key, openai_base_url, model_name, logger, system_prompt_path, user_prompt_path):
        self.logger = logger # Ensure logger is passed and used
        try:
            self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url if openai_base_url else None)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None # Ensure client is None if init fails

        self.model_name = model_name

        if not user_prompt_path:
            self.logger.critical("LLM 'user_prompt_path' is missing in the configuration.")
            raise ValueError("Missing LLM user_prompt_path. This is mandatory if LLMExtractor is used.")
        
        try:
            self.logger.info(f"Loading LLM user prompt template from: {user_prompt_path}")
            self.user_prompt_template = self._load_prompt_from_file(user_prompt_path)
            self.logger.info("LLM user prompt template loaded successfully.")
        except Exception as e: # Catch any exception from _load_prompt_from_file
            self.logger.critical(f"Could not load mandatory LLM user prompt file '{user_prompt_path}': {e}.")
            raise # Re-raise as it's mandatory

        self.system_prompt = None
        if system_prompt_path:
            try:
                self.logger.info(f"Loading LLM system prompt from: {system_prompt_path}")
                self.system_prompt = self._load_prompt_from_file(system_prompt_path)
                if self.system_prompt: self.logger.info("LLM system prompt loaded successfully.")
            except FileNotFoundError:
                self.logger.warning(f"LLM system_prompt_path ('{system_prompt_path}') not found. Proceeding without it.")
            except ValueError as e: # Handles empty file
                self.logger.warning(f"LLM system prompt file ('{system_prompt_path}') issue: {e}. Proceeding without it.")
            except IOError as e: # Handles other read errors
                 self.logger.warning(f"LLM system prompt file IO error ('{system_prompt_path}'): {e}. Proceeding without it.")
        else:
            self.logger.info("No LLM system_prompt_path provided. Proceeding without a specific system prompt.")

    def _load_prompt_from_file(self, file_path): # This is a helper, can raise exceptions
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Prompt file '{file_path}' is empty.")
        return content

    def extract_entities(self, caption_text):
        if not self.client:
            self.logger.error("LLM client not initialized. Cannot extract entities.")
            return []
        if not self.user_prompt_template: # Should have been caught in init, but double check
            self.logger.error("LLM user prompt template not available. Cannot extract entities.")
            return []
            
        self.logger.debug(f"LLM: Requesting entity extraction for caption: {caption_text[:100]}...")
        if "{caption_text}" not in self.user_prompt_template:
            self.logger.error("LLM user prompt template is missing the '{caption_text}' placeholder. Entity extraction might fail or produce poor results.")
            # Allow to proceed, maybe the template is unusual
            
        try:
            user_prompt = self.user_prompt_template.format(caption_text=caption_text)
        except KeyError:
            self.logger.error("Failed to format LLM user prompt. Ensure '{caption_text}' placeholder is correct.")
            return []

        messages_for_api = []
        if self.system_prompt:
            messages_for_api.append({"role": "system", "content": self.system_prompt})
        messages_for_api.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=0.1, # As per original code
                # response_format={"type": "json_object"} # As per original code
            )
            llm_response_content = response.choices[0].message.content
            self.logger.debug(f"LLM raw response: {llm_response_content}")
            
            extracted_entities_raw = []
            if llm_response_content:
                try:
                    parsed_json = json.loads(llm_response_content)
                    if isinstance(parsed_json, list):
                        extracted_entities_raw = parsed_json
                    elif isinstance(parsed_json, dict):
                        possible_keys = ["entities", "objects", "list", "items", "data"]
                        found_list = False
                        for key in possible_keys:
                            if key in parsed_json and isinstance(parsed_json[key], list):
                                extracted_entities_raw = parsed_json[key]
                                found_list = True; break
                        if not found_list:
                            self.logger.warning(f"LLM JSON object response did not contain a recognized list key. Keys: {list(parsed_json.keys())}")
                    else:
                        self.logger.warning(f"LLM JSON response was not a list or dict. Type: {type(parsed_json)}")
                except json.JSONDecodeError:
                    self.logger.warning(f"LLM response not valid JSON: {llm_response_content[:200]}... Trying ast.literal_eval.")
                    if llm_response_content.strip().startswith('[') and llm_response_content.strip().endswith(']'):
                        try:
                            import ast # Keep import here as it's a fallback
                            temp_list = ast.literal_eval(llm_response_content)
                            if isinstance(temp_list, list): extracted_entities_raw = temp_list
                        except Exception as e_ast: self.logger.error(f"ast.literal_eval failed on LLM response: {e_ast}")
                    else: self.logger.warning("LLM response not list-like for ast.literal_eval.")
            
            valid_entities = []
            for entity_candidate in extracted_entities_raw:
                if isinstance(entity_candidate, str):
                    stripped_entity = entity_candidate.strip()
                    if stripped_entity: valid_entities.append(stripped_entity)
                elif isinstance(entity_candidate, list): 
                    self.logger.debug(f"LLM extracted a nested list, attempting to flatten: {entity_candidate}")
                    for sub_entity in entity_candidate:
                        if isinstance(sub_entity, str):
                            stripped_sub = sub_entity.strip()
                            if stripped_sub: valid_entities.append(stripped_sub)
                else:
                    self.logger.warning(f"LLM extracted non-string/non-list element: {type(entity_candidate)}")
            self.logger.info(f"LLM extracted {len(valid_entities)} entities from caption.")
            return valid_entities
        except Exception as e: # Catch broader exceptions during API call or processing
            self.logger.error(f"Error during LLM API call or response processing: {e}", exc_info=True)
            return []

# -----------------------------------------------------------------------------
# Main Execution Block (Modified for CLI I/O override)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2E Raw Entity and Attribute Extraction Pipeline (c2e_extractor)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_extractor.yaml", 
        help="Path to the YAML configuration file. Default: config_extractor.yaml"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None, # Optional CLI override for input path
        help="Path to the input JSON file (LLaVA description output, overrides YAML)."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None, # Optional CLI override for output path
        help="Path for the output JSON file (extracted entities, overrides YAML)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        # load_config prints error, so just exit
        sys.exit(1)

    # Setup main logger for the script from config
    log_conf = config.get('logging', {})
    log_level_str = log_conf.get('level', 'INFO').upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)
    
    script_logger = setup_logger("C2E_ExtractorScript", numeric_log_level) # Use specific name
    script_logger.info(f"Logger initialized for C2E_ExtractorScript, level {log_level_str}.")
    script_logger.info(f"Using configuration from: {args.config}")

    # Resolve input and output paths: CLI > YAML
    io_config = config.get('io', {})
    input_path = args.input_file if args.input_file else io_config.get('input_json_path')
    output_path = args.output_file if args.output_file else io_config.get('output_json_path')

    if not input_path:
        script_logger.critical("Input file path ('input_json_path') not specified in YAML or via --input_file CLI. Exiting.")
        sys.exit(1)
    if not output_path:
        script_logger.critical("Output file path ('output_json_path') not specified in YAML or via --output_file CLI. Exiting.")
        sys.exit(1)
    
    script_logger.info(f"Effective input file: {input_path}")
    script_logger.info(f"Effective output file: {output_path}")

    # Initialize LLM Extractor (optional, based on config)
    llm_config = config.get('llm', {})
    llm_extractor = None
    if llm_config and llm_config.get('user_prompt_path'): # Check if configured to be used
        try:
            llm_extractor = LLMExtractor(
                openai_api_key=llm_config.get('api_key'), 
                openai_base_url=llm_config.get('base_url'),
                model_name=llm_config.get('model_name', 'qwen-plus'), 
                logger=script_logger, # Pass the main script logger
                system_prompt_path=llm_config.get('system_prompt_path'),
                user_prompt_path=llm_config.get('user_prompt_path')
            )
            script_logger.info("LLMExtractor initialized.")
        except Exception as e:
            script_logger.error(f"LLMExtractor initialization failed: {e}. LLM extraction will be skipped.", exc_info=True)
            llm_extractor = None
    else:
        script_logger.info("LLM Extractor not configured (missing 'user_prompt_path') or 'llm' section missing. Skipping LLM extraction.")

    if not llm_extractor:
        script_logger.critical("LLM extractor could not be initialized. Nothing to do. Exiting.")
        sys.exit(1)

    # Load input data (output from LLaVA description generation)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data_list = json.load(f)
        if not isinstance(input_data_list, list):
            script_logger.error(f"Input JSON from '{input_path}' is not a list. Exiting."); sys.exit(1)
    except FileNotFoundError: 
        script_logger.error(f"Input JSON (LLaVA description output) not found: '{input_path}'. Exiting."); sys.exit(1)
    except json.JSONDecodeError as e: 
        script_logger.error(f"Error decoding JSON from '{input_path}': {e}. Exiting."); sys.exit(1)
    except Exception as e: 
        script_logger.error(f"Failed to load input JSON from '{input_path}': {e}", exc_info=True); sys.exit(1)
        
    script_logger.info(f"Starting C2E raw extraction for {len(input_data_list)} items. Input: '{input_path}', Output: '{output_path}'")
    
    results_list = []
    # Determine tqdm disabling based on logger level
    disable_tqdm = script_logger.level > logging.INFO

    for item_data_original in tqdm(input_data_list, desc="Extracting from items", unit="item", disable=disable_tqdm):
        if not isinstance(item_data_original, dict):
            script_logger.warning(f"Skipping non-dictionary item: {str(item_data_original)[:100]}")
            # results_list.append(item_data_original) # Optionally pass through
            continue

        # Ensure item_results starts as a copy of the original to preserve all original fields
        item_results = copy.deepcopy(item_data_original)

        current_item_id = item_results.get('image', f"unknown_item_idx_{len(results_list)}") # Prefer 'image' key

        captions_to_process_list = []
        raw_captions = item_results.get('captions') # Plural 'captions' is expected from batch_llava.py
        raw_caption_singular = item_results.get('caption') # Fallback

        if raw_captions and isinstance(raw_captions, list):
            for cap_text in raw_captions:
                if isinstance(cap_text, str) and cap_text.strip():
                    captions_to_process_list.append(cap_text)
        elif raw_caption_singular and isinstance(raw_caption_singular, str) and raw_caption_singular.strip():
            captions_to_process_list.append(raw_caption_singular)
        
        if not captions_to_process_list:
            script_logger.warning(f"Item '{current_item_id}': No valid descriptions found. Skipping extraction for this item.")
            item_results['llm_objects'] = [] # Ensure keys exist even if empty
            results_list.append(item_results)
            continue

        # Aggregated results for this item (from potentially multiple captions)
        aggregated_llm_objects_for_item = set()

        script_logger.debug(f"Item '{current_item_id}': Processing {len(captions_to_process_list)} caption(s).")
        for caption_idx, caption_text in enumerate(captions_to_process_list):
            script_logger.debug(f"  Caption {caption_idx+1}: {caption_text[:100]}...")
            if llm_extractor:
                try:
                    current_llm = llm_extractor.extract_entities(caption_text)
                    if current_llm: aggregated_llm_objects_for_item.update(current_llm)
                except Exception as e_llm: # Catch errors from the extractor call itself
                    script_logger.error(f"LLM extraction error for item '{current_item_id}', caption {caption_idx+1}: {e_llm}", exc_info=False) # exc_info=False for brevity in loop

        item_results['llm_objects'] = sorted(list(aggregated_llm_objects_for_item))
        
        results_list.append(item_results)

    # Save final results list
    # Ensure output directory exists for output_path
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string (e.g. if output_path is just a filename)
        os.makedirs(output_dir, exist_ok=True)
        script_logger.info(f"Created output directory: {output_dir}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        script_logger.info(f"Raw extractions saved to '{output_path}'. Processed {len(results_list)} items.")
    except Exception as e:
        script_logger.error(f"Error writing output JSON to '{output_path}': {e}", exc_info=True)

    script_logger.info("C2E Extractor pipeline finished.")