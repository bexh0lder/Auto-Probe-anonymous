import json
import yaml
import time
import logging
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple # Added Tuple
from tqdm import tqdm
import copy

# API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")


class EntityCleaner:
    def __init__(self, config_path: str = "config_entity_cleaner.yaml"):
        """初始化实体清洗器"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.system_prompt = self._load_system_prompt()
        self.user_prompt_template = self._load_user_prompt_template()
        self.client = self._setup_api_client()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Using user's specified config key
        if '02_entity_extract_entity_cleaner' not in config:
            raise ValueError("Configuration must contain '02_entity_extract_entity_cleaner' section")

        return config['02_entity_extract_entity_cleaner']

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())

        logger = logging.getLogger('EntityCleaner')
        logger.setLevel(log_level)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        log_file = log_config.get('log_file')
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

        return logger

    def _load_prompt_from_file(self, file_path: str) -> str:
        """从文件加载prompt内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Prompt file '{file_path}' is empty.")
        return content

    def _load_system_prompt(self) -> Optional[str]:
        """加载system prompt"""
        system_prompt_path = self.config.get('llm', {}).get('system_prompt_path')
        if not system_prompt_path:
            self.logger.info("No system_prompt_path provided. Proceeding without a specific system prompt.")
            raise ValueError("Missing system_prompt_path. This is mandatory for EntityCleaner.")
            return None
        
        try:
            self.logger.info(f"Loading system prompt from: {system_prompt_path}")
            system_prompt = self._load_prompt_from_file(system_prompt_path)
            self.logger.info("System prompt loaded successfully.")
            return system_prompt
        except Exception as e:
            self.logger.critical(f"Could not load mandatory user prompt file '{system_prompt_path}': {e}.")
            raise

    def _load_user_prompt_template(self) -> str:
        """加载user prompt模板"""
        user_prompt_path = self.config.get('llm', {}).get('user_prompt_path')
        if not user_prompt_path:
            self.logger.critical("'user_prompt_path' is missing in the configuration.")
            raise ValueError("Missing user_prompt_path. This is mandatory for EntityCleaner.")
        
        try:
            self.logger.info(f"Loading user prompt template from: {user_prompt_path}")
            user_prompt_template = self._load_prompt_from_file(user_prompt_path)
            self.logger.info("User prompt template loaded successfully.")
            return user_prompt_template
        except Exception as e:
            self.logger.critical(f"Could not load mandatory user prompt file '{user_prompt_path}': {e}.")
            raise

    def _setup_api_client(self):
        """设置API客户端"""
        api_config = self.config['llm']
        provider = api_config['provider'].lower()

        if provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")

            api_key = api_config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment variable OPENAI_API_KEY")

            client_kwargs = {'api_key': api_key}
            if api_config.get('base_url'):
                client_kwargs['base_url'] = api_config['base_url']

            return OpenAI(**client_kwargs)
        else:
            raise ValueError(f"Unsupported API provider: {provider}. Only 'openai' is supported.")

    def _call_llm_api(self, entity_list: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        调用LLM API进行实体清洗.
        返回: Tuple (解析后的列表 或 None, 原始响应文本 或 None)
        """
        api_config = self.config['llm']
        
        # Ensure entity_list is a list of strings for JSON dump
        if not isinstance(entity_list, list) or not all(isinstance(e, str) for e in entity_list):
            self.logger.error(f"Invalid entity_list format for LLM call: {entity_list}")
            return None, None

        if "{entity_list}" not in self.user_prompt_template:
            self.logger.error("User prompt template is missing the '{entity_list}' placeholder. Entity cleaning might fail or produce poor results.")

        try:
            user_prompt = self.user_prompt_template.format(entity_list=json.dumps(entity_list, ensure_ascii=False))
        except KeyError:
            self.logger.error("Failed to format user prompt. Ensure '{entity_list}' placeholder is correct.")
            return None, None

        messages_for_api = []
        if self.system_prompt:
            messages_for_api.append({"role": "system", "content": self.system_prompt})
        messages_for_api.append({"role": "user", "content": user_prompt})

        max_retries = api_config.get('max_retries', 3)
        timeout = api_config.get('timeout', 30)
        last_raw_response = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=api_config['model'],
                    messages=messages_for_api,
                    timeout=timeout
                )
                result_text = response.choices[0].message.content.strip()
                last_raw_response = result_text

                try:
                    cleaned_list = json.loads(result_text)
                    if not isinstance(cleaned_list, list):
                        self.logger.warning(f"LLM response content is not a list. Raw response: {result_text[:200]}...")
                        if attempt == max_retries - 1:
                            return None, last_raw_response
                        time.sleep(api_config.get('delay_on_parse_error', 1)) # Small delay for format error
                        continue # Next attempt
                    return cleaned_list, last_raw_response

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM response as JSON: {e}. Raw response: {result_text[:200]}...")
                    if attempt == max_retries - 1:
                        return None, last_raw_response
                    time.sleep(api_config.get('delay_on_parse_error', 1)) # Small delay for JSON error
                    continue # Next attempt
            
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1}/{max_retries} failed: {e}")
                last_raw_response = f"API Error: {e}" # Store error as raw response if API fails
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} API call attempts failed.")
                    return None, last_raw_response 
                time.sleep(2 ** attempt) # Exponential backoff for API errors

        return None, last_raw_response # Should only be reached if max_retries is 0 or less

    def _process_cleaned_results(self, original_objects: List[str],
                                 cleaned_objects: List[str]) -> Dict[str, Any]: # Return type includes Any for entity_mapping
        """处理清洗结果，生成映射关系"""
        if not isinstance(original_objects, list) or not isinstance(cleaned_objects, list):
             self.logger.warning(f"Invalid input types for _process_cleaned_results. Original: {type(original_objects)}, Cleaned: {type(cleaned_objects)}")
             return {"candidate_objects": [], "removed_entities": original_objects if isinstance(original_objects, list) else [], "entity_mapping": {}}

        if len(original_objects) != len(cleaned_objects):
            self.logger.warning(
                f"Length mismatch during _process_cleaned_results: original({len(original_objects)}) vs cleaned({len(cleaned_objects)})"
            )
            return {"candidate_objects": [], "removed_entities": original_objects, "entity_mapping": {}}

        entity_mapping = {}
        candidate_objects = []
        removed_entities = []

        for original, cleaned in zip(original_objects, cleaned_objects):
            if not isinstance(cleaned, str): # Ensure cleaned item is a string
                self.logger.warning(f"Cleaned item is not a string: '{cleaned}'. Original: '{original}'. Treating as REMOVE.")
                removed_entities.append(original)
                continue

            if cleaned == "<REMOVE>":
                removed_entities.append(original)
            else:
                candidate_objects.append(cleaned)
                entity_mapping[original] = cleaned
        
        return {
            "candidate_objects": sorted(list(set(candidate_objects))),
            "removed_entities": removed_entities,
            "entity_mapping": entity_mapping
        }

    def _update_sg_attributes(self, sg_attributes: Dict[str, List[str]],
                              removed_entities: List[str]) -> Dict[str, List[str]]:
        """更新sg_attributes，删除被标记为REMOVE的实体"""
        if not sg_attributes:
            return {}
        if not isinstance(sg_attributes, dict):
            self.logger.warning(f"sg_attributes is not a dict: {sg_attributes}. Returning original.")
            return sg_attributes


        updated_sg_attributes = {}
        removed_entities_set = set(removed_entities)

        for entity, attributes in sg_attributes.items():
            if entity not in removed_entities_set:
                updated_sg_attributes[entity] = attributes
            else:
                self.logger.debug(f"Removed entity '{entity}' from sg_attributes due to cleaning.")
        return updated_sg_attributes

    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据项，包括空实体结果的重试，并添加LLM原始回复"""
        image_name = item.get('image', 'unknown')
        self.logger.debug(f"Processing item: {image_name}")

        # Build the output dictionary, starting with original item's fields to preserve order for non-processed keys
        final_output_dict = {}
        # First, copy all keys from original item that are NOT the ones we will explicitly manage for order/value
        for k, v in item.items():
            if k not in ['LLM_response', 'candidate_objects', 'sg_attributes', 'error_cleaning_entities']:
                final_output_dict[k] = v
        
        original_caption_objects = item.get('caption_objects', [])
        if not isinstance(original_caption_objects, list): # Ensure it's a list
            self.logger.warning(f"Item {image_name}: caption_objects is not a list, but {type(original_caption_objects)}. Treating as empty.")
            original_caption_objects = []
            final_output_dict['caption_objects'] = [] # Ensure it's a list in output if malformed

        llm_raw_response_to_store = None
        final_parsed_list_from_llm = None # Stores the LLM's parsed list that resulted in non-empty candidates

        if not original_caption_objects:
            self.logger.warning(f"Item {image_name}: caption_objects is empty. No cleaning will be performed.")
            final_output_dict['LLM_response'] = None
            final_output_dict['candidate_objects'] = []
            final_output_dict['sg_attributes'] = item.get('sg_attributes', {}) # Keep original or default to empty
            return final_output_dict

        # Semantic retry loop (for getting non-empty candidate_objects)
        semantic_max_retries = self.config['llm'].get('max_retries', 3)
        
        for attempt in range(semantic_max_retries):
            self.logger.info(f"Item {image_name}: Semantic cleaning attempt {attempt + 1}/{semantic_max_retries} for {len(original_caption_objects)} objects.")
            
            current_parsed_list, current_raw_response = self._call_llm_api(original_caption_objects)
            
            if current_raw_response is not None:
                llm_raw_response_to_store = current_raw_response

            if current_parsed_list is not None:
                processing_result = self._process_cleaned_results(original_caption_objects, current_parsed_list)
                if processing_result['candidate_objects']:
                    self.logger.info(f"Item {image_name}: Successfully obtained {len(processing_result['candidate_objects'])} candidate_objects on attempt {attempt + 1}.")
                    final_parsed_list_from_llm = current_parsed_list
                    break 
                else:
                    self.logger.warning(
                        f"Item {image_name}: Candidate_objects list is empty after processing on attempt {attempt + 1}/{semantic_max_retries}. "
                        f"LLM raw response: {str(llm_raw_response_to_store)[:200]}..."
                    )
            else:
                self.logger.warning(
                    f"Item {image_name}: _call_llm_api returned no parsed list on semantic attempt {attempt + 1}/{semantic_max_retries}. "
                    f"LLM raw response (if any): {str(llm_raw_response_to_store)[:200]}..."
                )
            
            if attempt < semantic_max_retries - 1:
                delay_key = 'delay_between_empty_result_retries'
                default_delay = self.config['llm'].get('timeout', 30) / 10 # Heuristic: 1/10th of timeout or fixed value
                if default_delay < 1.0 : default_delay = 1.0 # minimum 1s
                if default_delay > 5.0 : default_delay = 5.0 # maximum 5s
                
                semantic_retry_delay = self.config.get('processing', {}).get(delay_key, default_delay)
                self.logger.debug(f"Item {image_name}: Delaying {semantic_retry_delay}s before next semantic retry.")
                time.sleep(semantic_retry_delay)
        
        # --- Populate final output dictionary with controlled order ---
        final_output_dict['LLM_response'] = llm_raw_response_to_store

        if final_parsed_list_from_llm is not None:
            # Re-process with the list that yielded success to get final results
            final_processing_result = self._process_cleaned_results(original_caption_objects, final_parsed_list_from_llm)
            final_output_dict['candidate_objects'] = final_processing_result['candidate_objects']
            
            original_sg_attributes = item.get('sg_attributes', {}) # Get original sg_attributes
            final_output_dict['sg_attributes'] = self._update_sg_attributes(
                original_sg_attributes,
                final_processing_result['removed_entities']
            )
            self.logger.info(
                f"Successfully processed item {image_name}: "
                f"{len(original_caption_objects)} original -> {len(final_output_dict['candidate_objects'])} candidate objects."
            )
        else: # All semantic retries failed
            self.logger.error(
                f"Item {image_name}: Failed to obtain non-empty candidate_objects after {semantic_max_retries} attempts. "
                f"Falling back to original caption_objects. Last LLM raw response: {str(llm_raw_response_to_store)[:200]}..."
            )
            final_output_dict['candidate_objects'] = original_caption_objects # Fallback
            final_output_dict['sg_attributes'] = item.get('sg_attributes', {}) # Keep original sg_attributes
            final_output_dict['error_cleaning_entities'] = f"Failed to obtain non-empty cleaned entities after {semantic_max_retries} retries."
            
        return final_output_dict

    def _save_compact_json(self, data_to_save: List[Dict[str, Any]], output_filepath: str):
        """以紧凑格式保存JSON文件，特定列表字段单行显示"""
        items_to_format = copy.deepcopy(data_to_save)
        placeholder_map = {}
        placeholder_idx_obj = [0] 

        def convert_lists_to_placeholders_recursive(current_data, p_map, p_idx_obj_ref, parent_key=None):
            if isinstance(current_data, dict):
                for key, value in list(current_data.items()): 
                    should_compact = False
                    # User's version also compacts 'candidate_objects'
                    if (key == 'caption_objects' or key == 'candidate_objects') and isinstance(value, list):
                        should_compact = True
                    elif parent_key == 'sg_attributes' and isinstance(value, list):
                        should_compact = True

                    if should_compact:
                        placeholder_key_string = f"__COMPACT_LIST_PLACEHOLDER_{p_idx_obj_ref[0]}__"
                        try:
                            p_map[placeholder_key_string] = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
                            current_data[key] = placeholder_key_string
                            p_idx_obj_ref[0] += 1
                        except TypeError as te:
                            self.logger.warning(f"Could not JSON dump list for key '{key}' (parent: {parent_key}) due to TypeError: {te}. List: {str(value)[:100]}...")
                    elif isinstance(value, dict):
                        convert_lists_to_placeholders_recursive(value, p_map, p_idx_obj_ref, key)
                    elif isinstance(value, list): 
                        for item_in_list in value:
                            if isinstance(item_in_list, dict):
                                convert_lists_to_placeholders_recursive(item_in_list, p_map, p_idx_obj_ref, None)
        
        for item_dict in items_to_format:
            if isinstance(item_dict, dict):
                convert_lists_to_placeholders_recursive(item_dict, placeholder_map, placeholder_idx_obj)

        json_string_with_placeholders = json.dumps(items_to_format, indent=2, ensure_ascii=False)

        for ph_key, compact_list_str in placeholder_map.items():
            json_string_with_placeholders = json_string_with_placeholders.replace(f'"{ph_key}"', compact_list_str)

        try:
            output_dir_path = os.path.dirname(output_filepath)
            if output_dir_path and not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
                self.logger.info(f"Created output directory: {output_dir_path}")

            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(json_string_with_placeholders)
            self.logger.info(f"Successfully saved {len(data_to_save)} processed items with compact lists to {output_filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save output file with compact lists '{output_filepath}': {e}")

    def process_file(self, input_file: Optional[str] = None,
                     output_file: Optional[str] = None) -> None:
        """处理整个文件"""
        input_path = input_file if input_file is not None else self.config['io']['input_file']
        output_path = output_file if output_file is not None else self.config['io']['output_file']

        if not input_path:
            self.logger.error("Input file path not specified. Use --input_file or set in config file.")
            return
        if not output_path:
            self.logger.error("Output file path not specified. Use --output_file or set in config file.")
            return

        self.logger.info(f"Starting entity cleaning process")
        self.logger.info(f"Input file: {input_path} (Source: {'command line' if input_file else 'config'})")
        self.logger.info(f"Output file: {output_path} (Source: {'command line' if output_file else 'config'})")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read input file: {input_path}. Error: {e}")
            return

        if not isinstance(input_data, list):
            self.logger.error(f"Input data from {input_path} must be a list of items.")
            return

        self.logger.info(f"Loaded {len(input_data)} items for processing from {input_path}")

        processed_data = []
        # delay_between_api_items is the delay between processing different items in the input file
        delay_between_api_items = self.config.get('processing', {}).get('delay_between_requests', 1.0) 

        for i, item in enumerate(tqdm(input_data, desc="Processing items")):
            try:
                processed_item = self.process_single_item(item)
                processed_data.append(processed_item)

                if i < len(input_data) - 1 and delay_between_api_items > 0:
                    time.sleep(delay_between_api_items)

            except Exception as e: # Catch unexpected errors during single item processing
                image_name_err = item.get('image', f'unknown_item_index_{i}')
                self.logger.error(f"Critical error processing item {image_name_err}: {e}", exc_info=True)
                
                fallback_item = copy.deepcopy(item)
                fallback_item['LLM_response'] = f"Error during processing: {e}"
                fallback_item['candidate_objects'] = item.get('caption_objects', []) # Fallback
                fallback_item['error_processing_item'] = str(e)
                processed_data.append(fallback_item)
        
        self._save_compact_json(processed_data, output_path)


def main():
    parser = argparse.ArgumentParser(description="Entity Cleaning Automation Script")
    parser.add_argument("--config", type=str, default="config_entity_cleaner.yaml",
                        help="Path to configuration file")
    parser.add_argument("--input_file", type=str,
                        help="Input JSON file path (overrides config setting)")
    parser.add_argument("--output_file", type=str,
                        help="Output JSON file path (overrides config setting)")

    args = parser.parse_args()

    try:
        cleaner = EntityCleaner(args.config)
        cleaner.process_file(args.input_file, args.output_file)
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        logging.getLogger().error(f"Configuration Error: {e}", exc_info=False) # also log it
        exit(1)
    except ValueError as e: # For config structure issues or API key issues
        print(f"Configuration or Setup Error: {e}")
        logging.getLogger().error(f"Configuration or Setup Error: {e}", exc_info=False)
        exit(1)
    except ImportError as e: # For OpenAI library missing
        print(f"Import Error: {e}")
        logging.getLogger().error(f"Import Error: {e}", exc_info=False)
        exit(1)
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
        logging.getLogger().critical(f"An unexpected critical error occurred in main: {e}", exc_info=True)
        exit(1)
    
    exit(0) # Explicitly exit with 0 on success


if __name__ == "__main__":
    main()