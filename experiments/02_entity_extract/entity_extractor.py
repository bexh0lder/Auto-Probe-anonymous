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


def load_config(config_path="config_extractor.yaml"):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            print(f"ERROR: Configuration file {config_path} is empty or invalid.")
            return None
        
        if config_data['02_entity_extract_caption_extractor'] is None:
            print("ERROR: 02_entity_extract_caption_extractor is not specified in the configuration file.")
            sys.exit(1)
        config_data = config_data['02_entity_extract_caption_extractor']

        # Initialize configuration sections with defaults
        config_data.setdefault('io', {})
        config_data.setdefault('logging', {})
        config_data.setdefault('llm', {})

        # Configure default logging and model settings
        config_data['logging'].setdefault('level', 'INFO')
        config_data['llm'].setdefault('model_name', 'qwen-plus')

        # Configure I/O paths (validated later if needed)
        io_conf = config_data['io']
        io_conf.setdefault('input_json_path', None) 
        io_conf.setdefault('output_json_path', None)
        
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
    """Set up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False
    return logger


class LLMExtractor:
    """LLM-based entity extraction from captions."""
    
    def __init__(self, openai_api_key, openai_base_url, model_name, logger, system_prompt_path, user_prompt_path):
        self.logger = logger
        try:
            self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url if openai_base_url else None)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

        self.model_name = model_name

        if not user_prompt_path:
            self.logger.critical("LLM 'user_prompt_path' is missing in the configuration.")
            raise ValueError("Missing LLM user_prompt_path. This is mandatory if LLMExtractor is used.")
        
        try:
            self.logger.info(f"Loading LLM user prompt template from: {user_prompt_path}")
            self.user_prompt_template = self._load_prompt_from_file(user_prompt_path)
            self.logger.info("LLM user prompt template loaded successfully.")
        except Exception as e:
            self.logger.critical(f"Could not load mandatory LLM user prompt file '{user_prompt_path}': {e}.")
            raise

        self.system_prompt = None
        if system_prompt_path:
            try:
                self.logger.info(f"Loading LLM system prompt from: {system_prompt_path}")
                self.system_prompt = self._load_prompt_from_file(system_prompt_path)
                self.logger.info("LLM system prompt loaded successfully.")
            except Exception as e:
                self.logger.warning(f"Failed to load optional LLM system prompt file '{system_prompt_path}': {e}. Proceeding without system prompt.")
        else:
            self.logger.info("No LLM system prompt path specified. Proceeding without system prompt.")

    def _load_prompt_from_file(self, file_path):
        """Load prompt content from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Prompt file '{file_path}' is empty.")
        return content

    def extract_entities(self, caption_text):
        """Extract entities from caption text using LLM."""
        if not self.client:
            self.logger.error("LLM client not initialized. Cannot extract entities.")
            return []
        if not self.user_prompt_template:
            self.logger.error("LLM user prompt template not available. Cannot extract entities.")
            return []
            
        self.logger.debug(f"LLM: Requesting entity extraction for caption: {caption_text[:100]}...")
        if "{caption_text}" not in self.user_prompt_template:
            self.logger.error("LLM user prompt template is missing the '{caption_text}' placeholder.")
            
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
                temperature=0.1
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
                        # Check for common list keys in the response dictionary
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
                    self.logger.warning(f"LLM response not valid JSON: {llm_response_content[:200]}... Attempting fallback parsing.")
                    # Fallback: try literal evaluation for list-like responses
                    if llm_response_content.strip().startswith('[') and llm_response_content.strip().endswith(']'):
                        try:
                            import ast
                            temp_list = ast.literal_eval(llm_response_content)
                            if isinstance(temp_list, list): extracted_entities_raw = temp_list
                        except Exception as e_ast: 
                            self.logger.error(f"Fallback literal evaluation failed: {e_ast}")
                    else: 
                        self.logger.warning("LLM response format not suitable for fallback parsing.")
            
            # Process and validate extracted entities
            valid_entities = []
            for entity_candidate in extracted_entities_raw:
                if isinstance(entity_candidate, str):
                    stripped_entity = entity_candidate.strip()
                    if stripped_entity: valid_entities.append(stripped_entity)
                elif isinstance(entity_candidate, list): 
                    self.logger.debug(f"Processing nested list from LLM response: {entity_candidate}")
                    for sub_entity in entity_candidate:
                        if isinstance(sub_entity, str):
                            stripped_sub = sub_entity.strip()
                            if stripped_sub: valid_entities.append(stripped_sub)
                else:
                    self.logger.warning(f"Unexpected entity type in LLM response: {type(entity_candidate)}")
            self.logger.info(f"Successfully extracted {len(valid_entities)} entities from caption.")
            return valid_entities
        except Exception as e:
            self.logger.error(f"Error during LLM API call or response processing: {e}", exc_info=True)
            return []

# Entity extraction pipeline main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity extraction pipeline for image captions")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_extractor.yaml", 
        help="Path to the YAML configuration file. Default: config_extractor.yaml"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None,
        help="Path to the input JSON file (overrides YAML configuration)."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path for the output JSON file (overrides YAML configuration)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        sys.exit(1)

    # Configure logging from configuration file
    log_conf = config.get('logging', {})
    log_level_str = log_conf.get('level', 'INFO').upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)
    
    script_logger = setup_logger("EntityExtractor", numeric_log_level)
    script_logger.info(f"Logger initialized with level {log_level_str}")
    script_logger.info(f"Using configuration from: {args.config}")

    # Determine I/O paths: command line arguments override YAML configuration
    io_config = config.get('io', {})
    input_path = args.input_file if args.input_file else io_config.get('input_json_path')
    output_path = args.output_file if args.output_file else io_config.get('output_json_path')

    if not input_path:
        script_logger.critical("Input file path not specified. Use --input_file or configure in YAML.")
        sys.exit(1)
    if not output_path:
        script_logger.critical("Output file path not specified. Use --output_file or configure in YAML.")
        sys.exit(1)
    
    script_logger.info(f"Input file: {input_path}")
    script_logger.info(f"Output file: {output_path}")

    # Initialize LLM-based entity extractor
    llm_config = config.get('llm', {})
    llm_extractor = None
    if llm_config and llm_config.get('user_prompt_path'):
        try:
            llm_extractor = LLMExtractor(
                openai_api_key=llm_config.get('api_key'), 
                openai_base_url=llm_config.get('base_url'),
                model_name=llm_config.get('model_name', 'qwen-plus'), 
                logger=script_logger,
                system_prompt_path=llm_config.get('system_prompt_path'),
                user_prompt_path=llm_config.get('user_prompt_path')
            )
            script_logger.info("LLM extractor initialized successfully.")
        except Exception as e:
            script_logger.error(f"Failed to initialize LLM extractor: {e}. Extraction will be skipped.", exc_info=True)
            llm_extractor = None
    else:
        script_logger.info("LLM extractor not configured. Skipping LLM-based entity extraction.")

    if not llm_extractor:
        script_logger.critical("LLM extractor initialization failed. Cannot proceed.")
        sys.exit(1)

    # Load input data from JSON file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data_list = json.load(f)
        if not isinstance(input_data_list, list):
            script_logger.error(f"Input JSON must be a list. Got: {type(input_data_list)}")
            sys.exit(1)
    except FileNotFoundError: 
        script_logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e: 
        script_logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e: 
        script_logger.error(f"Failed to load input file: {e}", exc_info=True)
        sys.exit(1)
        
    script_logger.info(f"Processing {len(input_data_list)} items from {input_path}")
    
    results_list = []
    # Disable progress bar for higher log levels to reduce output clutter
    disable_tqdm = script_logger.level > logging.INFO

    for item_data_original in tqdm(input_data_list, desc="Processing items", unit="item", disable=disable_tqdm):
        if not isinstance(item_data_original, dict):
            script_logger.warning(f"Skipping invalid item: {str(item_data_original)[:100]}")
            continue

        # Create a deep copy to preserve all original fields
        item_results = copy.deepcopy(item_data_original)

        current_item_id = item_results.get('image', f"item_{len(results_list)}")

        # Extract captions from the item data
        captions_to_process_list = []
        raw_captions = item_results.get('captions')
        raw_caption_singular = item_results.get('caption')

        if raw_captions and isinstance(raw_captions, list):
            for cap_text in raw_captions:
                if isinstance(cap_text, str) and cap_text.strip():
                    captions_to_process_list.append(cap_text)
        elif raw_caption_singular and isinstance(raw_caption_singular, str) and raw_caption_singular.strip():
            captions_to_process_list.append(raw_caption_singular)
        
        if not captions_to_process_list:
            script_logger.warning(f"No valid captions found for item: {current_item_id}")
            item_results['llm_objects'] = []
            results_list.append(item_results)
            continue

        # Process all captions for this item and aggregate results
        aggregated_llm_objects_for_item = set()

        script_logger.debug(f"Processing {len(captions_to_process_list)} caption(s) for item: {current_item_id}")
        for caption_idx, caption_text in enumerate(captions_to_process_list):
            script_logger.debug(f"  Caption {caption_idx+1}: {caption_text[:100]}...")
            if llm_extractor:
                try:
                    current_llm = llm_extractor.extract_entities(caption_text)
                    if current_llm: 
                        aggregated_llm_objects_for_item.update(current_llm)
                except Exception as e_llm:
                    script_logger.error(f"Entity extraction failed for item {current_item_id}, caption {caption_idx+1}: {e_llm}", exc_info=False)

        item_results['llm_objects'] = sorted(list(aggregated_llm_objects_for_item))
        results_list.append(item_results)

    # Save results to output file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        script_logger.info(f"Created output directory: {output_dir}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        script_logger.info(f"Results saved to {output_path}. Processed {len(results_list)} items.")
    except Exception as e:
        script_logger.error(f"Failed to write output file: {e}", exc_info=True)

    script_logger.info("Entity extraction pipeline completed successfully.")