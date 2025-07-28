import json
import yaml
import time
import logging
import argparse
import os
import re
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import copy

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")


class DetectionConsistencyCorrector:
    def __init__(self, config_path: str = "config_detection_corrector.yaml"):
        """Initialize detection consistency corrector."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.system_prompt = self._load_system_prompt()
        self.user_prompt_template = self._load_user_prompt_template()
        self.client = self._setup_api_client()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if '02_entity_extract_consistency_corrector' not in config:
            raise ValueError("Configuration must contain '02_entity_extract_consistency_corrector' section")

        return config['02_entity_extract_consistency_corrector']

    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())

        logger = logging.getLogger('DetectionConsistencyCorrector')
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
        """Load prompt content from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Prompt file '{file_path}' is empty.")
        return content

    def _load_system_prompt(self) -> Optional[str]:
        """Load system prompt"""
        system_prompt_path = self.config.get('llm', {}).get('system_prompt_path')
        if not system_prompt_path:
            self.logger.info("No system_prompt_path provided. Proceeding without a specific system prompt.")
            return None
        
        try:
            self.logger.info(f"Loading system prompt from: {system_prompt_path}")
            system_prompt = self._load_prompt_from_file(system_prompt_path)
            if system_prompt:
                self.logger.info("System prompt loaded successfully.")
            return system_prompt
        except FileNotFoundError:
            self.logger.warning(f"System prompt file not found: '{system_prompt_path}'. Proceeding without it.")
            return None
        except ValueError as e:
            self.logger.warning(f"System prompt file issue: {e}. Proceeding without it.")
            return None
        except IOError as e:
            self.logger.warning(f"System prompt file IO error: {e}. Proceeding without it.")
            return None

    def _load_user_prompt_template(self) -> str:
        """Load user prompt template"""
        user_prompt_path = self.config.get('llm', {}).get('user_prompt_path')
        if not user_prompt_path:
            self.logger.critical("'user_prompt_path' is missing in the configuration.")
            raise ValueError("Missing user_prompt_path. This is mandatory for DetectionConsistencyCorrector.")
        
        try:
            self.logger.info(f"Loading user prompt template from: {user_prompt_path}")
            user_prompt_template = self._load_prompt_from_file(user_prompt_path)
            self.logger.info("User prompt template loaded successfully.")
            return user_prompt_template
        except Exception as e:
            self.logger.critical(f"Could not load mandatory user prompt file '{user_prompt_path}': {e}.")
            raise

    def _setup_api_client(self):
        """Setup API client"""
        api_config = self.config['llm']
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        api_key = api_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found in config or environment variable OPENAI_API_KEY")

        client_kwargs = {'api_key': api_key}
        if api_config.get('base_url'):
            client_kwargs['base_url'] = api_config['base_url']

        return OpenAI(**client_kwargs)

    def _call_llm_api(self, detected_objects: List[str], not_detected_objects: List[str], uncertain_objects: List[str]) -> Dict[str, Any]:
        """Call LLM API for detection consistency correction"""
        api_config = self.config['llm']
        
        if "{detection_results}" not in self.user_prompt_template:
            self.logger.error("User prompt template is missing the '{detection_results}' placeholder.")
            return None

        input_data = {
            "detected_objects": detected_objects,
            "not_detected_objects": not_detected_objects,
            "uncertain_objects": uncertain_objects
        }

        try:
            user_prompt = self.user_prompt_template.format(
                detection_results=json.dumps(input_data, ensure_ascii=False)
            )
        except KeyError:
            self.logger.error("Failed to format user prompt. Ensure '{detection_results}' placeholder is correct.")
            return None

        messages_for_api = []
        if self.system_prompt:
            messages_for_api.append({"role": "system", "content": self.system_prompt})
        messages_for_api.append({"role": "user", "content": user_prompt})

        max_retries = api_config.get('max_retries', 3)
        timeout = api_config.get('timeout', 30)

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                response = self.client.chat.completions.create(
                    model=api_config['model_name'],
                    messages=messages_for_api,
                    timeout=timeout,
                    temperature=0.1
                )
                
                result_text = response.choices[0].message.content.strip()
                self.logger.debug(f"LLM raw response: {result_text[:200]}...")
                
                # Extract JSON answer
                corrected_result = self._extract_final_answer(result_text)
                if corrected_result:
                    return corrected_result
                
                self.logger.warning(f"Could not extract valid JSON from response on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        self.logger.error(f"All {max_retries} API call attempts failed.")
        return None

    def _extract_final_answer(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Extract final JSON answer from response with reasoning"""
        try:
            # Method 1: Look for JSON after "Step 2: Final Answer"
            step2_start = llm_response.find("Step 2: Final Answer")
            if step2_start != -1:
                json_part = llm_response[step2_start:]
                start = json_part.find('{')
                end = json_part.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = json_part[start:end]
                    parsed = json.loads(json_str)
                    # Validate contains required fields
                    if all(key in parsed for key in ['detected_objects', 'not_detected_objects', 'uncertain_objects']):
                        return parsed
            
            # Method 2: Regex extract last complete JSON containing three fields
            json_pattern = r'\{[^{}]*"detected_objects"[^{}]*"not_detected_objects"[^{}]*"uncertain_objects"[^{}]*\}'
            matches = re.findall(json_pattern, llm_response, re.DOTALL)
            if matches:
                return json.loads(matches[-1])
            
            # Method 3: Find any JSON containing the three target fields
            json_blocks = re.findall(r'\{[^{}]*\}', llm_response)
            for block in reversed(json_blocks):  
                try:
                    parsed = json.loads(block)
                    if all(key in parsed for key in ['detected_objects', 'not_detected_objects', 'uncertain_objects']):
                        return parsed
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error extracting JSON from response: {e}")
        
        return None

    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item"""
        image_name = item.get('image', 'unknown')
        self.logger.debug(f"Processing item: {image_name}")

        result_item = copy.deepcopy(item)
        
        detected_objects = item.get('ground_truth_objects', [])
        not_detected_objects = item.get('hallucinated_objects', [])
        
        discarded_uncertain_data = item.get('_discarded_uncertain_objects', {})
        uncertain_objects = list(discarded_uncertain_data.keys()) if isinstance(discarded_uncertain_data, dict) else []

        if not isinstance(detected_objects, list):
            self.logger.warning(f"Item {image_name}: ground_truth_objects is not a list. Skipping correction.")
            return result_item

        if not isinstance(not_detected_objects, list):
            self.logger.warning(f"Item {image_name}: hallucinated_objects is not a list. Skipping correction.")
            return result_item

        if not isinstance(uncertain_objects, list):
            self.logger.warning(f"Item {image_name}: uncertain_objects is not a list. Skipping correction.")
            return result_item

        if not detected_objects and not not_detected_objects and not uncertain_objects:
            self.logger.warning(f"Item {image_name}: All detection lists are empty. Skipping correction.")
            return result_item

        self.logger.info(f"Item {image_name}: Correcting consistency for {len(detected_objects)} detected, {len(not_detected_objects)} not detected, and {len(uncertain_objects)} uncertain objects.")

        correction_result = self._call_llm_api(detected_objects, not_detected_objects, uncertain_objects)
        
        if correction_result:
            result_item['ground_truth_objects'] = correction_result.get('detected_objects', detected_objects)
            result_item['hallucinated_objects'] = correction_result.get('not_detected_objects', not_detected_objects)
            
            self.logger.info(
                f"Item {image_name}: Correction completed. "
                f"Detected: {len(detected_objects)} → {len(result_item['ground_truth_objects'])}, "
                f"Not detected: {len(not_detected_objects)} → {len(result_item['hallucinated_objects'])}, "
                f"Uncertain: {len(uncertain_objects)} (unchanged)"
            )
        else:
            self.logger.error(f"Item {image_name}: Correction failed. Keeping original results.")
            result_item['correction_error'] = "Failed to correct detection consistency"

        return result_item

    def process_file(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
        input_path = input_file if input_file is not None else self.config['io']['input_file']
        output_path = output_file if output_file is not None else self.config['io']['output_file']

        if not input_path:
            self.logger.error("Input file path not specified. Use --input_file or set in config file.")
            return
        if not output_path:
            self.logger.error("Output file path not specified. Use --output_file or set in config file.")
            return

        self.logger.info(f"Starting detection consistency correction process")
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
        delay_between_requests = self.config.get('processing', {}).get('delay_between_requests', 1.0)

        for i, item in enumerate(tqdm(input_data, desc="Processing items")):
            try:
                processed_item = self.process_single_item(item)
                processed_data.append(processed_item)

                if i < len(input_data) - 1 and delay_between_requests > 0:
                    time.sleep(delay_between_requests)

            except Exception as e:
                image_name_err = item.get('image', f'unknown_item_index_{i}')
                self.logger.error(f"Critical error processing item {image_name_err}: {e}", exc_info=True)
                
                fallback_item = copy.deepcopy(item)
                fallback_item['correction_error'] = f"Critical error during processing: {e}"
                processed_data.append(fallback_item)
        
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Successfully saved {len(processed_data)} processed items to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save output file '{output_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Detection Consistency Correction Script")
    parser.add_argument("--config", type=str, default="config_detection_corrector.yaml",
                        help="Path to configuration file")
    parser.add_argument("--input_file", type=str,
                        help="Input JSON file path (overrides config setting)")
    parser.add_argument("--output_file", type=str,
                        help="Output JSON file path (overrides config setting)")

    args = parser.parse_args()

    try:
        corrector = DetectionConsistencyCorrector(args.config)
        corrector.process_file(args.input_file, args.output_file)
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Configuration or Setup Error: {e}")
        exit(1)
    except ImportError as e:
        print(f"Import Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
        exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()