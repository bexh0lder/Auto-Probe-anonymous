import sys
import json
import os
import logging
import argparse
import yaml
from tqdm import tqdm
import copy # ensure copy is imported
import torch
from PIL import Image

try:
    import inflect
except ImportError:
    # 使用 print 因为 logger 可能尚未配置
    print("WARNING: inflect library not found. Pluralization for OWL-ViT query will be basic (add 's').")
    inflect = None
    sys.exit(1) # Per original script, exit if inflect is missing

try:
    from transformers import AutoProcessor, Owlv2ForObjectDetection
except ImportError:
    print("ERROR: transformers library not found. Please install it: pip install transformers[torch]")
    sys.exit(1)

# --- 全局Logger ---
logger = logging.getLogger("OwlVerifierWithSingleInput")

def setup_main_logger(log_level_str="INFO"):
    """配置全局logger"""
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    if not logger.handlers: # 避免重复添加 handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False

def load_config(config_path):
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            print(f"ERROR: Configuration file {config_path} is empty or invalid.")
            return None

        if '02_entity_extract_owlv2_verifier' not in config_data or config_data['02_entity_extract_owlv2_verifier'] is None:
            print("❌ '02_entity_extract_owlv2_verifier' key is missing or null in the configuration file. Exiting.")
            sys.exit(1)
        config_data = config_data['02_entity_extract_owlv2_verifier']

        config_data.setdefault('io', {})
        config_data.setdefault('owl_vit', {})
        config_data.setdefault('logging', {})
        config_data['logging'].setdefault('level', 'INFO')

        io_conf = config_data['io']
        if not io_conf.get('image_base_dir'):
            print(f"ERROR: Config {config_path} 'io' section is missing 'image_base_dir'.")
            return None
        io_conf.setdefault('output_json_path', 'owl_verified_output_default.json')

        owl_conf = config_data['owl_vit']
        if not owl_conf.get('model_path'):
            print(f"ERROR: Config {config_path} 'owl_vit' section is missing 'model_path'.")
            return None
        owl_conf.setdefault('initial_detection_threshold', 0.05)
        owl_conf.setdefault('upper_confidence_threshold', 0.7)
        owl_conf.setdefault('lower_confidence_threshold', 0.1)

        return config_data
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}"); return None
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML file {config_path}: {e}"); return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading config {config_path}: {e}"); return None


class OwlVitVerifier:
    def __init__(self, config_dict):
        self.config = config_dict
        self.logger = logger

        owl_conf = self.config.get('owl_vit', {})
        self.initial_threshold = float(owl_conf.get('initial_detection_threshold'))
        self.upper_threshold = float(owl_conf.get('upper_confidence_threshold'))
        self.lower_threshold = float(owl_conf.get('lower_confidence_threshold'))

        self.logger.info(f"OwlVitVerifier initialized with thresholds: Initial={self.initial_threshold}, Upper={self.upper_threshold}, Lower={self.lower_threshold}")

        owl_model_path = owl_conf.get('model_path')

        self.pluralizer = None
        if inflect:
            try:
                self.pluralizer = inflect.engine()
                self.logger.info("Inflect pluralizer initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Inflect pluralizer: {e}. Pluralization will be basic.")
        else:
            self.logger.warning("Inflect library not available. Pluralization will be basic (add 's').")

        self.model, self.processor, self.device = None, None, None
        if os.path.exists(owl_model_path) and os.path.isdir(owl_model_path):
            self._load_owl_model(owl_model_path)
        else:
            self.logger.error(f"OWL-ViT model_path '{owl_model_path}' is invalid or not a directory. Verification non-functional.")

        if not self.model or not self.processor:
            self.logger.warning("OWL-ViT model/processor not loaded. Verification will be non-functional.")

    def _load_owl_model(self, model_path):
        try:
            self.logger.info(f"Loading OWL-ViT model from: {model_path}...")
            if torch.cuda.is_available(): self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = "mps"
            else: self.device = "cpu"
            self.logger.info(f"Using device for OWL-ViT: {self.device}")

            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.logger.info("OWL-ViT model and processor loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load OWL-ViT model from '{model_path}': {e}", exc_info=True)
            self.model, self.processor = None, None

    def verify_objects(self, image_path, object_candidates_input):
        object_candidates = list(object_candidates_input) if object_candidates_input else []
        
        # MODIFICATION START: Initialize details dict based on object_candidates for early returns
        initial_details_dict = {cand: 0.0 for cand in object_candidates}
        # MODIFICATION END

        if not self.model or not self.processor:
            self.logger.warning(f"Cannot verify for {image_path}: OWL-ViT model/processor not loaded.")
            # MODIFICATION START: Add initial_details_dict to return
            return [], object_candidates, [], initial_details_dict
            # MODIFICATION END

        if not object_candidates:
            # MODIFICATION START: Add empty dict for details
            return [], [], [], {}
            # MODIFICATION END

        if not os.path.exists(image_path):
            self.logger.warning(f"Image not found: {image_path}. All candidates considered absent.")
            # MODIFICATION START: Add initial_details_dict to return
            return [], object_candidates, [], initial_details_dict
            # MODIFICATION END

        expanded_queries_set, query_to_original_map, valid_original_candidates = set(), {}, []
        for candidate in object_candidates:
            if not isinstance(candidate, str) or not candidate.strip(): continue
            valid_original_candidates.append(candidate)
            cl = candidate.lower()
            expanded_queries_set.add(cl); query_to_original_map[cl] = candidate

            plural_form = None
            if self.pluralizer:
                try:
                    plural_form = self.pluralizer.plural_noun(cl)
                except Exception as e:
                    self.logger.debug(f"Inflect failed for '{cl}': {e}. Using basic pluralization.")
            
            pl = (plural_form if plural_form else cl + "s").lower()

            if pl and pl != cl:
                expanded_queries_set.add(pl)
                query_to_original_map.setdefault(pl, candidate)

        final_queries_for_owl = sorted(list(expanded_queries_set))
        
        # MODIFICATION START: Initialize candidate_max_scores using valid_original_candidates
        candidate_max_scores = {orig_cand: 0.0 for orig_cand in valid_original_candidates}
        # MODIFICATION END

        if not final_queries_for_owl:
            # MODIFICATION START: Return candidate_max_scores (all zeros)
            return [], list(valid_original_candidates), [], candidate_max_scores
            # MODIFICATION END

        self.logger.debug(f"OWL-ViT for {os.path.basename(image_path)}: {len(final_queries_for_owl)} queries: {final_queries_for_owl[:10]}...")
        # candidate_max_scores = {orig_cand: 0.0 for orig_cand in valid_original_candidates} # Moved up
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=[final_queries_for_owl], images=image, return_tensors="pt").to(self.device)
            with torch.no_grad(): outputs = self.model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
            results_pp = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.initial_threshold,
                text_labels=[final_queries_for_owl] 
            )
            img_res = results_pp[0]

            if isinstance(img_res, dict) and "text_labels" in img_res and "scores" in img_res:
                for score_t, label_str in zip(img_res["scores"], img_res["text_labels"]):
                    detected_query_string = label_str
                    original_cand_name = query_to_original_map.get(detected_query_string)
                    if original_cand_name:
                        candidate_max_scores[original_cand_name] = max(candidate_max_scores[original_cand_name], score_t.item())
        
        except Exception as e: self.logger.error(f"OWL-ViT error for {os.path.basename(image_path)}: {e}", exc_info=True)

        present, absent, uncertain = [], [], []
        for cand in valid_original_candidates:
            score = candidate_max_scores[cand] # Score is already from candidate_max_scores
            if score >= self.upper_threshold: present.append({"object": cand, "score": round(score, 4)})
            elif score <= self.lower_threshold: absent.append(cand)
            else: uncertain.append({"object": cand, "score": round(score, 4)})
        
        self.logger.debug(f"Result for {os.path.basename(image_path)} - Present: {len(present)}, Absent: {len(absent)}, Uncertain: {len(uncertain)}")
        # MODIFICATION START: Return candidate_max_scores as the fourth item
        return present, absent, uncertain, candidate_max_scores
        # MODIFICATION END

# --- 辅助 JSON 加载/保存函数 ---
def _load_json_data_helper(filepath, description, current_logger):
    if not filepath:
        current_logger.info(f"{description} file path not provided. Returning empty list.")
        return []
    if not os.path.exists(filepath):
        current_logger.error(f"{description} file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        current_logger.info(f"Loaded {description} from: {filepath}")
        if not isinstance(data, list):
            current_logger.error(f"Error: {description} data from {filepath} is not a list. Type: {type(data)}")
            return None
        return data
    except json.JSONDecodeError as e:
        current_logger.error(f"Error decoding JSON from {description} file {filepath}: {e}", exc_info=True)
        return None
    except Exception as e:
        current_logger.error(f"Error loading/parsing {description} file {filepath}: {e}", exc_info=True)
        return None

def _save_json_data_helper(data, filepath, description, current_logger):
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            current_logger.info(f"Created output directory: {output_dir}")

        # MODIFICATION START: JSON compaction logic
        items_to_format = copy.deepcopy(data)
        placeholder_map = {}
        placeholder_idx_obj = [0]  # Use list for pass-by-reference mutable integer

        # Keys whose list values should be directly compacted
        DIRECT_COMPACT_LIST_KEYS = {'caption_objects', 'hallucinated_objects', 'ground_truth_objects', 'candidate_objects'}

        def convert_lists_to_placeholders_recursive(current_data_node, p_map, p_idx_ref, parent_key_name=None):
            if isinstance(current_data_node, dict):
                for key in list(current_data_node.keys()): # Iterate over a copy for safe modification
                    value = current_data_node[key] # 'value' is assigned here

                    if isinstance(value, list):
                        # 'value' is a list. Check if it needs compaction.
                        if key in DIRECT_COMPACT_LIST_KEYS or parent_key_name == 'sg_attributes':
                            # Yes, compact this list
                            placeholder_str = f"__COMPACT_LIST_PLACEHOLDER_{p_idx_ref[0]}__"
                            try:
                                p_map[placeholder_str] = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
                                current_data_node[key] = placeholder_str
                                p_idx_ref[0] += 1
                            except TypeError as te:
                                current_logger.warning(f"Compaction: Could not JSON dump list for key '{key}' "
                                                       f"(parent key: {parent_key_name}) due to TypeError: {te}. "
                                                       f"List: {str(value)[:100]}...")
                        else:
                            # It's a list, but not for direct compaction.
                            # Recurse into it in case it contains dicts or other lists that might need processing.
                            convert_lists_to_placeholders_recursive(value, p_map, p_idx_ref, parent_key_name)
                    
                    elif isinstance(value, dict):
                        # 'value' is a dictionary.
                        # Recurse into it. The current 'key' becomes the parent_key_name for the next level.
                        convert_lists_to_placeholders_recursive(value, p_map, p_idx_ref, key)
                    
                    # If 'value' is neither a list nor a dict (e.g., string, int, bool),
                    # we do nothing further with it for compaction recursion.

            elif isinstance(current_data_node, list):
                # current_data_node itself is a list. Iterate its items.
                # The parent_key_name is inherited from the context where this list was found.
                for item in current_data_node:
                    convert_lists_to_placeholders_recursive(item, p_map, p_idx_ref, parent_key_name)
        
        # Apply the conversion starting from the root of the data to be saved.
        # The 'data' is typically a list of dictionaries.
        convert_lists_to_placeholders_recursive(items_to_format, placeholder_map, placeholder_idx_obj)

        json_string_with_placeholders = json.dumps(items_to_format, ensure_ascii=False, indent=2)

        for ph_key, compact_list_str in placeholder_map.items():
            # Ensure replacing the quoted placeholder string
            json_string_with_placeholders = json_string_with_placeholders.replace(f'"{ph_key}"', compact_list_str)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_string_with_placeholders)
        # MODIFICATION END

        current_logger.info(f"{description} successfully saved to '{filepath}'.")
    except Exception as e:
        current_logger.error(f"Failed to write {description} to file '{filepath}': {e}", exc_info=True)

# --- Main 函数 ---
def main():
    cli_parser = argparse.ArgumentParser(description="OWL-ViT Verifier with dual thresholds, processing a single JSON input file.")
    cli_parser.add_argument("--config", type=str, default="config_owl_verifier.yaml", help="Path to YAML config.")
    cli_parser.add_argument("--input_file", required=True, help="Path to the input JSON file containing a list of items to verify.")
    cli_parser.add_argument("--output_file", help="Path for the final JSON output. Overrides config's 'output_json_path'.")
    cli_parser.add_argument("--image_dir", help="Path to the image directory. Overrides config's 'image_base_dir'.")
    cli_parser.add_argument("--log_level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level. Overrides config.")
    cli_args = cli_parser.parse_args()

    config_data = load_config(cli_args.config)
    if not config_data:
        sys.exit(1)

    final_log_level = cli_args.log_level if cli_args.log_level else config_data.get('logging', {}).get('level', 'INFO')
    setup_main_logger(final_log_level)

    logger.info(f"Config loaded from '{cli_args.config}'. Log level: {final_log_level}")
    logger.info(f"Input JSON: {cli_args.input_file}")

    io_conf = config_data.get('io', {})
    input_file_path = cli_args.input_file
    output_file_path = cli_args.output_file if cli_args.output_file else io_conf.get('output_json_path')
    image_base_dir = cli_args.image_dir if cli_args.image_dir else io_conf.get('image_base_dir')

    if not output_file_path: logger.critical("Output file path not resolved (missing in CLI and config). Exiting."); sys.exit(1)
    if not image_base_dir or not os.path.isdir(image_base_dir):
        logger.critical(f"'image_base_dir' ('{image_base_dir}') not specified or invalid in config. Cannot proceed."); sys.exit(1)

    input_data_list = _load_json_data_helper(input_file_path, "Input data", logger)
    if input_data_list is None:
        logger.critical(f"Input JSON file {input_file_path} failed to load or is not in the expected list format. Exiting.")
        sys.exit(1)
    if not input_data_list:
        logger.warning(f"Input JSON file {input_file_path} is empty. No items to process.")
        _save_json_data_helper([], output_file_path, "Final verified data (empty input)", logger)
        sys.exit(0)

    logger.info(f"Successfully loaded {len(input_data_list)} items from {input_file_path}.")

    final_results_to_save = []

    try:
        verifier = OwlVitVerifier(config_data)
    except Exception as e:
        logger.critical(f"Failed to initialize OwlVitVerifier with loaded config: {e}", exc_info=True); sys.exit(1)

    if not verifier.model or not verifier.processor:
        logger.critical("OWL-ViT model/processor failed to load in verifier (check previous logs). Cannot proceed."); sys.exit(1)

    for current_item_data in tqdm(input_data_list, desc="Verifying items with OWL-ViT", unit="item", disable=logger.level > logging.INFO):
        output_item = copy.deepcopy(current_item_data)
        image_filename = output_item.get("image")

        raw_candidates = output_item.get("candidate_objects", [])
        if isinstance(raw_candidates, list):
            candidates_for_owl_input = sorted(list(set(str(c).strip() for c in raw_candidates if isinstance(c, str) and str(c).strip())))
        else:
            logger.warning(f"Item for image '{image_filename if image_filename else 'UNKNOWN'}' has 'candidate_objects' that is not a list or is missing. Treating as no candidates. Item sample: {str(output_item)[:200]}")
            candidates_for_owl_input = []
        
        owl_verification_details = {}
        
        # MODIFICATION START: Initialize _discarded_uncertain_objects as dict for skipped cases
        discarded_uncertain_objects_dict = {}
        # MODIFICATION END

        if not image_filename or not isinstance(image_filename, str) or not image_filename.strip():
            logger.warning(f"Skipping item (no valid 'image' field): {str(output_item)[:100]}")
            output_item["ground_truth_objects"] = []
            output_item["hallucinated_objects"] = list(candidates_for_owl_input)
            # output_item["_discarded_uncertain_objects"] = [] # 旧代码
            owl_verification_details = {cand: 0.0 for cand in candidates_for_owl_input}
            # discarded_uncertain_objects_dict 保持为 {} (正确)
        elif not candidates_for_owl_input:
            logger.info(f"No valid candidates to verify for image '{image_filename}'. All original fields preserved. Item sample: {str(output_item)[:100]}")
            output_item["ground_truth_objects"] = []
            output_item["hallucinated_objects"] = []
            # output_item["_discarded_uncertain_objects"] = [] # 旧代码
            # owl_verification_details 和 discarded_uncertain_objects_dict 保持为 {} (正确)
        else:
            full_image_path = os.path.join(image_base_dir, image_filename)
            present, absent, uncertain, returned_details = verifier.verify_objects(full_image_path, candidates_for_owl_input)
            owl_verification_details = returned_details 

            output_item["ground_truth_objects"] = [obj['object'] for obj in present]
            output_item["hallucinated_objects"] = absent
            # output_item["_discarded_uncertain_objects"] = uncertain # 旧代码

            # MODIFICATION START: Convert 'uncertain' list to the new dictionary format
            discarded_uncertain_objects_dict = {obj_info['object']: obj_info['score'] for obj_info in uncertain}
            # MODIFICATION END
        
        # 按照字典的权值从大到小排序
        owl_verification_details = dict(sorted(owl_verification_details.items(), key=lambda x: x[1], reverse=True))
        discarded_uncertain_objects_dict = dict(sorted(discarded_uncertain_objects_dict.items(), key=lambda x: x[1], reverse=True))
        
        output_item["_owl_verified_details"] = owl_verification_details
        # MODIFICATION START: Assign the new dictionary format
        output_item["_discarded_uncertain_objects"] = discarded_uncertain_objects_dict
        # MODIFICATION END
            
        final_results_to_save.append(output_item)

    logger.info(f"OWL-ViT verification finished. Processed {len(final_results_to_save)} items.")
    _save_json_data_helper(final_results_to_save, output_file_path, "Final verified data", logger)
    logger.info(f"--- OWL-ViT Verification (single input) Complete --- Output: {output_file_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()