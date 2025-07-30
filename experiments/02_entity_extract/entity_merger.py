import json
import os
import argparse
import re
import logging
import copy
import inflect
import nltk

try:
    # Check if WordNet corpus is available
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK 'wordnet' resource not found. Downloading...")
    nltk.download('wordnet', quiet=True)

try:
    # Check if Open Multilingual WordNet 1.4 is available
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("NLTK 'omw-1.4' resource not found. Downloading...")
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

# Initialize inflect engine for singularization
p = inflect.engine()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

def save_output_json(data_to_save, output_filepath, logger_instance):
    """Save JSON file with compact formatting for lists in caption_objects and sg_attributes."""
    items_to_format = copy.deepcopy(data_to_save)
    placeholder_map = {}
    placeholder_idx_obj = [0]

    def convert_lists_to_placeholders_recursive(current_data, p_map, p_idx_obj, parent_key=None):
        if isinstance(current_data, dict):
            keys_in_order = list(current_data.keys())
            for key in keys_in_order:
                value = current_data[key]
                if isinstance(value, list):
                    # Apply compact formatting to lists in caption_objects and sg_attributes
                    if key == 'caption_objects' or parent_key == 'sg_attributes':
                        placeholder_key_string = f"__COMPACT_LIST_PLACEHOLDER_{p_idx_obj[0]}__"
                        try:
                            p_map[placeholder_key_string] = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
                            current_data[key] = placeholder_key_string
                            p_idx_obj[0] += 1
                        except TypeError as te:
                            logger_instance.warning(f"Could not JSON dump list for key '{key}' (parent: {parent_key}) due to TypeError: {te}. List: {str(value)[:100]}...")
                elif isinstance(value, dict):
                    convert_lists_to_placeholders_recursive(value, p_map, p_idx_obj, key)

    data_list_to_process = items_to_format if isinstance(items_to_format, list) else [items_to_format]
    for item_dict in data_list_to_process:
        if isinstance(item_dict, dict):
            convert_lists_to_placeholders_recursive(item_dict, placeholder_map, placeholder_idx_obj)

    json_string_with_placeholders = json.dumps(items_to_format, indent=2, ensure_ascii=False)

    for ph_key, compact_list_str in placeholder_map.items():
        json_string_with_placeholders = json_string_with_placeholders.replace(f'"{ph_key}"', compact_list_str)

    try:
        output_dir_path = os.path.dirname(output_filepath)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            logger_instance.info(f"Created output directory: {output_dir_path}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(json_string_with_placeholders)
        logger_instance.info(f"Successfully saved data to '{output_filepath}' with compact lists.")
    except Exception as e:
        logger_instance.error(f"Error writing output file '{output_filepath}': {e}")

def load_json_file(file_path):
    """Load JSON file and return its contents."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unknown error loading {file_path}: {e}")
        return None

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    Natural sorting key for strings like "image1.jpg", "image2.jpg", "image10.jpg".
    If input is a dictionary, extracts the 'image' field for sorting.
    """
    if isinstance(s, dict):
        s = s.get('image', '')
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def is_valid_word(word):
    """Check if it's a valid English word that exists in WordNet"""
    if ' ' in word:
        parts = word.split()
        return wn.synsets(parts[-1]) != [] if parts else False
    return wn.synsets(word) != []

def singularize_object(obj):
    if not obj or not isinstance(obj, str):
        return obj

    obj_stripped = obj.strip()
    if not obj_stripped:
        return obj_stripped

    try:
        if ' ' in obj_stripped:
            words = obj_stripped.split()
            last_word = words[-1]
            singular_last_word = p.singular_noun(last_word)
            
            if singular_last_word: 
                if p.plural(singular_last_word) == last_word and is_valid_word(singular_last_word):
                    words[-1] = singular_last_word
                    return " ".join(words)
                else: 
                    return obj_stripped
            else: 
                return obj_stripped 
        else: 
            singular = p.singular_noun(obj_stripped)
            if singular: 
                if p.plural(singular) == obj_stripped and is_valid_word(singular):
                    return singular
                else:
                    return obj_stripped
            else: 
                return obj_stripped
    except Exception as e:
        logger.warning(f"Error singularizing '{obj_stripped}': {e}, using original form")
        return obj_stripped

def merge_data(c2e_extracted_data, e2e_data):
    _c2e_data_list = []
    if c2e_extracted_data is not None:
        if isinstance(c2e_extracted_data, list):
            _c2e_data_list = c2e_extracted_data
        else:
            logger.warning("c2e_extracted_data is not in the expected list format and will be treated as an empty list for processing.")

    _e2e_data_list = []
    if e2e_data is not None:
        if isinstance(e2e_data, list):
            _e2e_data_list = e2e_data
        else:
            logger.warning("e2e_data is not in the expected list format and will be treated as an empty list for processing.")

    if not _c2e_data_list and not _e2e_data_list:
        logger.warning("Both input data sources are empty or in incorrect format, or do not contain valid entries. Cannot perform merge.")
        return []

    c2e_map = {item['image']: item for item in _c2e_data_list if isinstance(item, dict) and 'image' in item}
    e2e_map = {item['image']: item for item in _e2e_data_list if isinstance(item, dict) and 'image' in item}

    all_image_filenames = sorted(list(set(c2e_map.keys()) | set(e2e_map.keys())), key=natural_sort_key)

    if not all_image_filenames:
        logger.warning("No mergeable image filenames found. Input data may be valid but has no shared image IDs or no image entries.")
        return []

    logger.info(f"Found {len(all_image_filenames)} unique images to process.")

    final_results_list = []

    for image_filename in all_image_filenames:
        merged_item = {'image': image_filename}
        c2e_item = c2e_map.get(image_filename) 
        e2e_item = e2e_map.get(image_filename) 

        # 1. Captions (unchanged)
        all_captions = []
        if c2e_item and isinstance(c2e_item.get('captions'), list):
            all_captions.extend(c2e_item['captions'])
        if e2e_item and isinstance(e2e_item.get('captions'), list):
            all_captions.extend(e2e_item['captions'])
        merged_item['captions'] = all_captions

        # 2. Model (unchanged)
        model_name = None
        c2e_model = c2e_item.get('model') if c2e_item else None
        e2e_model = e2e_item.get('model') if e2e_item else None
        if c2e_model and e2e_model and c2e_model != e2e_model:
            logger.warning(f"Model names for image '{image_filename}' are inconsistent: "
                           f"c2e is '{c2e_model}', e2e is '{e2e_model}'. Will prioritize using c2e's model name.")
        model_name = c2e_model if c2e_model else e2e_model
        merged_item['model'] = model_name

        # 3. Epochs (unchanged)
        epochs_val = None
        c2e_epochs_val = c2e_item.get('epochs') if c2e_item else None
        e2e_epochs_val = e2e_item.get('epochs') if e2e_item else None
        if c2e_epochs_val is not None:
            epochs_val = c2e_epochs_val
        elif e2e_epochs_val is not None:
            epochs_val = e2e_epochs_val
        if c2e_item and e2e_item and c2e_epochs_val is not None and e2e_epochs_val is not None and c2e_epochs_val != e2e_epochs_val:
                 logger.warning(f"Epochs for image '{image_filename}' are inconsistent: "
                               f"c2e is {c2e_epochs_val}, e2e is {e2e_epochs_val}. "
                               f"Current value used: {epochs_val} (prioritized from c2e's value).")
        merged_item['epochs'] = epochs_val

        # 4. Prompt Sets
        prompt_names = set()
        if c2e_item and isinstance(c2e_item.get('prompt_sets'), list):
            prompt_names.update(name for name in c2e_item['prompt_sets'] if isinstance(name, str))
        if e2e_item and isinstance(e2e_item.get('prompt_sets'), list):
            prompt_names.update(name for name in e2e_item['prompt_sets'] if isinstance(name, str))
        merged_item['prompt_sets'] = sorted(list(prompt_names))

        # 5. Caption Objects
        all_original_entities = []
        if c2e_item:
            if isinstance(c2e_item.get('llm_objects'), list):
                all_original_entities.extend(obj.strip() for obj in c2e_item['llm_objects'] if isinstance(obj, str) and obj.strip())
            # sg_objects removed, as scene graph models are no longer used
        if e2e_item:
            if isinstance(e2e_item.get('caption_objects'), list):
                all_original_entities.extend(obj.strip() for obj in e2e_item['caption_objects'] if isinstance(obj, str) and obj.strip())
        
        singular_entities_initial = set()
        for entity in all_original_entities:
            singular_form = singularize_object(entity)
            singular_entities_initial.add(singular_form)
        
        # Apply space phrase priority rule
        # no_space_to_spaced_preference_map is used to record which no-space version was replaced by its spaced version, for use in sg_attributes
        no_space_to_spaced_preference_map = {}
        caption_objects_final_candidates = set(singular_entities_initial) # Initial candidate set

        # To safely modify the set, we iterate over its copy or a list
        # Find all phrases with spaces and their no-space versions
        phrases_with_spaces = {s for s in singular_entities_initial if ' ' in s}

        for spaced_phrase in phrases_with_spaces:
            no_space_version = spaced_phrase.replace(" ", "")
            if no_space_version in singular_entities_initial: # Ensure the no-space version actually exists in the initial set
                # If both spaced and no-space versions exist, prioritize keeping the spaced version
                if no_space_version in caption_objects_final_candidates:
                    caption_objects_final_candidates.remove(no_space_version)
                    logger.info(f"Caption Objects for '{image_filename}': '{no_space_version}' removed in favor of '{spaced_phrase}'.")
                # Record this replacement relationship, ensuring the spaced version is definitely in the final result
                caption_objects_final_candidates.add(spaced_phrase) # Ensure the spaced version exists
                no_space_to_spaced_preference_map[no_space_version] = spaced_phrase
        
        merged_item['caption_objects'] = sorted(list(caption_objects_final_candidates))

        # 6. SG Attributes - Set to empty dictionary due to removal of scene graph extractor
        final_sg_attributes = {}
        # Note: sg_attributes functionality has been removed because scene graph models are no longer used
        merged_item['sg_attributes'] = final_sg_attributes
        
        final_results_list.append(merged_item)

    return final_results_list

def main():
    parser = argparse.ArgumentParser(description="Merge entity extraction results from c2e (processed by extract.py) and e2e pipelines.")
    parser.add_argument("--c2e", required=False, help="JSON file path from extract.py (result after processing c2e.json).")
    parser.add_argument("--e2e", required=False, help="e2e.json file path from batch_llava.")
    parser.add_argument("--output_file", required=True, help="Output JSON file path for merged results.")
    
    args = parser.parse_args()

    if not args.c2e and not args.e2e:
        parser.error("Error: Must specify at least one of --c2e or --e2e.")

    logger.info("Starting processing...")
    
    c2e_data = None
    if args.c2e:
        logger.info(f"C2E extracted file: {args.c2e}")
        c2e_data = load_json_file(args.c2e)
        if c2e_data is None:
            # load_json_file will log errors internally, can simply exit here
            logger.error(f"Processing aborted due to failed loading of C2E file {args.c2e}.")
            return 
    else:
        logger.info("No C2E extracted file provided, will skip.")

    e2e_data = None
    if args.e2e:
        logger.info(f"E2E file: {args.e2e}")
        e2e_data = load_json_file(args.e2e)
        if e2e_data is None:
            logger.error(f"Processing aborted due to failed loading of E2E file {args.e2e}.")
            return
    else:
        logger.info("No E2E file provided, will skip.")
        
    logger.info(f"Output file: {args.output_file}")

    merged_results = merge_data(c2e_data, e2e_data)

    if merged_results: 
        save_output_json(merged_results, args.output_file, logger)
        logger.info(f"Processing completed. Merged file contains {len(merged_results)} entries.")
    else:
        logger.warning("Processing completed, but merged result is empty list or data generation was not successful. No output file written.")

if __name__ == "__main__":
    main()