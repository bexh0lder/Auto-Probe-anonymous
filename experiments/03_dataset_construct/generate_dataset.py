# generate_dataset.py
import json
import argparse
import os
import re
import time
import logging # For logging
from tqdm import tqdm
import openai
import sys
import yaml # For loading YAML config
from collections import Counter

INVALID_ENTITY_PATTERNS = ['object', 'entity', 'entity1', 'entity2', 'entity3', 'hallucinated_objects', 'object1', 'object2', 'example object',"woman", "man", "person", "lady", "gentleman","individual", "pepper", "cup", "bottle", "plant", "tree", "branch","building","water"]


# --- Helper JSON loading/saving functions ---
def _load_json_data_helper(filepath, description, logger):
    """Helper function to load JSON data."""
    if not filepath:
        logger.error(f"{description} file path not provided.")
        return None
    if not os.path.exists(filepath):
        logger.error(f"{description} file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {description} from: {filepath}")
        if not isinstance(data, list): # Assuming top-level is a list of items
            logger.error(f"Error: {description} data from {filepath} is not a list. Type: {type(data)}")
            return None
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {description} file - {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {description} file - {filepath}: {e}", exc_info=True)
        return None

def _save_json_data_helper(data, filepath, description, logger):
    """Helper function to save JSON data."""
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{description} successfully saved to '{filepath}'.")
    except IOError as e:
        logger.error(f"Failed to write {description} to file '{filepath}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving {description} to '{filepath}': {e}", exc_info=True)

# --- Prompt template loading functions ---
def _load_prompt_template_file(file_path, logger):
    """Loads a prompt template string from a file."""
    if not file_path: # Path can be None if a particular template is not configured
        logger.warning(f"Prompt template file path not provided for: {file_path}") # Clarify which path is missing
        return None
    if not os.path.exists(file_path):
        logger.error(f"Prompt template file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template '{file_path}': {e}")
        return None

# --- OpenAI API Calling Class (for AttriBait) ---
class SingleChoiceQuestionGenerator:
    def __init__(self, api_key=None, base_url=None, model_name="gpt-4o", logger=None, system_prompt_content=None): # MODIFIED: Added system_prompt_content
        self.client = None
        self.logger = logger if logger else logging.getLogger(__name__)
        if api_key:
            try:
                self.client = openai.OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
                self.logger.info(f"OpenAI client initialized for model '{model_name}' at base_url '{base_url if base_url else 'default'}'.")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
        else:
            self.logger.warning("API Key not provided. SingleChoiceQuestionGenerator client not initialized.")

        self.model_name = model_name
        self.api_tokens_used_total = 0
        
        if not system_prompt_content:
            self.logger.error("System prompt content not provided to SingleChoiceQuestionGenerator. Enhanced question generation may fail.")
            # Depending on strictness, you might want to raise an error:
            # raise ValueError("System prompt content is required for SingleChoiceQuestionGenerator.")
        self.system_prompt = system_prompt_content # MODIFIED: Store system_prompt

    def generate_questions(self, prepared_prompt_content, max_retries=3, retry_delay_seconds=2):
        if not self.client:
            self.logger.error("OpenAI client not initialized. Cannot generate questions.")
            return []
        if not prepared_prompt_content:
            self.logger.error("Prepared prompt content is empty. Cannot generate questions.")
            return []
        if not self.system_prompt: # MODIFIED: Check if system_prompt was loaded/provided
            self.logger.error("System prompt is not configured for the generator. Cannot generate questions.")
            return []
        
        current_call_tokens = 0
        retries = 0
        response = None # Initialize response to None

        while retries <= max_retries:
            try:
                self.logger.debug(f"Attempting API call (retry {retries}/{max_retries}) with prompt: {prepared_prompt_content[:200]}...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt}, # MODIFIED: Use stored system_prompt
                        {"role": "user", "content": prepared_prompt_content}
                    ],
                )

                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    current_call_tokens = prompt_tokens + completion_tokens
                    self.api_tokens_used_total += current_call_tokens
                
                response_content = response.choices[0].message.content
                self.logger.debug(f"Raw response content: {response_content[:500]}...") # Log raw response
                response_content_cleaned = response_content.strip()
                if response_content_cleaned.startswith("```json"):
                    response_content_cleaned = response_content_cleaned[len("```json"):]
                if response_content_cleaned.endswith("```"):
                    response_content_cleaned = response_content_cleaned[:-len("```")]
                response_content_cleaned = response_content_cleaned.strip()
                
                # Handle cases where the response might be an empty string after cleaning
                if not response_content_cleaned:
                    self.logger.warning("API response content was empty after cleaning. Returning empty list.")
                    return []

                result = json.loads(response_content_cleaned)

                if isinstance(result, list):
                # If directly returns a question list
                    generated_q_list = result
                elif isinstance(result, dict):
                    # If returns a dictionary containing questions key
                    generated_q_list = result.get("questions", [])
                    if not isinstance(generated_q_list, list): # Ensure questions is a list
                        self.logger.error(f"Content of 'questions' key is not a list: {type(generated_q_list)}. Original dict: {result}")
                        generated_q_list = []
                else:
                    self.logger.error(f"Unexpected response JSON format: {type(result)}. Content: {response_content_cleaned[:200]}")
                    generated_q_list = []
                
                self.logger.debug(f"API call successful. Tokens used this call: {current_call_tokens}. Generated {len(generated_q_list)} questions.")
                return generated_q_list

            except json.JSONDecodeError as e_json:
                retries += 1
                error_message = f"API call failed (JSONDecodeError) for model '{self.model_name}' (retry {retries}/{max_retries+1}): {e_json}"
                failed_response_content_on_error = response.choices[0].message.content if response and hasattr(response, 'choices') and response.choices else 'No response object or choices available.'
                self.logger.error(f"{error_message}. Failed raw response: {failed_response_content_on_error[:500]}...")
                if retries > max_retries:
                    return []
                time.sleep(retry_delay_seconds)
            except Exception as e:
                retries += 1
                error_message = f"API call failed (General Exception) for model '{self.model_name}' (retry {retries}/{max_retries+1}): {e}"
                
                if retries > max_retries:
                    self.logger.error(error_message)
                    try:
                        failed_response_content = response.choices[0].message.content if response and hasattr(response, 'choices') and response.choices else 'No response object or choices available.'
                        self.logger.error(f"Failed raw response content on final attempt: {failed_response_content[:500]}...")
                    except Exception as E_resp:
                        self.logger.error(f"Error getting failed response content: {E_resp}")
                    return [] 
                else:
                    self.logger.warning(f"{error_message}, retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
        return [] # Should be unreachable if max_retries is handled correctly, but as a safeguard

    def get_total_tokens_used(self):
        return self.api_tokens_used_total

# --- Data validation functions ---
def validate_generated_questions(questions, logger):
    valid_questions = []
    invalid_count = 0
    for q in questions:
        entity = q.get("entity", "").strip().lower()
        question_text = q.get("text", "").lower()

        is_placeholder_entity = False
        if not entity or entity in INVALID_ENTITY_PATTERNS or entity.startswith('entity') or entity.startswith('object') or entity.startswith('placeholder'):
            is_placeholder_entity = True
        
        # Check for placeholder text in questions like "Is there a [object]..."
        if "[object]" in question_text or "[entity name]" in question_text or "placeholder object" in question_text:
            logger.warning(f"Placeholder text found in question for entity '{q.get('entity')}': '{q.get('text')}', skipping question {q.get('question_id')}")
            invalid_count += 1
            continue

        if is_placeholder_entity:
            logger.warning(f"Invalid or placeholder entity name detected: '{q.get('entity')}', skipping question {q.get('question_id')}")
            invalid_count += 1
            continue
            
        valid_questions.append(q)

    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} questions with invalid entity names or placeholder text.")
    return valid_questions

# --- Generate basic yes/no questions ($Q$) ---
def generate_yes_no_questions(eval_items, starting_question_id, include_hallucinated, include_ignored, logger):
    qa_dataset = []
    question_id = starting_question_id
    for item in tqdm(eval_items, desc="Generating Basic Yes/No Questions", unit="item", disable=logger.level > logging.INFO):
        image_id = item.get("image")
        if not image_id: 
            logger.warning(f"Item missing 'image' field, skipping basic Qs: {str(item)[:100]}")
            continue
        item_metadata = {
            "model": item.get("model", "unknown"),
            "epochs": item.get("epochs", None),
            "prompt_sets": item.get("prompt_sets", [])
        }
        if include_hallucinated:
            hallucinated_list = item.get("hallucinated_objects", []) 
            if not isinstance(hallucinated_list, list): 
                logger.warning(f"Item {image_id} 'hallucinated_objects' is not a list, skipping. Value: {hallucinated_list}")
                hallucinated_list = []

            for obj_name in hallucinated_list:
                if obj_name.lower() in INVALID_ENTITY_PATTERNS or obj_name.startswith('entity') or obj_name.startswith('object') or obj_name.startswith('placeholder'):
                    logger.warning(f"Skipping basic question for placeholder entity '{obj_name}' in image {image_id}")
                    continue
                if not obj_name or not isinstance(obj_name, str): 
                    logger.warning(f"Item {image_id} has invalid hallucinated object name, skipping. Value: {obj_name}")
                    continue
                qa_dataset.append({
                    "question_id": question_id, "image": image_id,
                    "text": f"Is there a {obj_name} in the image?", "label": "no",
                    **item_metadata, "entity": obj_name, 
                    "entity_type": "hallucinated", "question_type": "basic"
                })
                question_id += 1
        if include_ignored:
            real_obj_list = item.get("ground_truth_objects", []) 
            if not isinstance(real_obj_list, list): 
                logger.warning(f"Item {image_id} 'ground_truth_objects' is not a list, skipping. Value: {real_obj_list}")
                real_obj_list = []
            for obj_name in real_obj_list:
                if not obj_name or not isinstance(obj_name, str): 
                    logger.warning(f"Item {image_id} has invalid ground_truth object name, skipping. Value: {obj_name}")
                    continue
                qa_dataset.append({
                    "question_id": question_id, "image": image_id,
                    "text": f"Is there a {obj_name} in the image?", "label": "yes",
                    **item_metadata, "entity": obj_name, 
                    "entity_type": "ground_truth", "question_type": "basic"
                })
                question_id += 1
    return qa_dataset, question_id

# --- Generate enhanced multiple-choice questions ($Q_p$ - AttriBait) ---
def generate_enhanced_questions(eval_items, config_enhanced, starting_question_id, logger):
    """
    Generates enhanced multiple-choice questions (AttriBait) for hallucinated objects.
    Includes logic to re-request questions for missing objects and logs entities that generate multiple questions.
    """
    all_enhanced_questions = []
    question_id_counter = starting_question_id
    records_processed_for_enhanced = 0
    total_api_tokens_for_enhanced = 0
    
    # Add a new variable to record problematic entities
    multi_question_entities_log = {}

    prompt_with_attrs_path = config_enhanced.get('prompt_template_with_attributes')
    prompt_no_attrs_path = config_enhanced.get('prompt_template_without_attributes')
    system_prompt_file_path = config_enhanced.get('system_prompt_file')

    prompt_template_with_attrs_str = _load_prompt_template_file(prompt_with_attrs_path, logger)
    prompt_template_no_attrs_str = _load_prompt_template_file(prompt_no_attrs_path, logger)
    system_prompt_content_str = _load_prompt_template_file(system_prompt_file_path, logger)

    if not prompt_template_with_attrs_str and not prompt_template_no_attrs_str:
        logger.error("Neither 'prompt_template_with_attributes' nor 'prompt_template_without_attributes' could be loaded. Skipping enhanced question generation.")
        return [], 0, 0, starting_question_id, {}
    if not system_prompt_content_str:
        logger.error(f"System prompt could not be loaded from '{system_prompt_file_path}'. Skipping enhanced question generation.")
        return [], 0, 0, starting_question_id, {}

    generator = SingleChoiceQuestionGenerator(
        api_key=config_enhanced.get('api_key'),
        base_url=config_enhanced.get('base_url'),
        model_name=config_enhanced.get('model', "gpt-4o"),
        logger=logger,
        system_prompt_content=system_prompt_content_str
    )
    if not generator.client:
        logger.error("OpenAI client for enhanced Qs not initialized. Skipping enhanced question generation.")
        return [], 0, 0, starting_question_id, {}

    max_api_call_retries = config_enhanced.get('max_retries', 3)
    retry_delay = config_enhanced.get('retry_delay_seconds', 2)
    max_rerequest_attempts_for_missing = config_enhanced.get('max_rerequest_attempts', 2)

    for item in tqdm(eval_items, desc="Generating Enhanced (AttriBait) Questions", unit="item", disable=logger.level > logging.INFO):
        image_id = item.get("image")
        if not image_id:
            logger.warning(f"Item missing 'image' field, skipping enhanced Qs: {str(item)[:100]}")
            continue

        all_hallucinated_objects_for_item = item.get("hallucinated_objects", [])
        attriBait_map = item.get("sg_attributes", {})
        
        # Note: Since scene graph models have been removed, sg_attributes will always be empty
        # This means all enhanced questions will be generated using the "no attributes" template

        if not isinstance(all_hallucinated_objects_for_item, list):
            logger.warning(f"Item {image_id} 'hallucinated_objects' is not a list for enhanced Qs, skipping. Value: {all_hallucinated_objects_for_item}")
            all_hallucinated_objects_for_item = []
        if not isinstance(attriBait_map, dict):
            logger.warning(f"Item {image_id} 'sg_attributes' is not a dict for enhanced Qs, skipping. Value: {attriBait_map}")
            attriBait_map = {}
        
        item_metadata = {
            "model": item.get("model", "unknown"), "epochs": item.get("epochs", None),
            "prompt_sets": item.get("prompt_sets", [])
        }

        payload_with_attrs = []
        payload_no_attrs_names = []
        valid_hallucinated_object_names_for_item = set()

        for h_obj_name in all_hallucinated_objects_for_item:
            if not h_obj_name or not isinstance(h_obj_name, str):
                logger.warning(f"Item {image_id} has invalid hallucinated object name for enhanced Qs, skipping. Value: {h_obj_name}")
                continue
            valid_hallucinated_object_names_for_item.add(h_obj_name)
            attrs = attriBait_map.get(h_obj_name, [])
            if attrs and isinstance(attrs, list) and len(attrs) > 0 and prompt_template_with_attrs_str:
                payload_with_attrs.append({"name": h_obj_name, "attributes": attrs})
            elif prompt_template_no_attrs_str:
                payload_no_attrs_names.append(h_obj_name)
            else:
                logger.debug(f"Img {image_id}: Object '{h_obj_name}' has no attributes and no_attrs template missing, or with_attrs template missing. Cannot generate enhanced Q.")

        if not payload_with_attrs and not payload_no_attrs_names:
            logger.debug(f"Item {image_id} has no valid hallucinated objects for enhanced Q generation based on available templates. Skipping API calls for this item.")
            continue
        
        records_processed_for_enhanced += 1
        questions_generated_this_item_cumulative = []
        
        # --- Process objects WITH attributes ---
        if payload_with_attrs and prompt_template_with_attrs_str:
            current_payload_with_attrs = list(payload_with_attrs)
            processed_entities_with_attrs = set()
            
            for rerequest_attempt in range(max_rerequest_attempts_for_missing + 1):
                if not current_payload_with_attrs: break

                target_object_names_this_call = [obj['name'] for obj in current_payload_with_attrs]
                logger.info(f"Img {image_id}: API Call (Attempt {rerequest_attempt+1}) for AttriBait WITH attributes. Target objects: {target_object_names_this_call}")

                info_lines = [f"Object: {obj['name']}, Suspected attributes: {json.dumps(obj['attributes'])}" for obj in current_payload_with_attrs]
                info_str = "\n".join(info_lines)
                prepared_prompt = prompt_template_with_attrs_str.replace("{objects_and_their_attributes}", info_str)
                
                tokens_before_call = generator.get_total_tokens_used()
                llm_qs_batch = generator.generate_questions(prepared_prompt, max_retries=max_api_call_retries, retry_delay_seconds=retry_delay)
                total_api_tokens_for_enhanced += (generator.get_total_tokens_used() - tokens_before_call)

                # --- New: Logging logic ---
                if llm_qs_batch:
                    entity_counts = Counter((q.get("entity") or q.get("object")) for q in llm_qs_batch if (q.get("entity") or q.get("object")))
                    for entity, count in entity_counts.items():
                        if count > 1:
                            log_key = f"Image '{image_id}' - Entity '{entity}'"
                            multi_question_entities_log[log_key] = count
                            logger.warning(f"Img {image_id}: Entity '{entity}' generated {count} questions in a single API call.")
                # --- End of logging logic ---

                newly_generated_entities_this_batch = set()
                for q_data in llm_qs_batch:
                    if not isinstance(q_data, dict) or not all(k in q_data for k in ["question", "options"]):
                        logger.warning(f"Img {image_id} (With Attrs, Attempt {rerequest_attempt+1}): API returned incomplete Q structure, skipping: {str(q_data)[:150]}"); continue
                    
                    # Compatible with both "entity" and "object" field names
                    entity_from_llm = q_data.get("entity") or q_data.get("object")
                    if not entity_from_llm:
                        logger.warning(f"Img {image_id} (With Attrs, Attempt {rerequest_attempt+1}): API returned Q without 'entity' or 'object' field, skipping: {str(q_data)[:150]}"); continue
                    
                    # Standardize field name to "entity"
                    if "object" in q_data and "entity" not in q_data:
                        q_data["entity"] = q_data.pop("object")
                    if entity_from_llm not in target_object_names_this_call:
                        logger.warning(f"Img {image_id} (With Attrs, Attempt {rerequest_attempt+1}): LLM returned entity '{entity_from_llm}' not in current request batch {target_object_names_this_call}. Skipping."); continue
                    
                    questions_generated_this_item_cumulative.append(q_data)
                    newly_generated_entities_this_batch.add(entity_from_llm)
                
                processed_entities_with_attrs.update(newly_generated_entities_this_batch)
                current_payload_with_attrs = [obj for obj in current_payload_with_attrs if obj['name'] not in processed_entities_with_attrs]

                if not current_payload_with_attrs:
                    logger.info(f"Img {image_id} (With Attrs): All targeted objects processed after attempt {rerequest_attempt+1}.")
                    break
                elif rerequest_attempt < max_rerequest_attempts_for_missing:
                    logger.warning(f"Img {image_id} (With Attrs): After attempt {rerequest_attempt+1}, {len(current_payload_with_attrs)} objects still missing questions. Retrying for: {[obj['name'] for obj in current_payload_with_attrs]}")
                elif rerequest_attempt == max_rerequest_attempts_for_missing:
                    logger.error(f"Img {image_id} (With Attrs): Max re-request attempts reached. {len(current_payload_with_attrs)} objects still missing AttriBait questions: {[obj['name'] for obj in current_payload_with_attrs]}")

        # --- Process objects WITHOUT attributes ---
        if payload_no_attrs_names and prompt_template_no_attrs_str:
            current_payload_no_attrs = list(payload_no_attrs_names)
            processed_entities_no_attrs = set()

            for rerequest_attempt in range(max_rerequest_attempts_for_missing + 1):
                if not current_payload_no_attrs: break

                target_object_names_this_call = list(current_payload_no_attrs)
                logger.info(f"Img {image_id}: API Call (Attempt {rerequest_attempt+1}) for AttriBait NO attributes. Target objects: {target_object_names_this_call}")
                
                objects_json_array = json.dumps(target_object_names_this_call)
                prepared_prompt = prompt_template_no_attrs_str.replace("{hallucinated_objects}", objects_json_array)

                tokens_before_call = generator.get_total_tokens_used()
                llm_qs_batch = generator.generate_questions(prepared_prompt, max_retries=max_api_call_retries, retry_delay_seconds=retry_delay)
                total_api_tokens_for_enhanced += (generator.get_total_tokens_used() - tokens_before_call)
                
                # --- New: Logging logic ---
                if llm_qs_batch:
                    entity_counts = Counter((q.get("entity") or q.get("object")) for q in llm_qs_batch if (q.get("entity") or q.get("object")))
                    for entity, count in entity_counts.items():
                        if count > 1:
                            log_key = f"Image '{image_id}' - Entity '{entity}'"
                            multi_question_entities_log[log_key] = count
                            logger.warning(f"Img {image_id}: Entity '{entity}' generated {count} questions in a single API call.")
                # --- End of logging logic ---

                newly_generated_entities_this_batch = set()
                for q_data in llm_qs_batch:
                    if not isinstance(q_data, dict) or not all(k in q_data for k in ["question", "options"]):
                        logger.warning(f"Img {image_id} (No Attrs, Attempt {rerequest_attempt+1}): API returned incomplete Q structure, skipping: {str(q_data)[:150]}"); continue
                    
                    # Compatible with both "entity" and "object" field names
                    entity_from_llm = q_data.get("entity") or q_data.get("object")
                    if not entity_from_llm:
                        logger.warning(f"Img {image_id} (No Attrs, Attempt {rerequest_attempt+1}): API returned Q without 'entity' or 'object' field, skipping: {str(q_data)[:150]}"); continue
                    
                    # Standardize field name to "entity"
                    if "object" in q_data and "entity" not in q_data:
                        q_data["entity"] = q_data.pop("object")
                    if entity_from_llm not in target_object_names_this_call:
                        logger.warning(f"Img {image_id} (No Attrs, Attempt {rerequest_attempt+1}): LLM returned entity '{entity_from_llm}' not in current request batch {target_object_names_this_call}. Skipping."); continue
                    
                    questions_generated_this_item_cumulative.append(q_data)
                    newly_generated_entities_this_batch.add(entity_from_llm)

                processed_entities_no_attrs.update(newly_generated_entities_this_batch)
                current_payload_no_attrs = [name for name in current_payload_no_attrs if name not in processed_entities_no_attrs]
                
                if not current_payload_no_attrs:
                    logger.info(f"Img {image_id} (No Attrs): All targeted objects processed after attempt {rerequest_attempt+1}.")
                    break
                elif rerequest_attempt < max_rerequest_attempts_for_missing:
                    logger.warning(f"Img {image_id} (No Attrs): After attempt {rerequest_attempt+1}, {len(current_payload_no_attrs)} objects still missing questions. Retrying for: {current_payload_no_attrs}")
                elif rerequest_attempt == max_rerequest_attempts_for_missing:
                    logger.error(f"Img {image_id} (No Attrs): Max re-request attempts reached. {len(current_payload_no_attrs)} objects still missing AttriBait questions: {current_payload_no_attrs}")

        # --- Consolidate generated questions for the current item ---
        for q_data in questions_generated_this_item_cumulative:
            entity_from_llm = q_data["entity"]
            if entity_from_llm not in valid_hallucinated_object_names_for_item:
                logger.warning(f"Img {image_id}: LLM returned entity '{entity_from_llm}' which was not in the item's original valid hallucinated list ({valid_hallucinated_object_names_for_item}). Skipping question.")
                continue
            
            options = q_data.get("options", {})
            if not isinstance(options, dict) or not all(k in options for k in ["A", "B"]):
                logger.warning(f"Img {image_id}: API options for entity '{entity_from_llm}' format incorrect or missing A/B, skipping. Opts: {options}"); continue
            
            # 1. Add C option as "Others"
            options["C"] = "Others"
            
            # 2. Add D option as the correct answer
            options["D"] = f"There is no {entity_from_llm} in the image"
            
            all_enhanced_questions.append({
                "question_id": question_id_counter, "image": image_id,
                "text": q_data["question"], 
                "options": options,       # Use new options containing A,B,C,D
                "label": "D",             # New correct label is D
                **item_metadata, "entity": entity_from_llm,
                "entity_type": "hallucinated", "question_type": "enhanced"
            })
            question_id_counter += 1
            
    # Modify function return value to include multi_question_entities_log
    return all_enhanced_questions, records_processed_for_enhanced, total_api_tokens_for_enhanced, question_id_counter, multi_question_entities_log

# --- Main Data Processing Orchestrator ---
def process_data_from_config(config, cli_append_mode, input_path, output_path, logger):
    """
    Main orchestrator that loads data, calls question generation functions,
    and saves the final dataset with statistics.
    """
    if not input_path or not output_path:
        logger.critical("Input or output file path not specified. Exiting.")
        sys.exit(1)
    logger.info(f"Loading verified entities from: {input_path}")
    eval_items = _load_json_data_helper(input_path, "Verified Entities Data", logger)
    if eval_items is None: 
        logger.critical(f"Failed to load or parse input data from {input_path}. Exiting.")
        sys.exit(1) 

    controls = config.get('generation_controls', {})
    enhanced_cfg = config.get('enhanced_question_generation', {})
    append_to_output = cli_append_mode
    newly_gen_basic_qs, newly_gen_enhanced_qs = [], []
    
    # Add a field in run_stats for recording
    run_stats = {
        "input_records_processed": len(eval_items),
        "basic_qs_generated_this_run": 0,
        "enhanced_qs_generated_this_run": 0,
        "enhanced_records_processed_this_run": 0, 
        "api_tokens_used_this_run": 0,
        "multi_question_entity_warnings": {}
    }
    
    existing_basic_qs, existing_enhanced_qs = [], []
    next_q_id = 1

    if append_to_output and os.path.exists(output_path):
        logger.info(f"Append mode ON. Loading existing data from: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f: existing_data = json.load(f)
            existing_basic_qs = existing_data.get("basic_questions", [])
            existing_enhanced_qs = existing_data.get("enhanced_questions", [])
            if not isinstance(existing_basic_qs, list): existing_basic_qs = []
            if not isinstance(existing_enhanced_qs, list): existing_enhanced_qs = []
            all_q_ids = [q['question_id'] for q_list in [existing_basic_qs, existing_enhanced_qs] for q in q_list if isinstance(q.get('question_id'), int)]
            if all_q_ids: next_q_id = max(all_q_ids) + 1
            logger.info(f"Found {len(existing_basic_qs)} basic and {len(existing_enhanced_qs)} enhanced Qs. Next question_id: {next_q_id}.")
        except Exception as e:
            logger.warning(f"Could not load/parse existing output '{output_path}' for append: {e}. Will overwrite.")
            append_to_output, existing_basic_qs, existing_enhanced_qs, next_q_id = False, [], [], 1
    else:
        logger.info("Output file does not exist or append mode is OFF. Starting fresh.")

    if controls.get('generate_basic_questions', False):
        inc_h, inc_i = controls.get('include_basic_hallucinated', True), controls.get('include_basic_ignored', True)
        logger.info(f"Generating Basic Yes/No Qs (Hallucinated: {inc_h}, GroundTruth: {inc_i})...")
        b_qs, next_q_id = generate_yes_no_questions(eval_items, next_q_id, inc_h, inc_i, logger)
        newly_gen_basic_qs.extend(b_qs)
        run_stats["basic_qs_generated_this_run"] = len(b_qs)

    total_tokens_this_run = 0

    if controls.get('generate_enhanced_questions', False):
        logger.info("Generating Enhanced (AttriBait) questions...")
        api_key_cfg, api_key_env = enhanced_cfg.get('api_key'), os.environ.get("OPENAI_API_KEY")
        prompt_paths_ok = enhanced_cfg.get('prompt_template_with_attributes') or enhanced_cfg.get('prompt_template_without_attributes')
        system_prompt_path_ok = enhanced_cfg.get('system_prompt_file') 

        if not api_key_cfg and not api_key_env:
            logger.warning("API Key for enhanced Qs not in config or OPENAI_API_KEY env. Skipping.")
        elif not prompt_paths_ok:
            logger.warning("Core prompt templates for enhanced Qs not specified in config. Skipping.")
        elif not system_prompt_path_ok: 
            logger.warning("Path to 'system_prompt_file' for enhanced Qs not specified in config. Skipping.")
        else:
            current_enhanced_cfg = enhanced_cfg.copy()
            if not api_key_cfg and api_key_env:
                current_enhanced_cfg['api_key'] = api_key_env
                logger.info("Using OPENAI_API_KEY from environment for enhanced Qs.")
            elif api_key_cfg:
                logger.info("Using API key from config file for enhanced Qs.")
            
            if 'max_rerequest_attempts' not in current_enhanced_cfg:
                current_enhanced_cfg['max_rerequest_attempts'] = 2
                logger.info(f"Using default max_rerequest_attempts: {current_enhanced_cfg['max_rerequest_attempts']}")

            # Modify this function call to receive new return value
            e_qs, e_rec_proc, e_tokens_enhanced, next_q_id, multi_q_log = generate_enhanced_questions(eval_items, current_enhanced_cfg, next_q_id, logger)
            
            newly_gen_enhanced_qs.extend(e_qs)
            run_stats["enhanced_qs_generated_this_run"] = len(e_qs)
            run_stats["enhanced_records_processed_this_run"] = e_rec_proc
            run_stats["multi_question_entity_warnings"] = multi_q_log # Store log in statistics
            total_tokens_this_run += e_tokens_enhanced
    
    run_stats["api_tokens_used_this_run"] = total_tokens_this_run

    if newly_gen_enhanced_qs:
        logger.info("Validating generated enhanced questions...")
        newly_gen_enhanced_qs = validate_generated_questions(newly_gen_enhanced_qs, logger)
        run_stats["enhanced_qs_generated_this_run"] = len(newly_gen_enhanced_qs) 
            
    final_basic_qs, final_enhanced_qs = existing_basic_qs + newly_gen_basic_qs, existing_enhanced_qs + newly_gen_enhanced_qs
    output_data_structure = {
        "statistics_this_run": run_stats,
        "statistics_total": {
            "total_basic_questions": len(final_basic_qs), "total_enhanced_questions": len(final_enhanced_qs),
            "total_questions": len(final_basic_qs) + len(final_enhanced_qs),
            "hallucinated_basic_questions": sum(1 for q in final_basic_qs if q.get("entity_type") == "hallucinated"),
            "ground_truth_basic_questions": sum(1 for q in final_basic_qs if q.get("entity_type") == "ground_truth"),
        },
        "basic_questions": final_basic_qs, "enhanced_questions": final_enhanced_qs,
    }
    _save_json_data_helper(output_data_structure, output_path, "TIDE Dataset", logger)
    return run_stats

# --- Main CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Generate TIDE - LVLM Hallucination Detection Question Dataset")
    parser.add_argument("--config", type=str, default="config_generate_tide.yaml", help="Path to YAML config")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input (overrides YAML)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output (overrides YAML)")
    parser.add_argument("--append", action="store_true", help="Append to output (overrides YAML file setting if this flag is present)")
    # New CLI arg to override append from config explicitly to False
    parser.add_argument("--no-append", action="store_false", dest="append_cli_val", help="Explicitly disable append mode (overrides YAML and --append if both config and --append are true)")
    parser.set_defaults(append_cli_val=None)


    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"FATAL: Config file not found: '{args.config}'"); sys.exit(1)
    try:
        with open(args.config, 'r', encoding='utf-8') as f: config_all = yaml.safe_load(f) # Load entire config
        if not config_all: print(f"FATAL: Config '{args.config}' is empty/invalid."); sys.exit(1)
        config = config_all.get('03_dataset_construct') # Get specific section
        if config is None:
            print(f"FATAL: '03_dataset_construct' not found in config '{args.config}'."); sys.exit(1)
    except Exception as e:
        print(f"FATAL: Error parsing YAML '{args.config}': {e}"); sys.exit(1)
    
    log_cfg = config.get('logging', {})
    log_level_str = log_cfg.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    script_logger = logging.getLogger("TIDE_GeneratorScript")
    script_logger.setLevel(log_level)
    if not script_logger.handlers: 
        ch = logging.StreamHandler(sys.stdout) # Output to stdout
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
        script_logger.addHandler(ch)
    script_logger.propagate = False # Avoid duplicate logs if root logger is configured
    script_logger.info(f"Loaded config from: {args.config}. Log level: {log_level_str}")

    io_conf = config.get('io', {})
    input_path = args.input_file if args.input_file is not None else io_conf.get('input_verified_entities_file')
    output_path = args.output_file if args.output_file is not None else io_conf.get('output_tide_dataset_file')

    if not input_path: script_logger.critical("Input path not specified. Exiting."); sys.exit(1)
    if not output_path: script_logger.critical("Output path not specified. Exiting."); sys.exit(1)
    script_logger.info(f"Input: {input_path}, Output: {output_path}")

    # Determine append mode: CLI args have highest precedence
    append_mode_yaml = io_conf.get('append_to_output', False)
    append_mode = append_mode_yaml
    if args.append_cli_val is not None: # --no-append was used
        append_mode = args.append_cli_val
    elif args.append: # --append was used
        append_mode = True
        
    script_logger.info(f"Append mode: {append_mode}.")
    
    run_summary_stats = process_data_from_config(config, append_mode, input_path, output_path, script_logger)

    if run_summary_stats:
        script_logger.info("\n--- Summary of This Run ---")
        for key, value in run_summary_stats.items():
            script_logger.info(f"{key.replace('_', ' ').capitalize()}: {value}")
        script_logger.info("--- Generation Process Complete ---")
    else:
        script_logger.error("Process may have failed or generated no data. Check logs.")

if __name__ == "__main__":
    main()