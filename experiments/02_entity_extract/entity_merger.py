import json
import os
import argparse
import re
import logging
import copy
import inflect
import nltk

try:
    # 检查 WordNet 是否可用。如果找不到，nltk.data.find 会引发 LookupError。
    nltk.data.find('corpora/wordnet')
except LookupError:
    # logger 实例可能尚未在此处完全配置，但 nltk.download 可以独立工作
    print("NLTK 'wordnet' resource not found. Downloading...") # 使用 print 以防 logger 未初始化
    nltk.download('wordnet', quiet=True)

try:
    # 检查 OMW (Open Multilingual WordNet) 1.4 是否可用。
    # OMW 依赖于 WordNet，因此最好在 WordNet 之后检查/下载它。
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("NLTK 'omw-1.4' resource not found. Downloading...") # 使用 print
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

p = inflect.engine()


# 配置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # 添加控制台处理器

def save_output_json(data_to_save, output_filepath, logger_instance):
    """保存JSON文件，只对caption_objects和sg_attributes中的列表进行紧凑化显示"""
    items_to_format = copy.deepcopy(data_to_save)
    placeholder_map = {}
    placeholder_idx_obj = [0]

    def convert_lists_to_placeholders_recursive(current_data, p_map, p_idx_obj, parent_key=None):
        if isinstance(current_data, dict):
            keys_in_order = list(current_data.keys())
            for key in keys_in_order:
                value = current_data[key]
                if isinstance(value, list):
                    # 对 caption_objects 和 sg_attributes 下的所有列表应用紧凑格式
                    if key == 'caption_objects' or parent_key == 'sg_attributes':
                        placeholder_key_string = f"__COMPACT_LIST_PLACEHOLDER_{p_idx_obj[0]}__"
                        try:
                            # 对于 sg_attributes 的值（也是列表），确保其元素也是简单类型以便紧凑化
                            # 如果列表内元素复杂（如字典），此紧凑化可能不理想，但按原逻辑处理
                            p_map[placeholder_key_string] = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
                            current_data[key] = placeholder_key_string
                            p_idx_obj[0] += 1
                        except TypeError as te:
                            logger_instance.warning(f"Could not JSON dump list for key '{key}' (parent: {parent_key}) due to TypeError: {te}. List: {str(value)[:100]}...")
                elif isinstance(value, dict):
                    convert_lists_to_placeholders_recursive(value, p_map, p_idx_obj, key) # 传递当前key作为下一层的parent_key

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
    """加载JSON文件并返回其内容。"""
    if not os.path.exists(file_path):
        logger.error(f"错误：文件未找到于 {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载 {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"错误：解析JSON文件 {file_path} 失败: {e}")
        return None
    except Exception as e:
        logger.error(f"加载 {file_path} 时发生未知错误: {e}")
        return None

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    用于自然排序的键。例如："image1.jpg", "image2.jpg", "image10.jpg"。
    如果传入的是字典，则从 'image' 字段获取字符串进行排序。
    """
    if isinstance(s, dict):
        s = s.get('image', '')
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def is_valid_word(word):
    """检查是否为 WordNet 中存在的有效英文词"""
    # 对于多词短语，WordNet 可能无法直接找到 synsets。
    # 这个函数主要用于单数化后的词的校验。
    # 如果传入的是短语，可以考虑只校验短语中的主要名词，或者直接返回 True。
    # 为简单起见，如果包含空格，暂时认为其“有效性”由上下文保证，或不强制校验。
    if ' ' in word:
        parts = word.split()
        # 简单校验：如果最后一个词是有效的，则认为短语可能有效
        # 更复杂的校验可能需要词性标注
        return wn.synsets(parts[-1]) != [] if parts else False
    return wn.synsets(word) != []

def singularize_object(obj):
    """
    更稳健的单数化逻辑：
    - 使用 inflect 判断是否为复数；
    - 可逆性校验；
    - 验证是否是英文有效词；
    - 对于多词短语，尝试单数化最后一个词。
    """
    if not obj or not isinstance(obj, str):
        return obj

    obj_stripped = obj.strip()
    if not obj_stripped:
        return obj_stripped

    try:
        # 处理多词短语，如 "apple pies" -> "apple pie"
        if ' ' in obj_stripped:
            words = obj_stripped.split()
            last_word = words[-1]
            singular_last_word = p.singular_noun(last_word)
            
            if singular_last_word: # inflect 成功单数化了最后一个词
                # 可逆性校验 (只对最后一个词) 和有效性校验
                if p.plural(singular_last_word) == last_word and is_valid_word(singular_last_word):
                    words[-1] = singular_last_word
                    return " ".join(words)
                else: # 单数化失败或校验失败，返回原短语
                    return obj_stripped
            else: # inflect 认为最后一个词已经是单数或无法处理
                return obj_stripped # 返回原短语
        else: # 单个词
            singular = p.singular_noun(obj_stripped)
            if singular: # inflect 成功单数化
                # 可逆性校验和有效性校验
                if p.plural(singular) == obj_stripped and is_valid_word(singular):
                    return singular
                else: # 校验失败
                    return obj_stripped
            else: # inflect 认为已经是单数或无法处理
                return obj_stripped
    except Exception as e:
        logger.warning(f"单数化 '{obj_stripped}' 时出错: {e}，使用原始形式")
        return obj_stripped

def merge_data(c2e_extracted_data, e2e_data):
    """
    合并来自 c2e_extracted (extract.py 对 c2e.json 的处理结果)
    和 e2e_data (直接来自 batch_llava 的 e2e.json) 的数据。
    其中一个输入数据源可以为 None。
    """
    _c2e_data_list = []
    if c2e_extracted_data is not None:
        if isinstance(c2e_extracted_data, list):
            _c2e_data_list = c2e_extracted_data
        else:
            logger.warning("c2e_extracted_data 不是预期的列表格式，将被视为空列表进行处理。")

    _e2e_data_list = []
    if e2e_data is not None:
        if isinstance(e2e_data, list):
            _e2e_data_list = e2e_data
        else:
            logger.warning("e2e_data 不是预期的列表格式，将被视为空列表进行处理。")

    if not _c2e_data_list and not _e2e_data_list:
        logger.warning("两个输入数据源均为空或格式不正确，或未包含有效条目。无法执行合并。")
        return []

    c2e_map = {item['image']: item for item in _c2e_data_list if isinstance(item, dict) and 'image' in item}
    e2e_map = {item['image']: item for item in _e2e_data_list if isinstance(item, dict) and 'image' in item}

    all_image_filenames = sorted(list(set(c2e_map.keys()) | set(e2e_map.keys())), key=natural_sort_key)

    if not all_image_filenames:
        logger.warning("未找到可合并的图片文件名。输入数据可能有效但无共享图片ID或无图片条目。")
        return []

    logger.info(f"找到 {len(all_image_filenames)} 个唯一图片进行处理。")

    final_results_list = []

    for image_filename in all_image_filenames:
        merged_item = {'image': image_filename}
        c2e_item = c2e_map.get(image_filename) 
        e2e_item = e2e_map.get(image_filename) 

        # 1. Captions (保持不变)
        all_captions = []
        if c2e_item and isinstance(c2e_item.get('captions'), list):
            all_captions.extend(c2e_item['captions'])
        if e2e_item and isinstance(e2e_item.get('captions'), list):
            all_captions.extend(e2e_item['captions'])
        merged_item['captions'] = all_captions

        # 2. Model (保持不变)
        model_name = None
        c2e_model = c2e_item.get('model') if c2e_item else None
        e2e_model = e2e_item.get('model') if e2e_item else None
        if c2e_model and e2e_model and c2e_model != e2e_model:
            logger.warning(f"图片 '{image_filename}' 的模型名称不一致: "
                           f"c2e为 '{c2e_model}', e2e为 '{e2e_model}'。将优先使用c2e的模型名称。")
        model_name = c2e_model if c2e_model else e2e_model
        merged_item['model'] = model_name

        # 3. Epochs (保持不变)
        epochs_val = None
        c2e_epochs_val = c2e_item.get('epochs') if c2e_item else None
        e2e_epochs_val = e2e_item.get('epochs') if e2e_item else None
        if c2e_epochs_val is not None:
            epochs_val = c2e_epochs_val
        elif e2e_epochs_val is not None:
            epochs_val = e2e_epochs_val
        if c2e_item and e2e_item and c2e_epochs_val is not None and e2e_epochs_val is not None and c2e_epochs_val != e2e_epochs_val:
                 logger.warning(f"图片 '{image_filename}' 的轮数 (epochs) 不一致: "
                               f"c2e为 {c2e_epochs_val}, e2e为 {e2e_epochs_val}。 "
                               f"当前使用的值为: {epochs_val} (优先来自c2e的值)。")
        merged_item['epochs'] = epochs_val

        # 4. Prompt Sets (保持不变)
        prompt_names = set()
        if c2e_item and isinstance(c2e_item.get('prompt_sets'), list):
            prompt_names.update(name for name in c2e_item['prompt_sets'] if isinstance(name, str))
        if e2e_item and isinstance(e2e_item.get('prompt_sets'), list):
            prompt_names.update(name for name in e2e_item['prompt_sets'] if isinstance(name, str))
        merged_item['prompt_sets'] = sorted(list(prompt_names))

        # 5. Caption Objects (合并去重，单数化，并应用空格短语优先规则)
        all_original_entities = []
        if c2e_item:
            if isinstance(c2e_item.get('llm_objects'), list):
                all_original_entities.extend(obj.strip() for obj in c2e_item['llm_objects'] if isinstance(obj, str) and obj.strip())
            # sg_objects 已移除，因为不再使用场景图模型
        if e2e_item:
            if isinstance(e2e_item.get('caption_objects'), list):
                all_original_entities.extend(obj.strip() for obj in e2e_item['caption_objects'] if isinstance(obj, str) and obj.strip())
        
        singular_entities_initial = set()
        for entity in all_original_entities:
            singular_form = singularize_object(entity)
            singular_entities_initial.add(singular_form)
        
        # 应用空格短语优先规则
        # no_space_to_spaced_preference_map 用于记录哪个无空格版本被其有空格版本替代了，供 sg_attributes 使用
        no_space_to_spaced_preference_map = {}
        caption_objects_final_candidates = set(singular_entities_initial) # 初始候选集合

        # 为了安全地修改集合，我们遍历其副本或一个列表
        # 找出所有含空格的短语及其无空格版本
        phrases_with_spaces = {s for s in singular_entities_initial if ' ' in s}

        for spaced_phrase in phrases_with_spaces:
            no_space_version = spaced_phrase.replace(" ", "")
            if no_space_version in singular_entities_initial: # 确保无空格版本也确实存在于初始集合中
                # 如果有空格版本和无空格版本都存在，优先保留有空格版本
                if no_space_version in caption_objects_final_candidates:
                    caption_objects_final_candidates.remove(no_space_version)
                    logger.info(f"Caption Objects for '{image_filename}': '{no_space_version}' removed in favor of '{spaced_phrase}'.")
                # 记录这种替换关系，确保有空格版本一定在最终结果中
                caption_objects_final_candidates.add(spaced_phrase) # 确保有空格版本存在
                no_space_to_spaced_preference_map[no_space_version] = spaced_phrase
        
        merged_item['caption_objects'] = sorted(list(caption_objects_final_candidates))

        # 6. SG Attributes - 由于移除了场景图提取器，设置为空字典
        final_sg_attributes = {}
        # 注意：sg_attributes 功能已被移除，因为不再使用场景图模型
        merged_item['sg_attributes'] = final_sg_attributes
        
        final_results_list.append(merged_item)

    return final_results_list

def main():
    parser = argparse.ArgumentParser(description="合并来自 c2e (经 extract.py 处理) 和 e2e 流程的实体提取结果。")
    parser.add_argument("--c2e", required=False, help="来自 extract.py 的JSON文件路径 (处理c2e.json后的结果)。")
    parser.add_argument("--e2e", required=False, help="来自 batch_llava 的 e2e.json 文件路径。")
    parser.add_argument("--output_file", required=True, help="合并后的输出JSON文件路径。")
    
    args = parser.parse_args()

    if not args.c2e and not args.e2e:
        parser.error("错误：必须至少指定 --c2e 或 --e2e 中的一个。")

    logger.info("开始处理...")
    
    c2e_data = None
    if args.c2e:
        logger.info(f"C2E提取后文件: {args.c2e}")
        c2e_data = load_json_file(args.c2e)
        if c2e_data is None:
            # load_json_file 内部会记录错误, 这里可以简单退出
            logger.error(f"由于C2E文件 {args.c2e} 加载失败，处理中止。")
            return 
    else:
        logger.info("未提供 C2E提取后文件，将跳过。")

    e2e_data = None
    if args.e2e:
        logger.info(f"E2E文件: {args.e2e}")
        e2e_data = load_json_file(args.e2e)
        if e2e_data is None:
            logger.error(f"由于E2E文件 {args.e2e} 加载失败，处理中止。")
            return
    else:
        logger.info("未提供 E2E文件，将跳过。")
        
    logger.info(f"输出文件: {args.output_file}")

    merged_results = merge_data(c2e_data, e2e_data)

    if merged_results: 
        save_output_json(merged_results, args.output_file, logger)
        logger.info(f"处理完成。合并后的文件包含 {len(merged_results)} 个条目。")
    else:
        logger.warning("处理完成，但合并结果为空列表或未成功生成数据。未写入输出文件。")

if __name__ == "__main__":
    main()