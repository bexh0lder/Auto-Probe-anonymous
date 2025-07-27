#!/bin/bash
# pipeline_llava_1.6.sh - 主流程脚本，用于运行 LLaVA 1.6 模型从生成到评估的完整工作流
# (特点：所有步骤使用screen运行，保留完整日志)

# --- 📜 全局配置与初始化 ---
# 如果任何命令失败，立即退出
set -e
# 如果管道中的任何命令失败，则使整个管道失败
set -o pipefail

echo "🎉🎉🎉 开始执行 LLaVA 1.6 完整工作流程 (步骤01-04) 🎉🎉🎉"
START_TIME=$(date +%s)

# --- 路径定义 ---
BASE_PROJECT_DIR="/root/Auto-Probe/experiments"
# 【重要】确保这个配置文件是为 LLaVA 1.6 准备的
PIPELINE_CONFIG_FILE="${BASE_PROJECT_DIR}/pipeline_llava1.6.yaml" 

# --- Python 解释器路径定义 (用户必须核实这些路径) ---
# 假设 LLaVA 1.6 及其依赖都在 had 环境中
PYTHON_HAD_EXE="/root/miniconda3/envs/auto/bin/python"
# 步骤02中的某些脚本可能在不同环境中，根据实际情况修改
PYTHON_EVALUATE_EXE="/root/miniconda3/envs/auto/bin/python"

# --- 动态路径管理 ---
RUN_ID="mild_$(date +%Y%m%d_%H%M%S)"
# 【重要】修改输出路径以反映这是 LLaVA 1.6 的运行结果
RESULTS_ROOT_DIR="${BASE_PROJECT_DIR}/results/coco/llava_1.6/${RUN_ID}"
# RESULTS_ROOT_DIR="/root/Auto-Probe/experiments/results/coco/llava_1.6/llava_1.6_20250630_212614"
# --- 步骤间输入输出文件路径 ---
# 步骤01
STEP01_OUTPUT_DIR="${RESULTS_ROOT_DIR}/01_caption_generate"
STEP01_SCREEN_LOG_DIR="${STEP01_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP01_OUTPUT_DIR}" "${STEP01_SCREEN_LOG_DIR}"
STEP01_CAPTIONS_FILE="${STEP01_OUTPUT_DIR}/c2e.json"

# 步骤02 (这部分脚本调用通常与模型无关，因此保持不变)
STEP02_OUTPUT_DIR="${RESULTS_ROOT_DIR}/02_entity_extract"
STEP02_SCREEN_LOG_DIR="${STEP02_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP02_OUTPUT_DIR}" "${STEP02_SCREEN_LOG_DIR}"
FILENAME_BASE_FOR_STEP02="c2e"
STEP02_EXTRACTED_RAW_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_01_extracted_raw.json"
STEP02_NORMALIZED_MERGED_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_02_normalized_merged.json"
STEP02_LLM_CLEANED_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_03_llm_cleaned.json"
STEP02_VERIFIED_ENTITIES_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_04_owl_verified_final.json"
STEP02_CORRECTED_ENTITIES_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_05_consistency_corrected.json"

# 步骤03
STEP03_OUTPUT_DIR="${RESULTS_ROOT_DIR}/03_dataset_construct"
STEP03_SCREEN_LOG_DIR="${STEP03_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP03_OUTPUT_DIR}" "${STEP03_SCREEN_LOG_DIR}"
STEP03_INITIAL_TIDE_DATASET_FILE="${STEP03_OUTPUT_DIR}/tide.json"

# 步骤04
STEP04_OUTPUT_DIR="${RESULTS_ROOT_DIR}/04_model_evaluate"
STEP04_SCREEN_LOG_DIR="${STEP04_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP04_OUTPUT_DIR}" "${STEP04_SCREEN_LOG_DIR}"
STEP04_EVAL_OUTPUT_FILENAME="pipeline_llava_tide_eval_results.json"
RESULTS_CSV_PATH="${BASE_PROJECT_DIR}/results/coco/results.csv"

# 主图像目录
MAIN_IMAGE_DIR="/root/autodl-tmp/datasets/coco/val2014"

# --- Screen会话配置 ---
CONFIG_BASENAME=$(basename "$PIPELINE_CONFIG_FILE" .yaml)
TIMESTAMP_SUFFIX=$(date +%H%M%S)

# --- 检查核心文件与目录 ---
if [[ ! -f "$PIPELINE_CONFIG_FILE" ]]; then echo "🚨 严重错误: 主流程配置文件未找到: ${PIPELINE_CONFIG_FILE}"; exit 1; fi
echo "🛠️  使用主流程配置文件: ${PIPELINE_CONFIG_FILE}"
if [[ ! -x "$PYTHON_HAD_EXE" ]]; then echo "🚨 严重错误: Python (had) 解释器未找到或不可执行: ${PYTHON_HAD_EXE}"; exit 1; fi
if [[ ! -x "$PYTHON_EVALUATE_EXE" ]]; then echo "🚨 严重错误: Python (evaluate) 解释器未找到或不可执行: ${PYTHON_EVALUATE_EXE}"; exit 1; fi
echo "🐍 Python (had) 解释器: ${PYTHON_HAD_EXE}"
echo "🐍 Python (evaluate) 解释器: ${PYTHON_EVALUATE_EXE}"
if [[ ! -d "$MAIN_IMAGE_DIR" ]]; then echo "🚨 严重错误: 主图像目录未找到: ${MAIN_IMAGE_DIR}"; exit 1; fi
echo "🏞️  使用主图像目录: ${MAIN_IMAGE_DIR}"

# --- 流程正式开始 ---
cd "$BASE_PROJECT_DIR" || { echo "🚨 错误: 无法切换到项目基础目录 ${BASE_PROJECT_DIR}"; exit 1; }
echo "📂 当前工作目录: $(pwd)"
echo "📂 正在创建结果根目录: ${RESULTS_ROOT_DIR}"
mkdir -p "$RESULTS_ROOT_DIR"
echo "📂 复制配置文件到输出根目录..."
cp "$PIPELINE_CONFIG_FILE" "$RESULTS_ROOT_DIR"

# === 🚀 步骤 01: 图像描述生成 (LLaVA 1.6) ===
echo ""
echo "===== 1️⃣  开始步骤 01: 图像描述生成 (LLaVA 1.6) ====="
LLAVA_PYTHON_SCRIPT="${BASE_PROJECT_DIR}/01_caption_generate/batch_llava1.6.py"
SCREEN_SESSION_NAME_STEP01="llava_caption_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP01="${STEP01_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP01}.log"
echo "⏳ 正在运行 ${LLAVA_PYTHON_SCRIPT} (在 screen 会话 ${SCREEN_SESSION_NAME_STEP01} 中)..."
echo "    Screen日志: ${SCREEN_LOG_FILE_STEP01}"
screen -S "${SCREEN_SESSION_NAME_STEP01}" -L -Logfile "${SCREEN_LOG_FILE_STEP01}" -m \
    "${PYTHON_HAD_EXE}" "${LLAVA_PYTHON_SCRIPT}" --config "${PIPELINE_CONFIG_FILE}" --output_dir "${STEP01_OUTPUT_DIR}"

# 等待screen会话结束
echo "    等待 screen 会话 ${SCREEN_SESSION_NAME_STEP01} 结束..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP01}"; do
    sleep 10 # 每10秒检查一次
done
echo "    Screen会话 ${SCREEN_SESSION_NAME_STEP01} 已完成。"

if [[ ! -f "$STEP01_CAPTIONS_FILE" ]]; then
    echo "🚨 严重错误: 步骤 01 的输出文件未找到: ${STEP01_CAPTIONS_FILE}"
    echo "👉 请检查 Screen日志: ${SCREEN_LOG_FILE_STEP01}"
    exit 1
fi
echo "✅ 步骤 01 完成。描述文件位于: ${STEP01_CAPTIONS_FILE}"


# === 🚀 步骤 02: 实体提取与验证 ===
echo ""
echo "===== 2️⃣  开始步骤 02: 实体提取与验证 ====="
EXTRACTOR_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_extractor.py"
MERGER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_merger.py"
CLEANER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_cleaner.py"
VERIFIER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/owlv2_verifier.py"
CORRECTOR_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/consistency_corrector.py"

SCREEN_SESSION_NAME_STEP02="llava_entity_extract_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP02="${STEP02_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP02}.log"
echo "⏳ 在screen会话 ${SCREEN_SESSION_NAME_STEP02} 中运行步骤02的所有子步骤..."
echo "    Screen日志: ${SCREEN_LOG_FILE_STEP02}"

STEP02_COMMANDS="set -e; set -o pipefail; \
echo '开始步骤02的所有子任务 - '\$(date); \
echo '⏳ 2.1: 实体提取...'; \
'${PYTHON_EVALUATE_EXE}' '${EXTRACTOR_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP01_CAPTIONS_FILE}' --output_file '${STEP02_EXTRACTED_RAW_FILE}' || exit 1; \
echo '    ✅ 实体提取完成。'; \
echo '⏳ 2.2: 实体合并与规范化...'; \
'${PYTHON_EVALUATE_EXE}' '${MERGER_SCRIPT_STEP02}' --c2e '${STEP02_EXTRACTED_RAW_FILE}' --output_file '${STEP02_NORMALIZED_MERGED_FILE}' || exit 1; \
echo '    ✅ 实体合并完成。'; \
echo '⏳ 2.3: 基于LLM的实体清洗...'; \
'${PYTHON_EVALUATE_EXE}' '${CLEANER_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_NORMALIZED_MERGED_FILE}' --output_file '${STEP02_LLM_CLEANED_FILE}' || exit 1; \
echo '    ✅ 实体清洗完成。'; \
echo '⏳ 2.4: OWL-ViT 验证...'; \
'${PYTHON_EVALUATE_EXE}' '${VERIFIER_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_LLM_CLEANED_FILE}' --output_file '${STEP02_VERIFIED_ENTITIES_FILE}' --image_dir '${MAIN_IMAGE_DIR}' || exit 1; \
echo '    ✅ OWL验证完成。'; \
echo '⏳ 2.5: 检测一致性修正...'; \
'${PYTHON_EVALUATE_EXE}' '${CORRECTOR_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_VERIFIED_ENTITIES_FILE}' --output_file '${STEP02_CORRECTED_ENTITIES_FILE}' || exit 1; \
echo '    ✅ 检测一致性修正完成。'; \
echo '步骤02的所有子任务完成 - '\$(date)"

screen -S "${SCREEN_SESSION_NAME_STEP02}" -L -Logfile "${SCREEN_LOG_FILE_STEP02}" -m bash -c "${STEP02_COMMANDS}"

# 等待screen会话结束
echo "    等待 screen 会话 ${SCREEN_SESSION_NAME_STEP02} 结束..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP02}"; do
    sleep 10
done
echo "    Screen会话 ${SCREEN_SESSION_NAME_STEP02} 已完成。"

if [[ ! -f "$STEP02_CORRECTED_ENTITIES_FILE" ]]; then
    curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=Step2Error_LLaVA"
    echo "🚨 严重错误: 步骤 02 的最终输出文件未找到: ${STEP02_CORRECTED_ENTITIES_FILE}"
    echo "👉 请检查 Screen日志: ${SCREEN_LOG_FILE_STEP02}"
    exit 1
fi
echo "✅ 步骤 02 完成。已修正实体文件位于: ${STEP02_CORRECTED_ENTITIES_FILE}"


# === 🚀 步骤 03: TIDE 数据集生成 ===
echo ""
echo "===== 3️⃣  开始步骤 03: TIDE 数据集生成 (LLaVA 1.6) ====="
PY_SCRIPT_03_GENERATE="${BASE_PROJECT_DIR}/03_dataset_construct/generate_dataset.py"
if [[ ! -f "$PY_SCRIPT_03_GENERATE" ]]; then echo "🚨 错误: TIDE生成脚本未找到: ${PY_SCRIPT_03_GENERATE}"; exit 1; fi

SCREEN_SESSION_NAME_STEP03="llava_tide_construct_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP03="${STEP03_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP03}.log"
echo "⏳ 在screen会话 ${SCREEN_SESSION_NAME_STEP03} 中运行TIDE数据集生成..."
echo "    Screen日志: ${SCREEN_LOG_FILE_STEP03}"

STEP03_COMMANDS="set -e; set -o pipefail; \
echo '⏳ 3.1: TIDE数据集生成...'; \
'${PYTHON_HAD_EXE}' '${PY_SCRIPT_03_GENERATE}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_CORRECTED_ENTITIES_FILE}' --output_file '${STEP03_INITIAL_TIDE_DATASET_FILE}' --no-append || { echo '🚨 步骤 3.1 失败'; exit 1; }; \
echo '    ✅ 步骤 3.1 完成。'; \
echo '步骤03 (数据集生成) 全部完成 - '\$(date)"
screen -S "${SCREEN_SESSION_NAME_STEP03}" -L -Logfile "${SCREEN_LOG_FILE_STEP03}" -m \
    bash -c "${STEP03_COMMANDS}"
    
echo "    等待 screen 会话 ${SCREEN_SESSION_NAME_STEP03} 结束..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP03}"; do sleep 10; done
echo "    Screen会话 ${SCREEN_SESSION_NAME_STEP03} 已完成。"

if [[ ! -f "$STEP03_INITIAL_TIDE_DATASET_FILE" ]]; then 
    echo "🚨 严重错误: 步骤 03 的最终输出文件未找到: ${STEP03_INITIAL_TIDE_DATASET_FILE}"; 
    curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=Step3Error_LLaVA";
    exit 1; 
fi
echo "✅ 步骤 03 完成。TIDE数据集位于: ${STEP03_INITIAL_TIDE_DATASET_FILE}"


# === 🚀 步骤 04: 模型评估 (LLaVA 1.6) ===
echo ""
echo "===== 4️⃣  开始步骤 04: 模型评估 (LLaVA 1.6) ====="
PY_SCRIPT_04="${BASE_PROJECT_DIR}/04_model_evaluate/evaluate_llava1.6.py"
if [[ ! -f "$PY_SCRIPT_04" ]]; then echo "🚨 错误: 评估脚本 (LLaVA 1.6) 未找到: ${PY_SCRIPT_04}"; exit 1; fi

SCREEN_SESSION_NAME_STEP04="llava_evaluate_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP04="${STEP04_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP04}.log"
echo "⏳ 在screen会话 ${SCREEN_SESSION_NAME_STEP04} 中运行数据集评估..."
echo "    Screen日志: ${SCREEN_LOG_FILE_STEP04}"

CMD_EVALUATE="'${PYTHON_HAD_EXE}' '${PY_SCRIPT_04}' --config '${PIPELINE_CONFIG_FILE}' --tide_dataset_file '${STEP03_INITIAL_TIDE_DATASET_FILE}' --image_dir '${MAIN_IMAGE_DIR}' --output_dir '${STEP04_OUTPUT_DIR}' --output_filename '${STEP04_EVAL_OUTPUT_FILENAME}' --summary_csv '${RESULTS_CSV_PATH}'"

STEP04_COMMANDS="set -e; set -o pipefail; \
echo '⏳ 4.1: 开始评估数据集...'; \
${CMD_EVALUATE} || { echo '🚨 评估数据集失败'; exit 1; }; \
echo '    ✅ 数据集评估完成。'; \
echo '步骤04 (模型评估) 全部完成 - '\$(date)"
screen -S "${SCREEN_SESSION_NAME_STEP04}" -L -Logfile "${SCREEN_LOG_FILE_STEP04}" -m \
    bash -c "${STEP04_COMMANDS}"

echo "    等待 screen 会话 ${SCREEN_SESSION_NAME_STEP04} 结束..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP04}"; do
    sleep 10
done
echo "    Screen会话 ${SCREEN_SESSION_NAME_STEP04} 已完成。"
echo "✅ 步骤 04 完成。评估结果位于 ${STEP04_OUTPUT_DIR}。"

# --- 🏁 流程结束 ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✨✨✨ LLaVA 1.6 完整工作流程成功结束! ✨✨✨"
echo "⏱️  总执行时间: ${DURATION} 秒。"
echo "📂 所有相关的日志和输出文件位于 ${RESULTS_ROOT_DIR} 下。"
echo ""
echo "📋 Screen日志文件位置: "
echo "➡️  步骤01 (图像描述): ${SCREEN_LOG_FILE_STEP01}"
echo "➡️  步骤02 (实体提取): ${SCREEN_LOG_FILE_STEP02}"
echo "➡️  步骤03 (数据集生成): ${SCREEN_LOG_FILE_STEP03}"
echo "➡️  步骤04 (模型评估): ${SCREEN_LOG_FILE_STEP04}"
echo ""
echo "📁 关键输出文件位置："
echo "➡️  TIDE数据集在: ${STEP03_INITIAL_TIDE_DATASET_FILE}"
echo "➡️  数据集评估结果在: ${STEP04_OUTPUT_DIR}/${STEP04_EVAL_OUTPUT_FILENAME}"
echo "➡️  模型评估摘要已追加到: ${RESULTS_CSV_PATH}"

curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=ExperimentOver_LLaVA"

exit 0