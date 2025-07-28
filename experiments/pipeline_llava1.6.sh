#!/bin/bash
# LLaVA 1.6 complete pipeline script from generation to evaluation
# Features: All steps run in screen with full logging

# Global configuration and initialization
set -e  # Exit immediately if any command fails
set -o pipefail  # Pipeline fails if any command fails

echo "🎉 Starting LLaVA 1.6 complete workflow (steps 01-04) 🎉"
START_TIME=$(date +%s)

# Path definitions
BASE_PROJECT_DIR="/root/Auto-Probe/experiments"
PIPELINE_CONFIG_FILE="${BASE_PROJECT_DIR}/pipeline_llava1.6.yaml" 

# Python interpreter paths (users must verify these paths)
# Assumes LLaVA 1.6 and dependencies are in the auto environment
PYTHON_HAD_EXE="/root/miniconda3/envs/auto/bin/python"
PYTHON_EVALUATE_EXE="/root/miniconda3/envs/auto/bin/python"

# Dynamic path management
RUN_ID="mild_$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT_DIR="${BASE_PROJECT_DIR}/results/coco/llava_1.6/${RUN_ID}"

# Input/output file paths between steps
# Step 01
STEP01_OUTPUT_DIR="${RESULTS_ROOT_DIR}/01_caption_generate"
STEP01_SCREEN_LOG_DIR="${STEP01_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP01_OUTPUT_DIR}" "${STEP01_SCREEN_LOG_DIR}"
STEP01_CAPTIONS_FILE="${STEP01_OUTPUT_DIR}/c2e.json"

# Step 02 (usually model-independent, keeping unchanged)
STEP02_OUTPUT_DIR="${RESULTS_ROOT_DIR}/02_entity_extract"
STEP02_SCREEN_LOG_DIR="${STEP02_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP02_OUTPUT_DIR}" "${STEP02_SCREEN_LOG_DIR}"
FILENAME_BASE_FOR_STEP02="c2e"
STEP02_EXTRACTED_RAW_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_01_extracted_raw.json"
STEP02_NORMALIZED_MERGED_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_02_normalized_merged.json"
STEP02_LLM_CLEANED_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_03_llm_cleaned.json"
STEP02_VERIFIED_ENTITIES_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_04_owl_verified_final.json"
STEP02_CORRECTED_ENTITIES_FILE="${STEP02_OUTPUT_DIR}/${FILENAME_BASE_FOR_STEP02}_05_consistency_corrected.json"

# Step 03
STEP03_OUTPUT_DIR="${RESULTS_ROOT_DIR}/03_dataset_construct"
STEP03_SCREEN_LOG_DIR="${STEP03_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP03_OUTPUT_DIR}" "${STEP03_SCREEN_LOG_DIR}"
STEP03_INITIAL_TIDE_DATASET_FILE="${STEP03_OUTPUT_DIR}/tide.json"

# Step 04
STEP04_OUTPUT_DIR="${RESULTS_ROOT_DIR}/04_model_evaluate"
STEP04_SCREEN_LOG_DIR="${STEP04_OUTPUT_DIR}/screen_logs"
mkdir -p "${STEP04_OUTPUT_DIR}" "${STEP04_SCREEN_LOG_DIR}"
STEP04_EVAL_OUTPUT_FILENAME="pipeline_llava_tide_eval_results.json"
RESULTS_CSV_PATH="${BASE_PROJECT_DIR}/results/coco/results.csv"

# Main image directory
MAIN_IMAGE_DIR="/root/autodl-tmp/datasets/coco/val2014"

# Screen session configuration
CONFIG_BASENAME=$(basename "$PIPELINE_CONFIG_FILE" .yaml)
TIMESTAMP_SUFFIX=$(date +%H%M%S)

# Check core files and directories
if [[ ! -f "$PIPELINE_CONFIG_FILE" ]]; then echo "❌ Critical error: Pipeline config file not found: ${PIPELINE_CONFIG_FILE}"; exit 1; fi
echo "🛠️  Using pipeline config file: ${PIPELINE_CONFIG_FILE}"
if [[ ! -x "$PYTHON_HAD_EXE" ]]; then echo "❌ Critical error: Python (had) interpreter not found or not executable: ${PYTHON_HAD_EXE}"; exit 1; fi
if [[ ! -x "$PYTHON_EVALUATE_EXE" ]]; then echo "❌ Critical error: Python (evaluate) interpreter not found or not executable: ${PYTHON_EVALUATE_EXE}"; exit 1; fi
echo "🐍 Python (had) interpreter: ${PYTHON_HAD_EXE}"
echo "🐍 Python (evaluate) interpreter: ${PYTHON_EVALUATE_EXE}"
if [[ ! -d "$MAIN_IMAGE_DIR" ]]; then echo "❌ Critical error: Main image directory not found: ${MAIN_IMAGE_DIR}"; exit 1; fi
echo "🏞️  Using main image directory: ${MAIN_IMAGE_DIR}"

# Pipeline officially begins
cd "$BASE_PROJECT_DIR" || { echo "❌ Error: Cannot switch to project base directory ${BASE_PROJECT_DIR}"; exit 1; }
echo "📂 Current working directory: $(pwd)"
echo "📂 Creating results root directory: ${RESULTS_ROOT_DIR}"
mkdir -p "$RESULTS_ROOT_DIR"
echo "📂 Copying config file to output root directory..."
cp "$PIPELINE_CONFIG_FILE" "$RESULTS_ROOT_DIR"

# === 🚀 Step 01: Image Caption Generation (LLaVA 1.6) ===
echo ""
echo "===== 1️⃣  Starting Step 01: Image Caption Generation (LLaVA 1.6) ====="
LLAVA_PYTHON_SCRIPT="${BASE_PROJECT_DIR}/01_caption_generate/batch_llava1.6.py"
SCREEN_SESSION_NAME_STEP01="llava_caption_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP01="${STEP01_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP01}.log"
echo "⏳ Running ${LLAVA_PYTHON_SCRIPT} (in screen session ${SCREEN_SESSION_NAME_STEP01})..."
echo "    Screen log: ${SCREEN_LOG_FILE_STEP01}"
screen -S "${SCREEN_SESSION_NAME_STEP01}" -L -Logfile "${SCREEN_LOG_FILE_STEP01}" -m \
    "${PYTHON_HAD_EXE}" "${LLAVA_PYTHON_SCRIPT}" --config "${PIPELINE_CONFIG_FILE}" --output_dir "${STEP01_OUTPUT_DIR}"

# Wait for screen session to end
echo "    Waiting for screen session ${SCREEN_SESSION_NAME_STEP01} to end..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP01}"; do
    sleep 10 # Check every 10 seconds
done
echo "    Screen session ${SCREEN_SESSION_NAME_STEP01} completed."

if [[ ! -f "$STEP01_CAPTIONS_FILE" ]]; then
    echo "🚨 Critical error: Step 01 output file not found: ${STEP01_CAPTIONS_FILE}"
    echo "👉 Please check Screen log: ${SCREEN_LOG_FILE_STEP01}"
    exit 1
fi
echo "✅ Step 01 completed. Caption file located at: ${STEP01_CAPTIONS_FILE}"


# === 🚀 Step 02: Entity Extraction and Verification ===
echo ""
echo "===== 2️⃣  Starting Step 02: Entity Extraction and Verification ====="
EXTRACTOR_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_extractor.py"
MERGER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_merger.py"
CLEANER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/entity_cleaner.py"
VERIFIER_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/owlv2_verifier.py"
CORRECTOR_SCRIPT_STEP02="${BASE_PROJECT_DIR}/02_entity_extract/consistency_corrector.py"

SCREEN_SESSION_NAME_STEP02="llava_entity_extract_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP02="${STEP02_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP02}.log"
echo "⏳ Running all sub-steps of step 02 in screen session ${SCREEN_SESSION_NAME_STEP02}..."
echo "    Screen log: ${SCREEN_LOG_FILE_STEP02}"

STEP02_COMMANDS="set -e; set -o pipefail; \
echo 'Starting all sub-tasks of step 02 - '\$(date); \
echo '⏳ 2.1: Entity extraction...'; \
'${PYTHON_EVALUATE_EXE}' '${EXTRACTOR_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP01_CAPTIONS_FILE}' --output_file '${STEP02_EXTRACTED_RAW_FILE}' || exit 1; \
echo '    ✅ Entity extraction completed.'; \
echo '⏳ 2.2: Entity merging and normalization...'; \
'${PYTHON_EVALUATE_EXE}' '${MERGER_SCRIPT_STEP02}' --c2e '${STEP02_EXTRACTED_RAW_FILE}' --output_file '${STEP02_NORMALIZED_MERGED_FILE}' || exit 1; \
echo '    ✅ Entity merging completed.'; \
echo '⏳ 2.3: LLM-based entity cleaning...'; \
'${PYTHON_EVALUATE_EXE}' '${CLEANER_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_NORMALIZED_MERGED_FILE}' --output_file '${STEP02_LLM_CLEANED_FILE}' || exit 1; \
echo '    ✅ Entity cleaning completed.'; \
echo '⏳ 2.4: OWL-ViT verification...'; \
'${PYTHON_EVALUATE_EXE}' '${VERIFIER_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_LLM_CLEANED_FILE}' --output_file '${STEP02_VERIFIED_ENTITIES_FILE}' --image_dir '${MAIN_IMAGE_DIR}' || exit 1; \
echo '    ✅ OWL verification completed.'; \
echo '⏳ 2.5: Detection consistency correction...'; \
'${PYTHON_EVALUATE_EXE}' '${CORRECTOR_SCRIPT_STEP02}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_VERIFIED_ENTITIES_FILE}' --output_file '${STEP02_CORRECTED_ENTITIES_FILE}' || exit 1; \
echo '    ✅ Detection consistency correction completed.'; \
echo 'All sub-tasks of step 02 completed - '\$(date)"

screen -S "${SCREEN_SESSION_NAME_STEP02}" -L -Logfile "${SCREEN_LOG_FILE_STEP02}" -m bash -c "${STEP02_COMMANDS}"

# Wait for screen session to end
echo "    Waiting for screen session ${SCREEN_SESSION_NAME_STEP02} to end..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP02}"; do
    sleep 10
done
echo "    Screen session ${SCREEN_SESSION_NAME_STEP02} completed."

if [[ ! -f "$STEP02_CORRECTED_ENTITIES_FILE" ]]; then
    curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=Step2Error_LLaVA"
    echo "🚨 Critical error: Step 02 final output file not found: ${STEP02_CORRECTED_ENTITIES_FILE}"
    echo "👉 Please check Screen log: ${SCREEN_LOG_FILE_STEP02}"
    exit 1
fi
echo "✅ Step 02 completed. Corrected entities file located at: ${STEP02_CORRECTED_ENTITIES_FILE}"


# === 🚀 Step 03: TIDE Dataset Generation ===
echo ""
echo "===== 3️⃣  Starting Step 03: TIDE Dataset Generation (LLaVA 1.6) ====="
PY_SCRIPT_03_GENERATE="${BASE_PROJECT_DIR}/03_dataset_construct/generate_dataset.py"
if [[ ! -f "$PY_SCRIPT_03_GENERATE" ]]; then echo "🚨 Error: TIDE generation script not found: ${PY_SCRIPT_03_GENERATE}"; exit 1; fi

SCREEN_SESSION_NAME_STEP03="llava_tide_construct_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP03="${STEP03_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP03}.log"
echo "⏳ Running TIDE dataset generation in screen session ${SCREEN_SESSION_NAME_STEP03}..."
echo "    Screen log: ${SCREEN_LOG_FILE_STEP03}"

STEP03_COMMANDS="set -e; set -o pipefail; \
echo '⏳ 3.1: TIDE dataset generation...'; \
'${PYTHON_HAD_EXE}' '${PY_SCRIPT_03_GENERATE}' --config '${PIPELINE_CONFIG_FILE}' --input_file '${STEP02_CORRECTED_ENTITIES_FILE}' --output_file '${STEP03_INITIAL_TIDE_DATASET_FILE}' --no-append || { echo '🚨 Step 3.1 failed'; exit 1; }; \
echo '    ✅ Step 3.1 completed.'; \
echo 'Step 03 (dataset generation) fully completed - '\$(date)"
screen -S "${SCREEN_SESSION_NAME_STEP03}" -L -Logfile "${SCREEN_LOG_FILE_STEP03}" -m \
    bash -c "${STEP03_COMMANDS}"
    
echo "    Waiting for screen session ${SCREEN_SESSION_NAME_STEP03} to end..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP03}"; do sleep 10; done
echo "    Screen session ${SCREEN_SESSION_NAME_STEP03} completed."

if [[ ! -f "$STEP03_INITIAL_TIDE_DATASET_FILE" ]]; then 
    echo "🚨 Critical error: Step 03 final output file not found: ${STEP03_INITIAL_TIDE_DATASET_FILE}"; 
    curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=Step3Error_LLaVA";
    exit 1; 
fi
echo "✅ Step 03 completed. TIDE dataset located at: ${STEP03_INITIAL_TIDE_DATASET_FILE}"


# === 🚀 Step 04: Model Evaluation (LLaVA 1.6) ===
echo ""
echo "===== 4️⃣  Starting Step 04: Model Evaluation (LLaVA 1.6) ====="
PY_SCRIPT_04="${BASE_PROJECT_DIR}/04_model_evaluate/evaluate_llava1.6.py"
if [[ ! -f "$PY_SCRIPT_04" ]]; then echo "🚨 Error: Evaluation script (LLaVA 1.6) not found: ${PY_SCRIPT_04}"; exit 1; fi

SCREEN_SESSION_NAME_STEP04="llava_evaluate_${CONFIG_BASENAME}_${TIMESTAMP_SUFFIX}"
SCREEN_LOG_FILE_STEP04="${STEP04_SCREEN_LOG_DIR}/${SCREEN_SESSION_NAME_STEP04}.log"
echo "⏳ Running dataset evaluation in screen session ${SCREEN_SESSION_NAME_STEP04}..."
echo "    Screen log: ${SCREEN_LOG_FILE_STEP04}"

CMD_EVALUATE="'${PYTHON_HAD_EXE}' '${PY_SCRIPT_04}' --config '${PIPELINE_CONFIG_FILE}' --tide_dataset_file '${STEP03_INITIAL_TIDE_DATASET_FILE}' --image_dir '${MAIN_IMAGE_DIR}' --output_dir '${STEP04_OUTPUT_DIR}' --output_filename '${STEP04_EVAL_OUTPUT_FILENAME}' --summary_csv '${RESULTS_CSV_PATH}'"

STEP04_COMMANDS="set -e; set -o pipefail; \
echo '⏳ 4.1: Starting dataset evaluation...'; \
${CMD_EVALUATE} || { echo '🚨 Dataset evaluation failed'; exit 1; }; \
echo '    ✅ Dataset evaluation completed.'; \
echo 'Step 04 (model evaluation) fully completed - '\$(date)"
screen -S "${SCREEN_SESSION_NAME_STEP04}" -L -Logfile "${SCREEN_LOG_FILE_STEP04}" -m \
    bash -c "${STEP04_COMMANDS}"

echo "    Waiting for screen session ${SCREEN_SESSION_NAME_STEP04} to end..."
while screen -list | grep -q "${SCREEN_SESSION_NAME_STEP04}"; do
    sleep 10
done
echo "    Screen session ${SCREEN_SESSION_NAME_STEP04} completed."
echo "✅ Step 04 completed. Evaluation results located in ${STEP04_OUTPUT_DIR}."

# --- 🏁 Pipeline End ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✨✨✨ LLaVA 1.6 complete workflow successfully finished! ✨✨✨"
echo "⏱️  Total execution time: ${DURATION} seconds."
echo "📂 All related logs and output files are located under ${RESULTS_ROOT_DIR}."
echo ""
echo "📋 Screen log file locations: "
echo "➡️  Step 01 (image captioning): ${SCREEN_LOG_FILE_STEP01}"
echo "➡️  Step 02 (entity extraction): ${SCREEN_LOG_FILE_STEP02}"
echo "➡️  Step 03 (dataset generation): ${SCREEN_LOG_FILE_STEP03}"
echo "➡️  Step 04 (model evaluation): ${SCREEN_LOG_FILE_STEP04}"
echo ""
echo "📁 Key output file locations："
echo "➡️  TIDE dataset at: ${STEP03_INITIAL_TIDE_DATASET_FILE}"
echo "➡️  Dataset evaluation results at: ${STEP04_OUTPUT_DIR}/${STEP04_EVAL_OUTPUT_FILENAME}"
echo "➡️  Model evaluation summary appended to: ${RESULTS_CSV_PATH}"

curl -X "POST" "https://sctapi.ftqq.com/SCT189116TXDIE4j8RemGyBR9d116w5FaA.send?title=ExperimentOver_LLaVA"

exit 0