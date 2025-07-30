# Run evaluation and record result summary to all_pope_summaries.csv file
# methods=("popular" "random" "adversarial")
# for method in "${methods[@]}"; do
#     echo "Testing llava POPE method: ${method}"
#     python evaluate_llava_pope.py \
#         --model_path /root/autodl-tmp/models/liuhaotian/llava-v1.5-7b \
#         --question_file "/root/Auto-Probe/experiments/results/coco/llava/pope/coco_pope_${method}.json" \
#         --image_dir /root/autodl-tmp/datasets/coco/val2014 \
#         --output_file /root/Auto-Probe/experiments/results/coco/llava/pope/llava-v1.5-7b_${method}_results.json \
#         --summary_csv /root/Auto-Probe/experiments/results/coco/results.csv
# done
# methods=("popular" "random" "adversarial")
# methods=("adversarial")
# for method in "${methods[@]}"; do
#     echo "Testing instructblip POPE method: ${method}"
#     python evaluate_instructblip_pope.py \
#         --model_name "blip2_vicuna_instruct" \
#         --model_type "vicuna7b" \
#         --question_file "/root/Auto-Probe/experiments/results/coco/llava/pope/coco_pope_${method}.json" \
#         --image_dir /root/autodl-tmp/datasets/coco/val2014 \
#         --output_file /root/Auto-Probe/experiments/results/coco/instructblip/pope/blip2_vicuna_instruct_${method}_results.json \
#         --summary_csv /root/Auto-Probe/experiments/results/coco/results.csv 
# done

# for method in "${methods[@]}"; do
#     echo "Testing mplug_owl POPE method: ${method}"
#     python evaluate_mplugowl_pope.py \
#         --model_path "/root/autodl-tmp/models/MAGAer13/mplug-owl-llama-7b" \
#         --question_file "/root/Auto-Probe/experiments/results/coco/llava/pope/coco_pope_${method}.json" \
#         --image_dir "/root/autodl-tmp/datasets/coco/val2014" \
#         --output_file /root/Auto-Probe/experiments/results/coco/mplug_owl/pope/mplug-owl-llama-7b_${method}_results.json \
#         --summary_csv /root/Auto-Probe/experiments/results/coco/results.csv 

# done
# methods=("popular" "random" "adversarial")
# for method in "${methods[@]}"; do
#     echo "Testing mplug_owl2 POPE method: ${method}"
#     python evaluate_mplugowl2_pope.py \
#         --model_path "/root/autodl-tmp/models/MAGAer13/mplug-owl2-llama2-7b" \
#         --question_file "/root/Auto-Probe/experiments/results/coco/llava/pope/coco_pope_${method}.json" \
#         --image_dir "/root/autodl-tmp/datasets/coco/val2014" \
#         --output_file /root/Auto-Probe/experiments/results/coco/mplug_owl2/pope/mplug-owl2-llama2-7b_${method}_results.json \
#         --summary_csv /root/Auto-Probe/experiments/results/coco/results.csv 

# done

methods=("popular" "random" "adversarial")
for method in "${methods[@]}"; do
    echo "Testing minigpt4_v2 POPE method: ${method}"
    python evaluate_minigpt4_v2_pope.py \
        --minigpt4_cfg_path /root/Auto-Probe/MiniGPT-4/eval_configs/minigptv2_eval.yaml \
        --model_checkpoint_path /root/autodl-tmp/models/checkpoint_stage3.pth \
        --question_file "/root/Auto-Probe/experiments/results/coco/llava/pope/coco_pope_${method}.json" \
        --image_dir "/root/autodl-tmp/datasets/coco/val2014" \
        --output_file /root/Auto-Probe/experiments/results/coco/mplug_owl2/pope/minigpt4v2_${method}_results.json \
        --summary_csv /root/Auto-Probe/experiments/results/coco/results.csv 

done

