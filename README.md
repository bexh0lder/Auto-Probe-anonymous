Note: Make sure to complete all `[necessary]` fields in the /experiments/pipline_llava1.6.yaml configuration file prior to execution.

Please download:
- https://huggingface.co/google/owlv2-large-patch14-ensemble
- https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b

```bash
conda env create -f environment.yaml

conda activate auto

chmod +x pipeline_llava1.6.sh

./pipeline_llava1.6.sh
```
