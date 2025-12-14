# Modeling and Scaling of Generative AI System - MixDQ on SDXL-Turbo
## 1. The repository layout
This is the repository of our final project.
We study how mixed-precision quantization and a simple adaptive resolution controller affect the performance of SDXL Turbo in an online service. We chose to focus on MixDQ models only (W8/W6/W4) and left FP16 aside.

Our goals:
1. Build a perfomance model under different weight bit-widths (W8/W6/W4) and resolution (512, 768, 1024).
2. Evaluate latency under two synthetic workloads, burst and Poisson.
3. Compare a static policy with an adaptive controller.

Main folders:

- `configs/` – Diffusers / SDXL-Turbo configuration files.
- `scripts/`  
  - `txt2img.py` – FP16 SDXL-Turbo image generation.  
  - `quant_txt2img.py` – MixDQ image generation (W8/W6/W4).  
  - `eval_fp16_mixdq_cleanfid.py` – FID & CLIPScore evaluation.  
  - `eval_grid.py` – grid over bit-widths × resolutions.  
  - `latency_probe.py` – latency / VRAM probing for different modes.  
  - `load_test_stub.py` – queue simulator (burst / Poisson) using the latency model.
  - `utils/` – prompts, COCO helper scripts, etc.
- `quant_utils/` – MixDQ implementation and wrappers around SDXL-Turbo.
- `mixed_precision_scripts/` – YAML configs and sensitivity files for weight / activation quantization.
- `results/` – CSV files and plots produced by our experiments.
- `logs/` – local generation logs and intermediate artifacts (ignored in the cleaned MSGAI repo).
- `report/` – LaTeX / PDF for the MSGAI project report (if present).


## bash commands -

## 0. Environment / PYTHONPATH

```bash
export PYTHONPATH=".":"$(pwd)/quant_utils"
```

## 1. Image generation & FID/CLIP on COCO

### 1.1 FP16 generation (COCO prompts)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py \
  --config ./configs/stable-diffusion/sdxl_turbo.yaml \
  --base_path ./logs/sdxl_fp_eval_big \
  --num_imgs 1024 \
  --batch_size 4 \
  --fp16
```

Images will be saved under:

```text
./logs/sdxl_fp_eval_big/generated_images
```

### 1.2 MixDQ W8 generation

```bash
WEIGHT_MP_CFG="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_8.00.yaml"
ACT_MP_CFG="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_7.77.yaml"
ACT_PROTECT="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt"

CUDA_VISIBLE_DEVICES=0 python scripts/quant_txt2img.py \
  --base_path "./logs/sdxl_mixdq_eval" \
  --image_folder "./logs/sdxl_mixdq_eval_images" \
  --batch_size 4 \
  --num_imgs 1024 \
  --config_weight_mp "$WEIGHT_MP_CFG" \
  --config_act_mp "$ACT_MP_CFG" \
  --act_protect "$ACT_PROTECT" \
  --fp16 \
  --wbits 8
```

Images will be saved under:

```text
./logs/sdxl_mixdq_eval_images
```


### 1.3 FID + CLIP evaluation (FP16 vs MixDQ, COCO val2014)

```bash
python scripts/eval_fp16_mixdq_cleanfid.py \
  --ref_folder ./scripts/utils/val2014 \
  --fp16_folder ./logs/sdxl_fp_eval_big/generated_images \
  --mixdq_folder ./logs/sdxl_mixdq_eval_images \
  --prompts_file ./scripts/utils/prompts.txt \
  --batch_size 16 \
  --device cuda \
  --num_samples 1024
```


## 2. Global grid: bit-width × resolution (grid_bits_res_32.csv)

```bash
export PYTHONPATH=".":"$(pwd)/quant_utils"

python scripts/eval_grid.py \
  --config ./configs/stable-diffusion/sdxl_turbo.yaml \
  --ckpt ./logs/sdxl_mixdq_eval/ckpt.pth \
  --prompts_file ./scripts/utils/prompts_32.txt \
  --out_csv ./logs/grid_bits_res_32.csv \
  --res 512 768 1024 \
  --wbits 8 6 4
```

Output:

```text
./logs/grid_bits_res_32.csv
```


## 3. Latency probe (p50/p95 + VRAM vs resolution)

### 3.1 FP16

```bash
python scripts/latency_probe.py \
  --mode fp16 \
  --base_path ./logs/sdxl_mixdq_eval \
  --res 512 768 1024 \
  --repeats 30 \
  --out ./logs/latency_fp16.csv
```


### 3.2 W8A8

```bash
python scripts/latency_probe.py \
  --mode w8a8 \
  --base_path ./logs/sdxl_mixdq_eval \
  --config_weight_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_8.00.yaml \
  --config_act_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_7.77.yaml \
  --act_protect ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt \
  --res 512 768 1024 \
  --repeats 30 \
  --out ./logs/latency_w8a8.csv
```


### 3.3 W6A8

```bash
python scripts/latency_probe.py \
  --mode w6a8 \
  --base_path ./logs/sdxl_mixdq_eval \
  --config_weight_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_6.00.yaml \
  --config_act_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_7.77.yaml \
  --act_protect ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt \
  --res 512 768 1024 \
  --repeats 30 \
  --out ./logs/latency_w6a8.csv
```


### 3.4 W4A8

```bash
python scripts/latency_probe.py \
  --mode w4a8 \
  --base_path ./logs/sdxl_mixdq_eval \
  --config_weight_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight/weight_4.00.yaml \
  --config_act_mp ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_7.77.yaml \
  --act_protect ./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt \
  --res 512 768 1024 \
  --repeats 30 \
  --out ./logs/latency_w4a8.csv
```

## Static vs Adaptive policy
```export PYTHONPATH=".:$(pwd)/quant_utils"

#makes Poisson arrivals repeatable
```export MIXDQ_LOADTEST_SEED=123

#Runs the queue tests
```python scripts/load_test_stub.py
#Creates txt and csv results files at
#./logs/sdxl_mixdq_eval/queue_tests/<timestamp>/
 #mixdq_queue_results.csv
#mixdq_queue_results.txt

#Plot results
```python scripts/plot_mixdq_queue_results.py ./logs/sdxl_mixdq_eval/queue_tests/<timestamp>/mixdq_queue_results.csv



