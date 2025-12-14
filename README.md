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

