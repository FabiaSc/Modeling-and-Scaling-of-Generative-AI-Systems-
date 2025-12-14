import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.cuda import amp
from contextlib import nullcontext
import json
import sys

from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, prepare_coco_text_and_image
from controller import ResolutionController

from tqdm.auto import tqdm
import sys

# Uses a simple resolution controller to choose resolution based on queue length
rc = ResolutionController(q_low=1, q_high=4, default=1024)
logger = logging.getLogger(__name__)

def _resolve_weight_cfg(wbits: int) -> str:
    # Resolves the default mixed-precision weight config path for a given bit-width
    base = "./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight"
    return os.path.join(base, f"weight_{wbits:.2f}.yaml")  # e.g. weight_6.00.yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="how many batches to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to config which constructs model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=32,
        help="the number of the output images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_act",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_weight",
        action='store_true',
    )
    parser.add_argument(
        "--config_weight_mp",
        type=str,
        help="path for weight configs",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        help="path for act configs",
    )
    parser.add_argument(
        "--act_protect",
        type=str,
        help="the path for extremely sensitive acts",
    )
    # Optional external queue length used by the resolution controller
    parser.add_argument(
        "--queue_len",
        type=int,
        default=None,
        help="External queue length for load-dependent control"
    )
    # ---- Optional flags for systems experiments ----
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write PNG files to disk (avoid I/O dominating latency in load tests).",
    )
    parser.add_argument(
        "--no_file_log",
        action="store_true",
        help="Disable logging to run.log (useful when running many concurrent processes).",
    )
    parser.add_argument(
        "--metrics_stdout",
        action="store_true",
        help="Print a single JSON metrics line to stdout prefixed with '__METRICS__'.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help="Optional path to write the same metrics JSON (one file per invocation).",
    )


    # Chooses default weight bit-width when no explicit mixed-precision config is set
    parser.add_argument(
        "--wbits",
        type=int,
        choices=[4, 6, 8],
        default=8,
        help="Weight bit-width (4,6 or 8)"
    )

    # Allows overriding steps and cfg, tuned for SDXL-Turbo style runs
    parser.add_argument(
        "--override_steps",
        type=int,
        default=None,
        help="Force num_inference_steps (e.g. 1 for SDXL-Turbo). If None, keeps default.",
    )
    parser.add_argument(
        "--override_cfg",
        type=float,
        default=None,
        help="Force guidance_scale (e.g. 0.0 for SDXL-Turbo). If None, keeps default.",
    )

    opt = parser.parse_args()

    # ---- Timing basepoints (for queue-service experiments) ----
    opt._t_total0 = time.perf_counter()
    opt._t_init0 = time.perf_counter()


    # Auto-resolves weight mixed-precision config from wbits if not provided
    if opt.config_weight_mp is None and not opt.skip_quant_weight:
        opt.config_weight_mp = _resolve_weight_cfg(opt.wbits)

    # Disables activation quantization when no activation config is provided
    if opt.config_act_mp is None:
        opt.skip_quant_act = True

    seed_everything(opt.seed)

    # Selects final sampler steps and cfg, falling back to simple Turbo defaults
    default_steps = 1
    chosen_steps = opt.override_steps if opt.override_steps is not None else default_steps
    chosen_cfg = opt.override_cfg if opt.override_cfg is not None else 0.0

    # Stores resolved steps and cfg on opt for consistent use in logs and generation
    opt._steps = int(chosen_steps)
    opt._cfg = float(chosen_cfg)

    opt.outdir = os.path.join(opt.base_path, 'generated_images')
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    log_path = os.path.join(opt.base_path, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=(
            [logging.StreamHandler()] if opt.no_file_log else
            [logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
        ),
    )
    logger = logging.getLogger(__name__)
    # Logs the core generation hyperparameters for this run
    logger.info(f"Using steps={opt._steps}, cfg={opt._cfg}, wbits={opt.wbits}")

    # load the config from the log path
    if opt.config is None:
        opt.config = os.path.join(opt.base_path, 'config.yaml')
    if opt.ckpt is None:
        opt.ckpt = os.path.join(opt.base_path, 'ckpt.pth')
    if opt.image_folder is None:
        opt.image_folder = os.path.join(opt.base_path, 'generated_images')
    config = OmegaConf.load(f"{opt.config}")

    # Prints which config file and model identifier are being used
    print("CONFIG FROM:", opt.config)
    print("MODEL ID   :", config.model.get("pretrained_model_name_or_path", "N/A"))

    if opt.cfg is None:
        opt.cfg = config.calib_data.scale_value

    # adapter_id = getattr(config.model, "adapter_id", None)
    # adapter_cache_dir = getattr(config.model, "adapter_cache_dir", None)
    model, pipe = get_model(config.model, fp16=opt.fp16, return_pipe=True, convert_model_for_quant=True)
    num_timesteps = config.calib_data.n_steps

    #chosen_steps = opt.override_steps if opt.override_steps is not None else num_timesteps
    #chosen_cfg   = opt.override_cfg if opt.override_cfg is not None else (opt.cfg if opt.cfg is not None else 0.0)

    assert (config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params, \
        act_quant_params=aq_params, \
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    dtype = torch.float32 if not opt.fp16 else torch.float16
    if opt.fp16:
        qnn = qnn.half()  # make some newly genrated quant-related modules FP16

    qnn.set_quant_state(False, False)
    calib_added_cond = {}
    calib_added_cond["text_embeds"] = torch.randn(1, 1280, dtype=dtype).cuda().to(dtype)
    calib_added_cond["time_ids"] = torch.randn(1, 6, dtype=dtype).cuda().to(dtype)

    with torch.no_grad():
        if config.model.model_type == "sdxl":
            _ = qnn(torch.randn(1, 4, 64, 64).cuda().to(dtype), \
                    torch.randint(0, 1000, (1,)).cuda().to(dtype), \
                    torch.randn(1, 77, 2048).cuda().to(dtype), \
                    added_cond_kwargs=calib_added_cond)
        elif config.model.model_type == "sd":
            _ = qnn(torch.randn(1, 4, 64, 64).cuda().to(dtype), \
                    torch.randint(0, 1000, (1,)).cuda().to(dtype), \
                    torch.randn(1, 77, 768).cuda().to(dtype))

    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant)  # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.ckpt, dtype=dtype)

    # Forward
    if opt.prompt is None:
        json_file = "./scripts/utils/captions_val2014.json"
        prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
        prompts = prompt_list[0:opt.num_imgs]
    else:
        prompts = [opt.prompt] * opt.num_imgs

    use_weight_mp = opt.config_weight_mp is not None
    use_act_mp = opt.config_act_mp is not None

    # inference with the quantized model with
    if use_weight_mp:
        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("---------------- load the bitwidth config for weight! -------------------")
        logger.info(f"------------------ config: {opt.config_weight_mp} ---------------------")

        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        if use_weight_mp and not use_act_mp:
            logger.info("-------- Inference with weight-only quantized -----------")
            # Uses opt._steps so generation stays consistent with override logic
            opt._init_ms = (time.perf_counter() - opt._t_init0) * 1000.0
            gen_image(prompts, qnn, pipe, opt._steps, opt)
            return None
    if use_act_mp:
        # protect the extremely sensitive layer
        acts_protected = torch.load(opt.act_protect)
        qnn.set_layer_quant(model=qnn, module_name_list=acts_protected, quant_level='per_layer', weight_quant=False, act_quant=False)

        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("load the bitwidth config for weight!")
        logger.info(f"the config for weight is {opt.config_weight_mp}")
        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        with open(opt.config_act_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("------------- load the bitwidth config for act! \nInference with w&a quantized! ------------")
        logger.info(f"------------ config: {opt.config_act_mp} -----------------")

        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='act')
        # Again uses opt._steps so mixed-precision path shares the same sampler settings
        opt._init_ms = (time.perf_counter() - opt._t_init0) * 1000.0
        gen_image(prompts, qnn, pipe, opt._steps, opt)
        return None
    else:
        logger.info("Inference without mixed precision!")
        # Falls back to pure quantization using the resolved steps and cfg
        opt._init_ms = (time.perf_counter() - opt._t_init0) * 1000.0
        gen_image(prompts, qnn, pipe, opt._steps, opt)
        return None


def gen_image(prompt, unet, pipe, num_timesteps, opt):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(opt.seed)  # Seed generator to create the initial latent noise

    # Accumulators for detailed timing (optional)
    gen_gpu_ms_sum = 0.0
    gen_wall_ms_sum = 0.0
    save_ms_sum = 0.0
    res_hist = []

    total = len(prompt)
    batch_size = opt.batch_size
    assert (total >= batch_size), "the length of prompts should larger than batch_size"
    num = total // batch_size

    img_id = 0
    logger.info(f"starting from image {img_id}")

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    pipe.unet = unet.half() if opt.fp16 else unet
    #pipe.unet = unet
    pipe.to("cuda")
    with torch.no_grad():
        for i in tqdm(
            range(num), desc="Generating image samples for FID evaluation."
        ):
            with amp.autocast(enabled=False):

                # Derives an effective queue length and asks the controller for a resolution
                queue_len = 0 if opt.queue_len is None else max(0, opt.queue_len)
                r = rc.choose(queue_len)  # picks among 512, 768, 1024
                res_hist.append(int(r))
                t_wall0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                image = pipe(
                    prompt=prompt[img_id:img_id + batch_size],
                    height=r,
                    width=r,
                    num_inference_steps=num_timesteps,   # num_timesteps already comes from opt._steps
                    guidance_scale=opt._cfg              # uses the cfg scale resolved in main
                ).images

                if torch.cuda.is_available():
                    ev1.record()
                    torch.cuda.synchronize()
                    gen_gpu_ms_sum += ev0.elapsed_time(ev1)
                gen_wall_ms_sum += (time.perf_counter() - t_wall0) * 1000.0


            for j in range(batch_size):
                if not opt.no_save:
                    t_s0 = time.perf_counter()
                    image[j].save(f"{opt.image_folder}/{img_id}.png")
                    save_ms_sum += (time.perf_counter() - t_s0) * 1000.0
                img_id += 1



    # ---- Emit metrics (optional) ----
    t_total1 = time.perf_counter()
    total_ms = (t_total1 - getattr(opt, "_t_total0", t_total1)) * 1000.0
    metrics = {
        "init_ms": round(float(getattr(opt, "_init_ms", float('nan'))), 3) if getattr(opt, "_init_ms", None) is not None else None,
        "gen_gpu_ms": round(float(gen_gpu_ms_sum), 3),
        "gen_wall_ms": round(float(gen_wall_ms_sum), 3),
        "save_ms": round(float(save_ms_sum), 3),
        "total_ms": round(float(total_ms), 3),
        "num_imgs": int(getattr(opt, "num_imgs", -1)),
        "batch_size": int(getattr(opt, "batch_size", -1)),
        "wbits": int(getattr(opt, "wbits", -1)) if getattr(opt, "wbits", None) is not None else None,
        "queue_len": int(getattr(opt, "queue_len", 0)) if getattr(opt, "queue_len", None) is not None else None,
        "res_hist": res_hist,
    }

    if getattr(opt, "metrics_json", None):
        try:
            os.makedirs(os.path.dirname(opt.metrics_json), exist_ok=True)
        except Exception:
            pass
        with open(opt.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f)

    if getattr(opt, "metrics_stdout", False):
        print("__METRICS__ " + json.dumps(metrics), flush=True)

if __name__ == "__main__":
    main()
