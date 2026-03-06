# DiffAnon

<span style="font-size:1.2em">This is the repo for \"DiffAnon: Diffusion-based Prosody Control for Voice Anonymization\".</span>

This repo contains the DiffAnon training and inference pipeline using classifier-free guidance (CFG) for controlling source prosody preservation during voice anonymization.

### You can check the [Demo Webpage](https://lightensyrup.github.io/diffanon/) for speech samples.

**Requirements**

- Python 3.9+
- PyTorch + torchaudio
- See `requirements.txt` for the full environment used originally.

**Data Preparation**
This pipeline expects precomputed features alongside each `.wav`:

- `*.st_first.pt` and `*.st_all.pt` first-level codec embeddings from SpeechTokenizer
- `*.freevcs.pt` speaker embeddings (FreeVC speaker encoder)
- `*.mpm.pt` (prosody features)

You can generate these with:

```bash
python preprocess_anon.py --in_dir /path/to/wavs --out_dir /path/to/output --freevc_emb --speechtokenizer --mpm
```

**Training**

```bash
python train_anon.py --config_path config_anon.json
```

Optional resume:

```bash
python train_anon.py --config_path config_anon.json --resume_dir /path/to/logs --resume_milestone 100
```

**Inference (directory, CFG)**

```bash
python infer_anon_dir_cfg_randref.py \
  --model_path /path/to/checkpoint.pt \
  --config_path config_anon.json \
  --input_dir /path/to/input_wavs \
  --output_dir /path/to/output \
  --mode anon_pool
```

**Notes**

- `--mode` controls which conditioning signals are used (e.g., `null`, `resynt`, `anon`, `anon_pool`, `random`).
- `anon_pool` expects pooled embeddings (`*.freevcs.pt`) under the `--anon_ref_root` directory.
- CFG sampling is configured inside the inference scripts (`sampling='ddim_cfg'`, `cfg_scales=[0.8, 0]`).

**Files**

- Training entrypoint: `train_anon.py`
- Model: `model_anon.py`
- Dataset: `dataset_anon.py`
- Inference: `infer_anon_dir_cfg_randref.py`, `infer_anon_list_cfg.py`

**Credits**

- NS2VC: https://github.com/adelacvg/NS2VC
- SpeechTokenizer: https://github.com/ZhangXInFD/SpeechTokenizer
- Masked Prosody Model (MPM): https://github.com/MiniXC/masked_prosody_model
- FreeVC: https://github.com/OlaWod/FreeVC
