# Phase 0 — Hardware & Software Compatibility Report

**Date:** 2026-11-05
**Machine:** HP Victus by HP Gaming Laptop 15-fb0xxx
**Purpose:** Determine what SLM inference and PEFT fine-tuning is realistically possible before downloading anything.

All values below were measured on this machine with PowerShell / WMI / `nvidia-smi`. Nothing is assumed.

---

## 1. System

| Item | Measured value |
|---|---|
| OS | Microsoft Windows 11 Home Single Language |
| Version / Build | 10.0.26300 (64-bit) |
| Manufacturer / Model | HP / Victus by HP Gaming Laptop 15-fb0xxx |
| System installed | 2025-04-12 |
| Last boot | 2026-10-30 |

## 2. CPU

| Item | Measured value |
|---|---|
| CPU | AMD Ryzen 5 5600H with Radeon Graphics |
| Physical cores | 6 |
| Logical processors | 12 (SMT) |
| Max clock | 3301 MHz |
| L2 / L3 cache | 3 MB / 16 MB |

## 3. Memory (RAM)

| Item | Measured value |
|---|---|
| Total physical RAM | 15.34 GB usable (16 GB installed) |
| Modules | 2 × 8 GB Micron DDR4-3200 (dual channel) |
| **Free RAM at inspection time** | **2.99 GB** |
| Page file | `D:\pagefile.sys`, 12.8 GB allocated |

> ⚠️ Only ~3 GB RAM was free during inspection. Host-side data processing, tokenization and
> dataset caching need headroom. Close games / launchers / browsers before training.

## 4. GPU

| Item | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| **VRAM** | **4096 MiB (4 GB)** |
| Driver | 592.82 |
| Driver-reported CUDA version | 13.1 (max supported by driver) |
| Compute capability | 8.6 (Ampere) — architecture fact, not measured |
| Display model | WDDM (Windows Desktop) |
| Also present | AMD Radeon integrated GPU (shared, not used for ML) |
| GPU compute process | None (dGPU idle at inspection) |

## 5. CUDA / ML software state

| Item | Measured value |
|---|---|
| Python | 3.10.9 (`C:\Users\shakthi\AppData\Local\Programs\Python\Python310`) |
| py launcher | Only 3.10-64 installed |
| pip | 25.3 |
| Virtual environment | **None** — packages are installed globally |
| NVIDIA driver | Present and healthy |
| **CUDA toolkit (`nvcc`)** | **Not installed** |
| `CUDA_HOME` / `CUDA_PATH` | Empty |
| **Installed PyTorch** | **2.13.0+cpu → `torch.cuda.is_available() == False`** |
| Internet | huggingface.co HTTP 200, pypi.org HTTP 200 |

### Relevant packages already installed (global site-packages)

```
torch               2.13.0+cpu     <-- CPU build, no CUDA
transformers        5.16.1
fastapi             0.141.1
uvicorn             0.52.4
pydantic            2.12.5
pytest              9.1.1
huggingface_hub     1.29.0
safetensors         0.8.0
sentence-transformers 6.0.0
```

### Missing for this project

```
peft, trl, accelerate, bitsandbytes, datasets, PyYAML, python-dotenv
```

## 6. Disk

| Drive | Size | Free | Note |
|---|---|---|---|
| C: (Windows) | 181.2 GB | **17.3 GB** | Tight — OS drive |
| D: (New Volume) | 294.8 GB | **129.8 GB** | Recommended location for venv + model cache |

Existing Hugging Face cache: `C:\Users\shakthi\.cache\huggingface` — 0.09 GB / 35 files.
No Ollama, LM Studio, or other local model runtimes found.

## 7. Project directory

`C:\Users\shakthi\OneDrive\Desktop\SLM-project` is **empty**. No venv, no existing code, no git repo.
It is inside **OneDrive**, which will attempt to sync large model weights/checkpoints. Avoid storing
models or the virtual environment there.

---

## 8. Compatibility assessment

### Verdict: 4 GB VRAM is viable for a ~1.5–1.7B model with **QLoRA**, not for full fine-tuning.

| Workload | Feasibility on this machine | Reasoning |
|---|---|---|
| 1.5B inference, fp16 | ✅ Yes (~3.1 GB weights) | Fits 4 GB, tight; short context |
| 1.5B inference, 4-bit NF4 | ✅ Yes (~1.1 GB weights) | Comfortable, leaves room for KV cache |
| QLoRA fine-tune 1.5B, 4-bit, seq 512, micro-batch 1 + grad ckpt | ✅ Yes (borderline) | ~3.0–3.8 GB estimated VRAM |
| QLoRA fine-tune 1.7B (SmolLM2), 4-bit | ✅ Borderline | Slightly larger than 1.5B; must keep seq ≤ 512 |
| LoRA (non-quantized) 1.5B fp16 | ⚠️ Very tight / likely OOM | 3.1 GB weights + activations > 4 GB |
| Full fine-tune 1.5B | ❌ No | Needs ≥ 12–16 GB VRAM |
| QLoRA fine-tune Phi-3.5-mini (3.8B) | ❌ No | 4-bit weights ≈ 2.2 GB + activations/optimizer overhead exceeds 4 GB |
| CPU-only fine-tune 1.5B | ❌ Impractical | Hours–days on 6C/12T; not a sensible fallback |

### Blockers that must be fixed before Phase 3/4

1. **PyTorch is CPU-only.** The NVIDIA driver supports CUDA 13.1, but the installed wheel does not.
   A CUDA-enabled PyTorch wheel must be installed inside a fresh virtual environment. A full CUDA
   toolkit install is **not** required — PyTorch wheels bundle the CUDA runtime.
2. **No virtual environment.** Global packages should not be modified; create an isolated venv.
3. **C: has only 17.3 GB free.** Model download + cache + checkpoints should live on `D:` via
   `HF_HOME` / `TRANSFORMERS_CACHE`, and the venv should ideally live on `D:` too.
4. **OneDrive working directory.** Keep `experiments/` checkpoints and HF cache out of OneDrive.
5. **~3 GB free RAM.** Close Epic Games Launcher / games before dataset and training runs.
6. **Windows + `bitsandbytes`.** 4-bit QLoRA depends on bitsandbytes Windows support; this will be
   verified with a small import test in Phase 4. If it fails, the documented fallback is a reduced
   fp16 LoRA configuration on GPU, and if that OOMs, an explicitly labelled alternative adaptation
   experiment — not a silent skip.
7. **`transformers 5.16.1` is very new.** PEFT/TRL compatibility must be pinned together in the venv
   rather than relying on the global install.

### Recommended SLM configuration (to be finalized in Phase 1)

- **Model family:** ~1.5B–1.7B instruction-tuned model (Qwen2.5-1.5B-Instruct or SmolLM2-1.7B-Instruct).
  Phi-3.5-mini (3.8B) is **not feasible** for adaptation on 4 GB VRAM.
- **Inference:** 4-bit NF4 (or fp16 for quality baseline), `max_new_tokens` 128–256, batch size 1.
- **Adaptation:** QLoRA, 4-bit NF4 + double quantization, compute dtype bf16 (Ampere supports it),
  LoRA `r=16`, `alpha=32`, `dropout=0.05`, all attention + MLP projection targets,
  LR ~2e-4, micro-batch 1, gradient accumulation 8–16, max sequence length 512,
  gradient checkpointing on, 2–3 epochs.
- **Storage:** venv and `HF_HOME` on `D:`, adapters + logs in the project (small files only).

### Estimated resource footprint (estimates, not measurements)

| Artifact | Approx. size |
|---|---|
| Base model download (safetensors, fp16/bf16) | ~3.0–3.5 GB |
| 4-bit quantized in-memory weights | ~0.9–1.2 GB |
| LoRA adapter checkpoint | ~20–80 MB |
| Experiment logs / eval outputs | < 50 MB |

---

## 9. Commands used (reproducible)

```powershell
Get-CimInstance Win32_OperatingSystem
Get-CimInstance Win32_Processor
Get-CimInstance Win32_ComputerSystem
Get-CimInstance Win32_PhysicalMemory
Get-CimInstance Win32_VideoController
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"
nvidia-smi
python --version
py -0p
python -m pip --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip list
```

## 10. Phase 0 status

- Environment inspected: **yes**
- Hardware compatibility report produced: **yes**
- Large downloads / installs performed: **no**
- Model selected: **not yet** (Phase 1)
- Blocking issues identified: **yes** (CPU-only torch, no venv, low free disk on C:, low free RAM)
- **Awaiting user confirmation before Phase 1.**

---

## 11. Phase 3 addendum (2026-11-05) — measured baseline environment

Measured after the approved pivot to `HuggingFaceTB/SmolLM2-135M-Instruct`:

| Item | Measured value |
|---|---|
| Active Python | 3.10.9, global interpreter |
| Torch at baseline time | `2.13.0+cpu` (CUDA unavailable) |
| Model load (CPU) | 7.916 s |
| 5-prompt inference | p50 4488.207 ms, p95 6641.939 ms, greedy, max_new_tokens 256 |
| Peak VRAM | 0 MB (CPU run) |

The CUDA wheel `torch==2.13.0+cu130` is 1.9 GB. The PyTorch CDN was measured between roughly
0.09 MB/s and 0.86 MB/s during this session. A detached background install was attempted, but the
download stalled with zero bytes read and no cache growth for over 10 minutes, so the process was
stopped. No GPU inference or training numbers are claimed until a stable download completes and an
actual GPU run is executed.

Static check for the 135M model: bf16 weights ~269 MB, 4-bit NF4 weights ~67 MB, and the planned
r=16 LoRA adapter is ~4.88M trainable parameters. All fit within 4 GB in principle; this is a
sizing estimate, not a measured training result.
