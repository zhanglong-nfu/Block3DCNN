# BlockCNN — Reproducibility Code (IEEE Code Artifact)

> **Core implementation withheld.** This package contains the **verification
> methodology** for the BlockCNN paper — the equivalence-checking harnesses and
> the independent standard-UNet reference — but **does NOT include the core
> BlockCNN implementation** (the chunked / domain-decomposed training backbone
> and gradient relay). The core is the authors' proprietary technique and is
> available from the corresponding author upon reasonable request, consistent
> with the paper's Data and Code Availability statement.

## What this package demonstrates

The paper's central claim is that chunked (domain-decomposed) 3-D training is
mathematically equivalent to a single whole-volume forward/backward pass. These
scripts verify that claim by comparing a chunked path against an **independent,
standard** whole-volume U-Net reference:

1. **Exactness of chunked training** — the chunked grad-relay path is
   bit-identical in forward/loss and matches the whole-volume reference to
   floating-point summation order in every parameter gradient.
2. **Memory/time trade-off** — peak activation memory is decoupled from the
   whole volume (it tracks a single chunk); the price is the halo overhead.

## Layout

```
paper/code/
├── verify_equiv_grad.py     # training-side gradient equivalence harness
├── verify_equiv.py          # inference-side exact-tiling equivalence harness
│                             #   (contains an INDEPENDENT standard-UNet reference)
├── measure_scaling.py       # memory / time scaling sweep (needs withheld core + artoo)
├── requirements.txt
└── ../config/imagecas_8ch_config.py   # optional 8-channel hyper-parameters
```

The core implementation (`model/` package: `blockcnn.py`, `blockcnn_ref_memopt.py`,
`blockcnn_backbone.py`, `blockcnn_head.py`, `blockcnn_network.py`,
`blockcnn_verify.py`, …) is **intentionally excluded** from this package.

## How the harnesses use the (withheld) core

Both scripts import the core `BlockCNN` implementation to run the comparison:

- `verify_equiv_grad.py` imports `model.backbones.blockcnn` and
  `model.blockcnn_verify` and drives the production grad-relay path.
- `verify_equiv.py` imports `model.backbones.blockcnn` and builds an
  **independent** `StandardUNet` reference (a standard encoder-decoder U-Net —
  *not* the BlockCNN technique) that the chunked path must match.

To run the harnesses, place the core `model/` package (supplied by the authors)
alongside these scripts; Python adds the script's directory to `sys.path`, so
`import model ...` resolves. The `imagecas_8ch_config.py` is resolved via
`../config` and the scripts fall back to built-in defaults if it is absent.

## What each script verifies

### `verify_equiv_grad.py` — training-side gradient equivalence
Compares the real chunked grad-relay `BlockCNN` against an independent
whole-volume reference network that shares its weights, for chunk grids
`(1,1,1)` / `(2,2,2)` / `(4,4,4)`. Reports, per parameter: max `|loss diff|`,
max `|forward logits diff|`, max `|grad diff|`.

Key environment variables (all optional; defaults shown):

| Var | Default | Meaning |
|-----|---------|---------|
| `VOLUME` | `256,256,256` | whole-volume spatial size |
| `CHUNK`  | `2,2,2`       | chunk grid `(c_x,c_y,c_z)` |
| `CHANNELS` | from config (8) | base channels |
| `BLOCKS` | `[3,3,3,3,3]` | residual blocks per stage |
| `SEP_DEPTH` | `2` | separation depth |
| `DEVICE` | `cuda:0` | compute device |
| `DTYPE` | `32` | set `64` for float64 |
| `TOL` | `1e-5` | equivalence tolerance |
| `NORM_TYPE` | `layernorm` | `layernorm` or `instancenorm` |
| `BACKBONE` | `memopt` | which backbone to verify |

```bash
cd paper/code
python verify_equiv_grad.py                       # 256^3, chunk (2,2,2)
VOLUME=512,512,512 CHUNK=4,4,4 python verify_equiv_grad.py
```

### `verify_equiv.py` — inference-side exact-tiling equivalence
An *independent* standard whole-volume U-Net (built from the same low-level
blocks, no chunking/halo/relay) is the ground truth. BlockCNN must match it
both non-chunked (`CHUNK_NUM=(1,1,1)`) and chunked (`CHUNK_NUM=(4,4,4)`).

| Var | Default | Meaning |
|-----|---------|---------|
| `CHUNK_NUM` | `4,4,4` | chunk grid |
| `VOLUME` | `512,512,512` | whole-volume size |
| `DEVICE` | `cuda:0` | compute device |
| `NORM_TYPE` | `layernorm` | normalisation |
| `GRAD_TOL` | `1e-5` | per-parameter gradient tolerance |

```bash
cd paper/code
python verify_equiv.py                            # 512^3, chunk (4,4,4)
```

### `measure_scaling.py` — memory / time scaling sweep
Sweeps `(volume, sep_depth, chunk_size)` and reports halo factor, peak GiB
(`torch.cuda.max_memory_allocated`), and s/iter. It drives the production
`BlockCNN_Network`, so it **requires the withheld core `model/` package** and
the external `artoo` framework (`artoo>=0.0.1`, plus `simpleitk`,
`scipy`, `torchgeometry`).

```bash
cd paper/code
VOLUMES=256,512 CHUNKS=2,2,2;4,4,4 SEP=2,3 CHANNELS=32 ITERS=1 \
    python measure_scaling.py
```

## Dependencies

- Python 3.11, PyTorch (CUDA build), `einops`, `tqdm`, `numpy`.
- The harnesses additionally require the withheld core `model/` package and,
  for the production network path, the external `artoo` framework
  (`artoo>=0.0.1`).

See `requirements.txt` for the pinned versions used in the paper's environment
(`nodl` conda env, Python 3.11.15).

## Fused CUDA kernels (in the withheld core)

The core uses custom fused `LayerNorm`/`GRN` CUDA kernels (JIT-compiled via
`torch.utils.cpp_extension.load`). If your toolchain cannot compile them, set
`NO_FUSED=1` to use the PyTorch fallback — numerics and peak memory are
identical.
