# BlockCNN — Data for Reproducibility (IEEE Code/Data Artifact)

## What data the article actually uses

The paper's **central claims** — exact gradient equivalence (Sections 3–4) and
the memory/time scaling result (Section 5) — are **proven analytically and
verified on random synthetic volumes**. They require *no* real dataset.

ImageCAS (whole-volume coronary-artery segmentation) is used only as a
real-world **sanity check** (Section `sec:imagecas`), and the paper states it is
"chosen to demonstrate a trainable model, not to claim accuracy leadership."
Therefore the only dataset-specific artifact needed to reproduce the paper is:

1. the **ImageCAS case-split manifest** (`imagecas_ful.csv`), and
2. the **exact preprocessing specification** the loader applies.

The raw ImageCAS volumes are ~hundreds of GB and are a **third-party dataset**
released under its own terms, so they are **not redistributed here**. Obtain them
from the official ImageCAS release and run `prepare_imagecas.py`.

## Included in this folder

| File | Purpose |
|------|---------|
| `imagecas_ful.csv` | Case-split manifest used by the experiments (1,000 cases: `image_id,type`). |
| `prepare_imagecas.py` | Standalone, dependency-light reimplementation of the loader's preprocessing (CT windowing, label remap, whole-volume resampling to 512³). Arranges the raw release into the layout the framework expects and/or exports the exact preprocessed volumes. |

## Dataset: ImageCAS

- **Source:** ImageCAS — a large-scale coronary-artery segmentation dataset on
  CT angiography (Wu et al., 2023). Obtain it from the official ImageCAS release
  / the authors (e.g. the `junkangwu/ImageCAS` repository), and honour the
  dataset's own license/access terms. **Do not redistribute the raw volumes.**
- The article cites it as `~cite{yang2023imagecas}`.

## Expected directory layout (consumed by `ImageCas_Dataset`)

```
<imagecas_root>/
└── Dataset001_CoronaryArtery/
    ├── imagesTr/   imagecas_<image_id>_0000.nii.gz   # CTA volume (float, HU)
    └── labelsTr/   imagecas_<image_id>.nii.gz         # segmentation label (int)
```

Raw label encoding (3-class task): `0` = background, `1` = left coronary artery
(LCA), `2` = right coronary artery (RCA). The 2-class task collapses LCA+RCA → 1.

## Preprocessing applied by the paper (exact)

Implemented in `ImageCas_Dataset` (`train/custom/dataset/imagecas_dataset.py`).
For each case:

1. **CT windowing** — clip to `[level − width/2, level + width/2]` with
   `level = 200 HU`, `width = 800 HU`, then normalize to `[0, 1]`.
2. **Label remap** — 2-class: `(label > 0)`; 3-class: LCA→1, RCA→2,
   background→0.
3. **Whole-volume resampling to `volume_size = [512, 512, 512]`** (no cropping;
   the whole scan is resampled to a fixed cube so the chunked backbone/head can
   partition it deterministically): images **trilinear**, labels
   **nearest-neighbour** (to keep class indices exact).
4. **Tensor shape** — image `(1, D, H, W)` float32; label `(D, H, W)` int
   (stored `int8` to shrink the GPU transfer).

This is done **on the fly** by the loader. `prepare_imagecas.py` reproduces
steps 1–3 offline, so you can inspect/verify the exact 512³ inputs the paper
uses — without the withheld BlockCNN core or the `artoo` framework.

## Split manifest (`imagecas_ful.csv`)

Columns: `image_id,type` (header included). The loader maps:

| `type` | split   | cases |
|--------|---------|-------|
| 1      | train   | 746   |
| 2      | val     | 100   |
| 3      | test    | 143   |
| 5, 6   | unused  | 11    |

Types 5/6 are edge cases not loaded by the paper.

## Running the prepare script

```bash
cd paper/data
python prepare_imagecas.py \
    --root /path/to/imagecas_root \
    --csv imagecas_ful.csv \
    --out prepared_512 \
    --volume_size 512 512 512 \
    --num_classes 3 --win_level 200 --win_width 800
```

It validates the CSV against the available volumes and writes preprocessed
`<out>/imagesTr` and `<out>/labelsTr` (`.nii.gz`, the exact 512³ inputs used by
the paper), plus a copy of the manifest.
