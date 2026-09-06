# Image Forgery Detector (IFD)

Classical forensics pipeline in Python that fuses **Error Level Analysis (ELA)**
with **blind noise-residual forensics** to judge whether a single image is
tampered with, and *where*. Two fusion tiers exist: a brittle heuristic
baseline and a learned block classifier (RandomForest). No GPU, no PyTorch.

```
 suspect image ──▶ ELA branch ─┐
                    ┌──────────┴──▶ FUSION ──▶ verdict + heatmap + mask
                    └─▶ noise branch ─┘
```

## The core idea

A genuine photo has **one uniform compression / sensor-noise history**. A
tampered photo contains regions with a **mismatched history**. IFD measures
two orthogonal fingerprints of that history and fuses them:

| Branch | Signal measured | Target manipulation |
|---|---|---|
| **ELA** | JPEG re-compression loss per block (`ifd/features/ela.py`) | Splicing / re-save artifacts |
| **Noise** | Local pixel-noise variance of the denoising residual (`ifd/features/noise.py`) | Spliced / synthetic patches |

Each branch uses the classic ELA recipe internally: re-save the image at a
JPEG quality, amplify the absolute difference, inspect the map. The learned
tier turns the raw maps into per-`64×64`-block statistics:
`[mean, std, p95, max]` over `{ela, noise, gray}` = 12 features per block,
then runs a RandomForest. The verdict requires a **contiguous cluster of ≥ 2
strongly-flagged blocks** (plus a peak probability ≥ 0.5), which drops the
classic single-texture-block false-positive mode.

## Quick start

Python 3.12+; dependencies in `requirements.txt`.

```powershell
pip install -r requirements.txt
```

Scripts self-bootstrap the project root onto `sys.path`, so they run from any
working directory with either `python` or `py`, no `PYTHONPATH` needed.
Anything interactive (`import ifd`, Jupyter) keeps the repo root as the working
directory.

### 1. Single image, heuristic tier

```powershell
python scripts/demo.py --image path/to/image.jpg
```

Prints a verdict + confidence and writes an HTML report
(`reports/single.html`) with side-by-side ELA / noise / fused-heat / overlay
panels.

### 2. Single image, learned tier

```powershell
python scripts/demo.py --image path/to/image.jpg --model models/block_rf.joblib
```

### 3. Smoke test (the pass/fail contract)

```powershell
python scripts/smoke_test.py
```

Trains a model on synthetic mixed-history forgeries, verifies verdicts +
localization end-to-end, and asserts the result:

```
=== smoke (learned) ===
  images  n=12  acc=1.000  prec=1.000  rec=1.000  F1=1.000  AUC=1.000
  pixels  n=6   IoU=0.551  Dice=0.707  F1=0.712  P=0.558   R=0.981
SMOKE TEST: PASS
```

## CLI reference

| Command | Purpose |
|---|---|
| `scripts/demo.py --image PATH [--model MODEL] [--out DIR]` | Single-image HTML report. |
| `scripts/demo.py --casia ROOT [--casia1 ROOT] [--columbia ROOT] [--max-images N]` | Dataset evaluation (heuristic by default; `--model` switches to learned). |
| `scripts/train_blocks.py` | Train the learned tier and print a held-out report. |
| `scripts/make_synthetic.py` | Generate detectable-by-construction synthetic forgery pairs. |
| `scripts/smoke_test.py` | Train → detect → evaluate → assert end-to-end. |

## Configuration

`config.toml` at the repository root is the single source of truth for paths
and training-data sources. All relative paths resolve against the repo root,
so the scripts behave the same from any working directory. Flags still
override per run:

```toml
[training]
authentic = "data/authentic"      # real photos added as label=0; "" disables
casia = ""                        # real dataset root; "" leaves it out
keep_synthetic = true             # mix synthetic pairs when casia is used

[paths]
models = "models"                 # train_blocks --out default
reports = "reports"               # demo --out default
synthetic = "synthetic"           # make_synthetic --out default
probes = ["test.jpg", "test2.jpg"]  # held-out verification images
```

Move a dataset, change where models are saved, or point at a different
authentic folder by editing `config.toml`, not the scripts. `ifd/config.py`
loads it (stdlib `tomllib`, no extra dependency).

### Training the learned tier

```powershell
# Synthetic mixed-history pairs only.
python scripts/train_blocks.py --n-train 14 --n-test 8

# Mix in real authentic photos, which fixes the "real photo = tampered" shortcut
# a synthetic-only model learns. verifiable held-out probes: test.jpg / test2.jpg.
python scripts/train_blocks.py --n-train 14 --n-test 8 --authentic-root data/authentic --real-block-budget 2500
```

Notes:

- Training uses a **random ELA quality per image** from the sweep
  `{70, 85, 95}`; inference always uses the default (85). This teaches the
  classifier to work without knowing the suspect's JPEG provenance.
- The shipped `models/block_rf.joblib` is trained on the **synthetic + real
  authentic mix** above and reproduces the probe numbers below. Retrain any
  time the features, denoiser, or training-data mix changes: the learned
  verdict is calibrated to the feature definition *and* the data mix.
- `--dataset-root` trains on real labeled datasets (CASIA v2 / Columbia).
  It replaces the synthetic set unless you pass `--keep-synthetic` to mix
  both. For CASIA, masks pair automatically (~97%: covers the official
  `CASIA 2 Groundtruth` folder and `_gtN` filename variants), and samples
  are drawn evenly across the dataset rather than from the alphabetically
  first subfolder. Measured outcome, see "Honest expectations": this does
  not currently beat the shipped mix.

### Synthetic forgeries are mixed-history by construction

`make_synthetic.py` does the classic ELA setup: host → JPEG encode at `qh`,
donor → JPEG encode at much lower `qp`, paste, re-encode composite at `qh`.
The host region then round-trips at its own quality (a JPEG fixpoint), while
the patch still carries the heavy `qp` quantization loss and lights up at
re-save `qh`. Splicing raw arrays and encoding once produces **no** ELA signal.

```powershell
python scripts/make_synthetic.py --out synthetic --n 1 --seed 0   # auth_0.jpg + forged_0.jpg + mask
```

## Python API

```python
from ifd import Detector, DetectorConfig
from ifd.data import load_image, load_mask

img = load_image("suspect.jpg")

det = Detector()                                          # heuristic tier
result = det.detect(img)                                  # -> DetectionResult

clf = ...  # joblib.load("models/block_rf.joblib")
det = Detector(classifier=clf)                            # learned tier

result.is_tampered   # bool
result.confidence    # float 0..1
result.heatmap       # float32 (H, W) soft localization
result.mask          # uint8 {0, 255} hard localization
result.bboxes        # [(y0, x0, y1, x1), ...]
result.feature_maps  # {'ela', 'noise', 'classifier'} normalized [0,1]
```

Images are **RGB `uint8`** at the API boundary; masks are **`uint8 {0, 255}`**
and feature maps are **float32, channel-last**. `build_feature_maps()` in
`ifd/detector.py` is the single canonical feature extractor shared by training
and inference.

## Repository layout

| Path | Responsibility |
|---|---|
| `ifd/detector.py` | `Detector` orchestration, both branches, `build_feature_maps()`. |
| `ifd/features/` | `ela.py`, `noise.py` (wavelet denoise residual), `prnu.py` (stretch track). |
| `ifd/fusion/` | `heuristic_fusion` + `BlockFeatureClassifier` (RandomForest). |
| `ifd/eval/metrics.py` | Image-level (acc/F1/ROC-AUC) + pixel-level (IoU/Dice/F1), per dataset. |
| `ifd/data/` | CASIA v1/v2 + Columbia loaders, mask pairing, known-broken-file skip list. |
| `scripts/` | `demo`, `train_blocks`, `make_synthetic`, `smoke_test`. |
| `config.toml` | Single source of truth for paths and training-data sources. |
| `models/` | Shipped `block_rf.joblib` (gitignored; regenerate with `train_blocks.py`). |
| `data/authentic/` | Known-authentic real photos used as label=0 training data. |
| `test.jpg` / `test2.jpg` | Known-authentic held-out verification probes. |

## Evaluation harness

`ifd/eval/metrics.py` reports **per dataset** (never a single aggregate):

- **Image-level**: accuracy, precision, recall, F1, ROC-AUC.
- **Pixel-level** (where ground-truth masks exist): IoU, Dice, pixel F1/P/R.

`demo.py --casia/--casia1/--columbia` wires a dataset up and prints the report.
Images and masks are decoded lazily one at a time, so memory stays flat no
matter how large the dataset is.

## Honest expectations & known limitations

These are documented in `architecture.md` §9:

- **Heuristic tier is a brittle baseline**: threshold + blob detection
  (~0.5 accuracy on realistic synthetic data). It exists to measure the
  learned tier's improvement.
- **The shipped learned model handles real photos**: on the held-out probes,
  `test2.jpg` → `AUTHENTIC confidence=0.153` (the synthetic-only model rated
  it 0.996). Dense foliage-type texture on `test.jpg` can still form a cluster
  that trips the ≥ 2-block rule.
- **Real CASIA v2 is wired but not yet won.** `--dataset-root` feeds real
  tampered data into training (masks auto-paired at ~97%, `--keep-synthetic`
  mixes pairs in), but measured on 113 held-out CASIA v2 images the
  12-feature RandomForest reaches only ~0.74 AUC and ~20% recall at the 0.5
  cluster threshold, and training on CASIA content re-flags the clean real
  photos. The limiting factor is tier capacity, not data availability;
  closing the gap needs richer features (e.g. ELA across the whole quality
  sweep) or a learned fingerprint.
- **Copy-move forgeries** defeat the noise branch by construction; ELA only
  helps when the moved patch has different compression history (SIFT/ORB is
  the right tool there).
- **Matching-quality recompression / GAN inpainting** wash out both classical
  signals; the field has moved to learned fingerprints for those.
- Published-grade numbers (85-92% on CASIA) require the real datasets and the
  harness; don't over-claim synthetic scores.

## Extending

See `architecture.md`: it holds the component/sequence diagrams, the
full workflow, and an extension-points table (swap denoiser, add ELA
qualities, swap RandomForest for SVM/CNN, add a SIFT copy-move track, wire
full PRNU with Dresden data). Keep `build_feature_maps()` as the single
feature source and retrain `models/block_rf.joblib` whenever features or the
training mix change.

Read `architecture.md` before extending anything; it pins the invariants the
pipeline depends on (two branches, one fusion; identical training/inference
features; the contiguous-cluster verdict rule; mixed-history synthetic
generator).