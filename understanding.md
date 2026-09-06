# Understanding IFD (Image Forgery Detector)

A from-zero walkthrough of what this project does and how it works. Every
technical term is defined inline, so no prior knowledge is assumed.

## 1. What this project does

You give it **one image** and it answers two questions:

1. **Has this image been doctored?** (verdict: AUTHENTIC or TAMPERED)
2. **If yes, where?** (a heatmap, a mask, and bounding boxes showing the
   tampered region)

It does this with **no other information**: no original photo, no camera, no
metadata. Just the pixels.

## 2. A crash course on digital images

A digital image is a grid of tiny colored squares called **pixels**. Each
pixel's color is stored as **three numbers**: Red, Green, Blue (the **RGB**
channels). `(0,0,0)` is black, `(255,255,255)` is white. Height x width x 3
numbers is the raw image.

Raw images are huge, so cameras **compress** them. **JPEG** is the most common
compressed image format. It is *lossy*: it throws away fine detail the human
eye barely notices, shrinking a multi-megabyte photo into a few hundred
kilobytes. The compression process is called **encoding**; the dial that
controls how much detail is discarded is the **quality** (90 keeps a lot, 50
discards a lot). The little imperfections a lossy encoder introduces are
**compression artifacts**.

The key phrase in this repo is **compression history**: the story of how a
photo was saved. Every JPEG encode stamps its own artifacts onto the pixels.

## 3. Why tampering is detectable at all

A real, untouched photo has **one uniform compression history**: it was shot
and saved by essentially one process, so its artifacts are consistent across
the whole image.

When a region is spliced in (formal word for copy-paste) from another image,
or edited and re-saved separately, that region now has a **different
compression history**. It was compressed at a different time, maybe at a
different quality, maybe by a different program.

The whole trick: **detect regions whose history does not match the rest of the
image.** Two independent methods ("branches") look for that mismatch from
opposite directions, and because each can fail in its own way, the project
fuses them.

## 4. Branch 1: Error Level Analysis (ELA)

**ELA** answers: *"If I save this image again, does the whole image change
equally, or is part of it extra sensitive?"*

Intuition. Take a JPEG that was saved at quality 90 and re-save it at 90. A
JPEG re-encoded at its own quality is a **fixpoint**: the encoder is basically
re-encoding its own output, so the difference is tiny everywhere.

Now splice in a patch that was originally saved at quality 50. That patch is
*not* at the 90-fixpoint; it still carries heavy, coarse 50-level artifacts.
Re-saving the whole photo at 90 squeezes that patch much harder, so the
before/after difference near the patch is large.

The algorithm (`ifd/features/ela.py`):

1. Re-save the image at a known quality (85 by default).
2. Subtract the new version from the old, pixel by pixel.
3. Amplify the difference.

Big difference = mismatched history = suspicious. The result is a **heatmap**:
an image-shaped grid of values, colored so high values glow.

```
suspect image -> save at 85 -> compare to original -> amplify -> ELA heatmap
```

Weakness: text and high-contrast edges also look "different after re-saving"
even on genuine photos, so ELA alone produces many **false positives**
(flagging a real image as tampered). It cries wolf.

## 5. Branch 2: Blind noise-residual forensics

Every real photo carries invisible **sensor noise**: tiny random bright/dark
speckles from the physics of the camera's light sensor. Every camera device
and every scene has a slightly different noise texture.

Two images from different sources (or a patch pasted in later) have
**different noise textures**. The idea: if a region's noise looks different
from its neighborhood's, it may be spliced.

"**Blind**" means we have no reference fingerprint of the camera's true noise;
we only compare the suspect image against itself, region to region.

The algorithm (`ifd/features/noise.py`):

1. **Denoise**: make a guess at what the photo looks like without noise (via
   wavelet decomposition: breaking the image into detail layers at different
   sizes and smoothing them).
2. **Residual**: subtract the denoised version from the real one. What remains
   is mostly the noise.
3. **Local variance**: measure how much the leftover noise jitters in each
   local window. High jitter = noisier = possibly a mismatched or synthetic
   patch.

Output: a noise heatmap, high where suspicious.

Weakness: fine details (foliage, fabric, hair) naturally have high noise
variance even when genuine, so this branch also produces false positives. And
**copy-move** (copying a piece of the same image into itself) keeps the noise
identical, so this branch cannot see it at all.

## 6. Fusion tier 1: the heuristic baseline

The simplest fusion combines both heatmaps with plain **heuristics**: hand-
written rules of thumb rather than anything learned from data
(`ifd/fusion/__init__.py`, `heuristic_fusion`).

Steps:

1. **Normalize** both maps to a 0-1 range (percentile clipping: squish the
   middle 99% of values into 0-1 so one exceptional pixel cannot dominate).
2. **Threshold**: mark a pixel "flagged" if its heat value exceeds a cutoff.
3. **Connected components**: group touching flagged pixels into blobs.
   Isolated specks are noise; keep only blobs above a minimum area (25 px).
   This is the "islands on a map" idea, adjacent flagged pixels form an
   island.
4. **Combine**: "union" (ELA *or* noise flags the pixel) or "intersection"
   (both must).
5. **Verdict**: if the flagged fraction reaches about 1% of the image, call it
   TAMPERED.

This tier is intentionally **brittle**: it scores around 50% accuracy on
realistic tests. It exists to give the smarter tier below something honest to
beat, and both numbers are always reported.

## 7. Tier 2: the learned classifier

Instead of hand-written thresholds, this tier uses **machine learning**: a
statistical model that has seen labeled examples and learned its own rules.

### Blocks and features

It does not judge single pixels. It slides **blocks** of 64x64 pixels across
the image, and for each block computes **12 numbers ("features")**: four
statistics (`mean`, `standard deviation`, `95th percentile`, `maximum`) on
each of three maps (the ELA map, the noise map, and the grayscale/brightness
map). Standard deviation measures how spread-out and wiggly a block is, i.e.
its **texture**. These 12 numbers summarize what the block looks like.

### RandomForest

The model is a **RandomForestClassifier** (scikit-learn).

- A **decision tree** is a flow chart of questions learned from data: "is the
  ELA mean above 0.3? then check feature 2...", eventually landing on a score.
- A **random forest** is many such trees, each trained on a random subset of
  the data and the features; the trees **vote**. The forest output is the
  average vote: a **probability** between 0 and 1 that a block is tampered.

Why a forest? A single tree overfits (memorizes the training data); averaging
many trees makes predictions more robust and general.

### Training data

The forest must learn "tampered blocks look like *this*, genuine blocks look
like *that*". That needs **labeled examples**: images where the truth is known,
including a **mask** (the exact tampered pixels). Training data comes from:

- **Synthetic forgeries** (`scripts/make_synthetic.py`): generated photos with
  a built-in, detectable history: host saved at quality 90, donor patch saved
  at 50, pasted in, whole composite re-saved at 90. This mixed-history recipe
  is essential. Pasting raw pixels and saving once creates exactly **one**
  history and an invisible ELA signal, so training on it would teach nothing.
- **Real authentic photos** (`data/authentic/`): genuine photos labeled "no
  tampering". These stop the model from learning the shortcut "high texture =
  tampered", which a synthetic-only model picks up and which wrongly flags any
  real photo.

### Inference

1. Compute the three maps, chop into blocks, get 12 features per block.
2. Ask the forest for a tamper probability per block.
3. Fold the probabilities back into a full-size **heatmap**.
4. **Cluster rule** (`ifd/detector.py`, `_detect_learned`): keep blocks with
   probability >= 0.5, then keep only **connected groups of at least two
   blocks** whose peak probability >= 0.5. A real patch forms a contiguous
   blob of confident blocks; isolated single blocks are usually texture
   flukes. This is a deliberate invariant: reverting to per-block thresholds
   brings back the texture false-positive mode.
5. **Confidence** (the final 0-1 number) blends the strongest block's
   probability with how much area was flagged.

Output: a **DetectionResult** with `is_tampered` (bool), `confidence`,
`heatmap` (float), `mask` (hard 0/255), `bboxes` (min/max row and column of
each flagged blob), and `feature_maps`.

## 8. The single canonical extractor

Training and detection both call `build_feature_maps()` in `ifd/detector.py`,
one shared function that computes `{ela, noise, gray}`. This is a hard rule:
the model learns what blocks look like in these exact map definitions. Change
the denoiser, block size, or ELA quality sweep, and you must **retrain**
(`scripts/train_blocks.py`); otherwise the model silently sees features unlike
anything it was trained on.

Training also picks a **random ELA quality per image** from {70, 85, 95},
while inference always uses 85. That teaches the model to work when the real-
world provenance of a suspect photo is unknown, i.e. you do not know what
quality the forger used.

## 9. Evaluation

`ifd/eval/metrics.py` scores the system **per dataset**, never as one blended
number:

- **Image-level**: accuracy (how many verdicts were right), precision (of the
  images called tampered, how many truly were), recall (of all real forgeries,
  how many were caught), F1 (a balanced blend of precision and recall),
  ROC-AUC (a general 0-1 "how well does it separate the classes" score; 1.0 is
  perfect).
- **Pixel-level** (where masks exist): IoU (mask/truth overlap divided by
  their union; 1.0 = perfect localization), Dice, and pixel-level
  precision/recall/F1.

## 10. What is CASIA?

**CASIA** is the Chinese Academy of Sciences, Institute of Automation, and the
name of the most widely used public benchmark dataset for image-forgery
detection.

Two collections matter:

- **CASIA v1**: smaller (~1,700 images), used as an out-of-distribution test.
  Name prefixes `Tp_*` / `Sp_*` mean tampered; it has **no ground-truth
  masks**.
- **CASIA v2**: the main benchmark (~12,000 images). `Au_*` = authentic,
  `Tp_*` / `Sp_*` = tampered (splicing, copy-move, retouching). Tampered
  images can be paired with `_gt.png` **masks** from the community repo
  `namtpham/casia2groundtruth`.

CASIA is what makes real-world evaluation possible, because the truth is known
per image. `ifd/data/casia.py` indexes the tree recursively, pairs tampered
images with their masks, and skips known-broken files (resolution mismatches,
one missing mask). A tampered image without a mask loads fine but with no
forged blocks labeled, i.e. it contributes nothing positive to training.

## 11. How to add CASIA v2 tampered images to training

1. **Get the images**: the CASIA v2 zip from its official academic source.
   Any tree of `Au_*` / `Tp_*` files works, layout is irrelevant.
2. **Get the masks**: from `namtpham/casia2groundtruth`. Place each as
   `<image-name>_gt.png` next to its image (or in a sibling `groundtruth/`
   folder).
3. **Train on the mix** (CASIA replaces the synthetic set; pass
   `--keep-synthetic` to mix pairs in, and real authentic photos still get
   added on top):

```powershell
python scripts/train_blocks.py --dataset-root path/to/CASIA2 --n-train 14 --n-test 8 --keep-synthetic --authentic-root data/authentic --real-block-budget 2500 --out models
```

Verify success from the log: `Trained on N blocks (M forged-labeled)` must
show **M > 0** (proof the masks were found). Then re-check the held-out
probes (`test.jpg`, `test2.jpg`) and run a dataset report:

```powershell
python scripts/demo.py --image test2.jpg --model models\block_rf.joblib
python scripts/demo.py --casia path/to/CASIA2 --max-images 200 --model models\block_rf.joblib
```

Measured outcome, kept honest on purpose: the loader pairs about 97% of masks
(the official `CASIA 2 Groundtruth` folder and `_gtN` filename variants are
handled), but with the current 12-feature RandomForest, training on CASIA
reaches only ~0.74 AUC on a held-out CASIA set and ~20% recall at the 0.5
cluster threshold, and it re-flags the clean real photos. The wiring works;
the tier capacity is the limit. See README "Honest expectations" for the
numbers.

## 12. Honest limitations

- **False positives** (called tampered, actually real) are the main failure
  mode. The heuristic tier has many; the learned tier fixed the worst, but
  dense foliage can still trip the 2-block cluster rule.
- **Copy-move** (patch from the same image) defeats the sensor-noise branch;
  ELA only helps if the copied patch's compression history differs.
- **Matching-quality recompression** washes out ELA; anti-forensics
  (recompression, added noise, GAN inpainting) defeats both classical signals.
  The field is moving toward neural fingerprints.
- **Synthetic training is not real data**: the model is largely trained on
  generated photos, which is why `data/authentic/` real photos and the
  `test.jpg`/`test2.jpg` probes exist. Real CASIA v2 tampered data is wired
  in, but measured results show the 12-feature tier cannot match both goals
  (clean real photos and real CASIA recall); closing the gap needs richer
  features or a learned fingerprint.

## 13. See it live

Run the smoke test (the pass/fail contract), then open a single-image report:

```powershell
python scripts/smoke_test.py
python scripts/demo.py --image some.jpg --model models/block_rf.joblib
```

The HTML report shows the original, the ELA heatmap, the noise heatmap, the
fused heat, and the overlay mask side by side, which makes the whole
"mismatched-history region glows differently" story visually obvious.