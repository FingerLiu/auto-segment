# auto-segment

Minimal demo scripts for SEM image segmentation and template matching.

## Scripts

### SAM3 auto-labeling (mask + template crop)

This script runs SAM3 with a box or point prompt and exports:
- a binary mask
- a tight template crop (bounding box of the mask)

```bash
python sam3_demo/sam3_auto_label.py \
  --image path/to/sem.png \
  --checkpoint /path/to/sam3_ckpt.pth \
  --model-type vit_h \
  --box 120 80 360 240 \
  --mask-out outputs/mask.png \
  --template-out outputs/template.png
```

Point prompt example:

```bash
python sam3_demo/sam3_auto_label.py \
  --image path/to/sem.png \
  --checkpoint /path/to/sam3_ckpt.pth \
  --point 200 150 \
  --mask-out outputs/mask.png \
  --template-out outputs/template.png
```

### Template matching demo (rectangle/polygon/mask)

This script matches a template in an SEM image using NCC/CCORR, optionally
with edge preprocessing and multi-scale/multi-angle sweeps.

#### Rectangle template (crop from image)

```bash
python demo_template_match.py \
  --image path/to/sem.png \
  --template-bbox 120 80 240 160 \
  --output outputs/match.png
```

#### Polygon template (arbitrary shape)

Provide polygon points as a comma-separated list of `x,y` pairs. The script
will build a polygon mask, crop the bounding box, and use masked matching.

```bash
python demo_template_match.py \
  --image path/to/sem.png \
  --template-polygon 120,80 360,80 340,200 140,220 \
  --output outputs/match.png
```

#### Template + mask from SAM3

Use the SAM3 mask as a template mask for masked matching:

```bash
python demo_template_match.py \
  --image path/to/sem.png \
  --template-path outputs/template.png \
  --template-mask outputs/mask.png \
  --output outputs/match.png
```

#### Optional flags

- `--use-edges`: apply Canny edge preprocessing.
- `--scales 0.9,1.0,1.1`: multi-scale search.
- `--angles -5,0,5`: multi-angle search.
- `--method ccorr`: use masked CCORR (default) or `ccoeff` for classic NCC.

