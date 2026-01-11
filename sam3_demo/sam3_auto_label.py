"""SAM3-assisted auto-labeling demo for SEM images.

This script uses SAM3 to generate a mask from a prompt (box or point), then
exports the binary mask and a tight crop that can be used as a template for
classical matching.

Example:
  python sam3_auto_label.py \
    --image data/sem.png \
    --checkpoint /path/to/sam3_ckpt.pth \
    --model-type vit_h \
    --box 120 80 360 240 \
    --mask-out outputs/mask.png \
    --template-out outputs/template.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Sam3Outputs:
    mask: np.ndarray
    score: float


def load_sam3(checkpoint: str, model_type: str, device: str):
    try:
        from sam3.build_sam3 import build_sam3
        from sam3.sam3_predictor import Sam3Predictor
    except ImportError as exc:
        raise ImportError(
            "SAM3 is not installed. Follow https://github.com/facebookresearch/sam3 to install."
        ) from exc

    try:
        model = build_sam3(model_type=model_type, checkpoint=checkpoint, device=device)
    except TypeError:
        try:
            model = build_sam3(model_type, checkpoint, device)
        except TypeError as exc:
            raise TypeError(
                "Unable to build SAM3 model with provided arguments. "
                "Check SAM3 build_sam3 signature."
            ) from exc
    return Sam3Predictor(model)


def pick_best_mask(masks: np.ndarray, scores: np.ndarray) -> Sam3Outputs:
    if masks.ndim != 3:
        raise ValueError("Expected masks with shape [N, H, W].")
    best_idx = int(np.argmax(scores))
    return Sam3Outputs(mask=masks[best_idx], score=float(scores[best_idx]))


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 auto-labeling for SEM images")
    parser.add_argument("--image", required=True, help="Path to SEM image")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM3 checkpoint")
    parser.add_argument("--model-type", default="vit_h", help="SAM3 model type")
    parser.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    parser.add_argument(
        "--box",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Prompt box in pixel coords (x0 y0 x1 y1)",
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Prompt point (x y)",
    )
    parser.add_argument("--mask-out", required=True, help="Output mask path")
    parser.add_argument("--template-out", required=True, help="Output template crop path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.box is None and args.point is None:
        raise ValueError("Provide --box or --point as the SAM3 prompt.")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    predictor = load_sam3(args.checkpoint, args.model_type, args.device)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    masks = None
    scores = None
    if args.box:
        x0, y0, x1, y1 = args.box
        box = np.array([x0, y0, x1, y1])
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    else:
        x, y = args.point
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    result = pick_best_mask(masks, scores)
    mask = (result.mask.astype(np.uint8) * 255)

    mask_path = Path(args.mask_out)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)

    bbox = mask_to_bbox(result.mask)
    if bbox is None:
        raise RuntimeError("SAM3 returned an empty mask.")
    x, y, w, h = bbox
    template = image[y : y + h, x : x + w]

    template_path = Path(args.template_out)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(template_path), template)

    print(
        "SAM3 label result:"
        f" score={result.score:.4f}, bbox=({x}, {y}, {w}, {h}),"
        f" mask_out={mask_path}, template_out={template_path}"
    )


if __name__ == "__main__":
    main()
