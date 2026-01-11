"""Minimal SEM template matching demo.

Usage:
  python demo_template_match.py \
    --image path/to/sem.png \
    --template-bbox x y w h \
    --output output.png

Optional:
  --template-path path/to/template.png
  --use-edges
  --scales 0.9,1.0,1.1
  --angles -5,0,5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass
class MatchResult:
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    scale: float
    angle: float


def parse_floats(csv: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in csv.split(",") if item.strip())


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess(image: np.ndarray, use_edges: bool) -> np.ndarray:
    gray = to_gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    eq = cv2.equalizeHist(blurred)
    if use_edges:
        return cv2.Canny(eq, 50, 150)
    return eq


def rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return template
    height, width = template.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(template, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated


def iter_scaled_templates(
    template: np.ndarray,
    scales: Iterable[float],
    angles: Iterable[float],
) -> Iterable[Tuple[np.ndarray, float, float]]:
    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            scaled = template
        else:
            scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for angle in angles:
            yield rotate_template(scaled, angle), scale, angle


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    scales: Iterable[float],
    angles: Iterable[float],
) -> MatchResult:
    best = MatchResult(score=-1.0, top_left=(0, 0), size=(0, 0), scale=1.0, angle=0.0)
    for candidate, scale, angle in iter_scaled_templates(template, scales, angles):
        if candidate.size == 0:
            continue
        h, w = candidate.shape[:2]
        if h > image.shape[0] or w > image.shape[1]:
            continue
        result = cv2.matchTemplate(image, candidate, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best.score:
            best = MatchResult(score=max_val, top_left=max_loc, size=(w, h), scale=scale, angle=angle)
    return best


def crop_template(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("Template bbox width/height must be positive.")
    return image[y : y + h, x : x + w]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEM template matching demo")
    parser.add_argument("--image", required=True, help="Path to SEM image")
    parser.add_argument("--template-path", help="Path to template image")
    parser.add_argument(
        "--template-bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Crop template from image using bbox",
    )
    parser.add_argument("--output", default="match_output.png", help="Output image path")
    parser.add_argument("--use-edges", action="store_true", help="Use Canny edges for matching")
    parser.add_argument("--scales", default="1.0", help="Comma-separated scale list")
    parser.add_argument("--angles", default="0", help="Comma-separated angle list in degrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.template_path:
        template_img = cv2.imread(args.template_path)
        if template_img is None:
            raise FileNotFoundError(f"Could not read template: {args.template_path}")
    elif args.template_bbox:
        template_img = crop_template(image, tuple(args.template_bbox))
    else:
        raise ValueError("Provide --template-path or --template-bbox.")

    image_proc = preprocess(image, args.use_edges)
    template_proc = preprocess(template_img, args.use_edges)

    scales = parse_floats(args.scales)
    angles = parse_floats(args.angles)
    if not scales:
        scales = (1.0,)
    if not angles:
        angles = (0.0,)

    result = match_template(image_proc, template_proc, scales, angles)

    output = image.copy()
    x, y = result.top_left
    w, h = result.size
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        output,
        f"score={result.score:.3f} scale={result.scale:.2f} angle={result.angle:.1f}",
        (max(0, x), max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.imwrite(args.output, output)
    print(
        "Match result:"
        f" score={result.score:.4f}, top_left={result.top_left}, size={result.size},"
        f" scale={result.scale}, angle={result.angle}"
    )


if __name__ == "__main__":
    main()
