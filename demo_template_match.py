"""Minimal SEM template matching demo.

Usage:
  python demo_template_match.py \
    --image path/to/sem.png \
    --template-bbox x y w h \
    --output output.png

Optional:
  --template-path path/to/template.png
  --template-mask path/to/mask.png
  --template-polygon x1,y1 x2,y2 x3,y3 ...
  --use-edges
  --scales 0.9,1.0,1.1
  --angles -5,0,5
  --method ccorr|ccoeff|sqdiff
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MatchResult:
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    scale: float
    angle: float


@dataclass
class TemplateData:
    image: np.ndarray
    mask: Optional[np.ndarray]


def parse_floats(csv: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in csv.split(",") if item.strip())


def parse_polygon(points: List[str]) -> np.ndarray:
    coords = []
    for token in points:
        if "," not in token:
            raise ValueError("Polygon points must be provided as x,y pairs.")
        x_str, y_str = token.split(",", 1)
        coords.append([int(x_str), int(y_str)])
    if len(coords) < 3:
        raise ValueError("Polygon needs at least 3 points.")
    return np.array(coords, dtype=np.int32)


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


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return mask
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated


def resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return mask
    resized = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return resized


def iter_scaled_templates(
    template: np.ndarray,
    mask: Optional[np.ndarray],
    scales: Iterable[float],
    angles: Iterable[float],
) -> Iterable[Tuple[np.ndarray, Optional[np.ndarray], float, float]]:
    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            scaled = template
            scaled_mask = mask
        else:
            scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scaled_mask = resize_mask(mask, scale) if mask is not None else None
        for angle in angles:
            yield (
                rotate_template(scaled, angle),
                rotate_mask(scaled_mask, angle) if scaled_mask is not None else None,
                scale,
                angle,
            )


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    mask: Optional[np.ndarray],
    scales: Iterable[float],
    angles: Iterable[float],
    method: int,
) -> MatchResult:
    best = MatchResult(score=-1.0, top_left=(0, 0), size=(0, 0), scale=1.0, angle=0.0)
    for candidate, candidate_mask, scale, angle in iter_scaled_templates(
        template, mask, scales, angles
    ):
        if candidate.size == 0:
            continue
        h, w = candidate.shape[:2]
        if h > image.shape[0] or w > image.shape[1]:
            continue
        if candidate_mask is not None:
            result = cv2.matchTemplate(image, candidate, method, mask=candidate_mask)
        else:
            result = cv2.matchTemplate(image, candidate, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
            score = -min_val
            loc = min_loc
        else:
            score = max_val
            loc = max_loc
        if score > best.score:
            best = MatchResult(score=score, top_left=loc, size=(w, h), scale=scale, angle=angle)
    return best


def crop_template(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("Template bbox width/height must be positive.")
    return image[y : y + h, x : x + w]


def polygon_template(image: np.ndarray, polygon: np.ndarray) -> TemplateData:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    x, y, w, h = cv2.boundingRect(polygon)
    template = image[y : y + h, x : x + w]
    mask_crop = mask[y : y + h, x : x + w]
    return TemplateData(image=template, mask=mask_crop)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def resolve_template_data(args: argparse.Namespace, image: np.ndarray) -> TemplateData:
    if args.template_polygon:
        polygon = parse_polygon(args.template_polygon)
        return polygon_template(image, polygon)
    if args.template_path:
        template_img = cv2.imread(args.template_path)
        if template_img is None:
            raise FileNotFoundError(f"Could not read template: {args.template_path}")
        mask = None
        if args.template_mask:
            mask_img = cv2.imread(args.template_mask, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Could not read mask: {args.template_mask}")
            mask = normalize_mask(mask_img)
            if mask.shape != template_img.shape[:2]:
                raise ValueError("Template mask must match template size.")
        return TemplateData(image=template_img, mask=mask)
    if args.template_bbox:
        template_img = crop_template(image, tuple(args.template_bbox))
        mask = None
        if args.template_mask:
            mask_img = cv2.imread(args.template_mask, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Could not read mask: {args.template_mask}")
            if mask_img.shape == image.shape[:2]:
                x, y, w, h = args.template_bbox
                mask_img = mask_img[y : y + h, x : x + w]
            mask = normalize_mask(mask_img)
            if mask.shape != template_img.shape[:2]:
                raise ValueError("Template mask must match template crop size.")
        return TemplateData(image=template_img, mask=mask)
    raise ValueError("Provide --template-path, --template-bbox, or --template-polygon.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEM template matching demo")
    parser.add_argument("--image", required=True, help="Path to SEM image")
    parser.add_argument("--template-path", help="Path to template image")
    parser.add_argument("--template-mask", help="Path to template mask (same size as template)")
    parser.add_argument(
        "--template-bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Crop template from image using bbox",
    )
    parser.add_argument(
        "--template-polygon",
        nargs="+",
        help="Polygon points as x,y pairs (e.g. 10,20 30,40 50,60)",
    )
    parser.add_argument("--output", default="match_output.png", help="Output image path")
    parser.add_argument("--use-edges", action="store_true", help="Use Canny edges for matching")
    parser.add_argument("--scales", default="1.0", help="Comma-separated scale list")
    parser.add_argument("--angles", default="0", help="Comma-separated angle list in degrees")
    parser.add_argument(
        "--method",
        default="ccorr",
        choices=("ccorr", "ccoeff", "sqdiff"),
        help="Template matching method",
    )
    return parser.parse_args()


def resolve_method(method: str, has_mask: bool) -> int:
    if method == "ccorr":
        return cv2.TM_CCORR_NORMED
    if method == "sqdiff":
        return cv2.TM_SQDIFF_NORMED
    if has_mask:
        raise ValueError("Mask is only supported with ccorr/sqdiff methods in OpenCV.")
    return cv2.TM_CCOEFF_NORMED


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    template_data = resolve_template_data(args, image)

    image_proc = preprocess(image, args.use_edges)
    template_proc = preprocess(template_data.image, args.use_edges)

    scales = parse_floats(args.scales)
    angles = parse_floats(args.angles)
    if not scales:
        scales = (1.0,)
    if not angles:
        angles = (0.0,)

    method = resolve_method(args.method, template_data.mask is not None)
    result = match_template(
        image_proc,
        template_proc,
        template_data.mask,
        scales,
        angles,
        method,
    )

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
