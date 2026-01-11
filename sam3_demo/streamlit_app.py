"""Streamlit SAM3 demo for SEM image labeling.

Workflow:
- Upload an SEM image
- Pick a prompt type (point or box)
- Optionally choose a preset label
- Run SAM3 to get mask + overlay
- Download mask as JSON (RLE)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from sam3.build_sam3 import build_sam3
from sam3.sam3_predictor import Sam3Predictor


@dataclass
class Sam3Outputs:
    mask: np.ndarray
    score: float


def load_sam3(checkpoint: str, model_type: str, device: str) -> Sam3Predictor:
    try:
        model = build_sam3(model_type=model_type, checkpoint=checkpoint, device=device)
    except TypeError:
        model = build_sam3(model_type, checkpoint, device)
    return Sam3Predictor(model)


def pick_best_mask(masks: np.ndarray, scores: np.ndarray) -> Sam3Outputs:
    best_idx = int(np.argmax(scores))
    return Sam3Outputs(mask=masks[best_idx], score=float(scores[best_idx]))


def encode_rle(mask: np.ndarray) -> List[int]:
    flat = mask.flatten(order="F").astype(np.uint8)
    counts: List[int] = []
    last = 0
    run = 0
    for val in flat:
        if val != last:
            counts.append(run)
            run = 0
            last = val
        run += 1
    counts.append(run)
    return counts


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * color).astype(np.uint8)
    return overlay


def build_mask_json(
    mask: np.ndarray,
    score: float,
    label: str,
    prompt_type: str,
    prompt: Tuple[int, ...],
) -> Dict[str, Any]:
    height, width = mask.shape[:2]
    rle = encode_rle(mask.astype(np.uint8))
    return {
        "label": label,
        "score": score,
        "image": {"width": width, "height": height},
        "prompt": {"type": prompt_type, "coords": list(prompt)},
        "mask": {"encoding": "rle", "counts": rle},
    }


def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="SAM3 SEM Labeling Demo", layout="wide")
    st.title("SAM3 SEM Labeling Demo")

    with st.sidebar:
        checkpoint = st.text_input("SAM3 checkpoint path", value="/path/to/sam3_ckpt.pth")
        model_type = st.text_input("Model type", value="vit_h")
        device = st.text_input("Device", value="cuda")
        prompt_type = st.selectbox("Prompt type", options=["point", "box"])
        preset = st.selectbox(
            "Preset label",
            options=["marker", "保护垫", "ucut 凹槽", "SEM 电镜中的针", "custom"],
        )
        label = preset
        if preset == "custom":
            label = st.text_input("Custom label", value="object")

    uploaded = st.file_uploader("Upload SEM image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an SEM image to begin.")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Failed to read image.")
        return
    image_rgb = to_rgb(image_bgr)

    st.subheader("Select prompt")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        coords = streamlit_image_coordinates(image_rgb, key="image")

    prompt: Optional[Tuple[int, ...]] = None
    with col_right:
        if prompt_type == "point":
            if coords and "x" in coords and "y" in coords:
                prompt = (int(coords["x"]), int(coords["y"]))
                st.success(f"Point: {prompt}")
            else:
                st.caption("Click on the image to select a point.")
        else:
            st.caption("Enter box coordinates (x0, y0, x1, y1).")
            x0 = st.number_input("x0", min_value=0, value=0)
            y0 = st.number_input("y0", min_value=0, value=0)
            x1 = st.number_input("x1", min_value=0, value=100)
            y1 = st.number_input("y1", min_value=0, value=100)
            if x1 > x0 and y1 > y0:
                prompt = (int(x0), int(y0), int(x1), int(y1))

    run = st.button("Run SAM3")
    if not run:
        return

    if prompt is None:
        st.error("Prompt is not set. Provide a point or box.")
        return

    predictor = load_sam3(checkpoint, model_type, device)
    predictor.set_image(image_rgb)

    if prompt_type == "box":
        x0, y0, x1, y1 = prompt
        masks, scores, _ = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=True)
    else:
        x, y = prompt
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    result = pick_best_mask(masks, scores)
    mask = result.mask.astype(np.uint8)

    overlay = make_overlay(image_bgr, mask)
    st.subheader("Mask overlay")
    st.image(to_rgb(overlay), channels="RGB", use_container_width=True)

    payload = build_mask_json(mask, result.score, label, prompt_type, prompt)
    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    st.download_button(
        label="Download mask JSON",
        data=json_str,
        file_name="mask.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
