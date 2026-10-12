"""
Local person instance segmentation for group extraction (YOLOv8-seg).

Bounding-box + rembg fails on crowded photos (neighbors leak in, rembg shreds
subjects). Segmentation masks isolate each person pixel-accurately.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_YOLO_MODEL = None
_YOLO_LOCK = threading.Lock()


def _model_name() -> str:
    # Prefer segmentation weights; fall back via env for older installs.
    return (
        (os.getenv("GROUP_EXTRACT_YOLO_MODEL") or "yolov8n-seg.pt").strip()
        or "yolov8n-seg.pt"
    )


def _conf_threshold() -> float:
    try:
        # Slightly lower default so partially-occluded kids/faces still register.
        return max(0.05, min(float(os.getenv("GROUP_EXTRACT_YOLO_CONF", "0.25")), 0.9))
    except ValueError:
        return 0.25


def _iou_threshold() -> float:
    try:
        return max(0.1, min(float(os.getenv("GROUP_EXTRACT_YOLO_IOU", "0.5")), 0.9))
    except ValueError:
        return 0.5


def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is not None:
            return _YOLO_MODEL
        os.environ.setdefault("YOLO_CONFIG_DIR", os.getenv("YOLO_CONFIG_DIR", "/tmp/Ultralytics"))
        from ultralytics import YOLO

        model_name = _model_name()
        print(f"🔧 Loading YOLO person segmenter model={model_name}")
        _YOLO_MODEL = YOLO(model_name)
        return _YOLO_MODEL


def warmup_person_detector() -> None:
    """Optional startup warmup so the first group extract is not a cold download."""
    try:
        _get_yolo_model()
        print("✅ YOLO person segmenter ready")
    except Exception as exc:
        print(f"⚠️ YOLO person detector warmup skipped: {exc}")


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _sort_reading_order(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Front row left→right, then rows behind (bottom-to-top of the frame)."""
    if not detections:
        return []

    enriched = []
    for det in detections:
        left, top, right, bottom = det["bbox"]
        enriched.append({
            **det,
            "_yc": (top + bottom) * 0.5,
            "_bottom": bottom,
            "_left": left,
        })

    heights = [d["bbox"][3] - d["bbox"][1] for d in enriched]
    median_h = sorted(heights)[len(heights) // 2] if heights else 0.2
    row_thresh = max(0.06, min(0.14, median_h * 0.55))

    enriched.sort(key=lambda d: d["_bottom"], reverse=True)
    rows: List[List[Dict[str, Any]]] = []
    for det in enriched:
        placed = False
        for row in rows:
            row_yc = sum(item["_yc"] for item in row) / len(row)
            if abs(det["_yc"] - row_yc) <= row_thresh:
                row.append(det)
                placed = True
                break
        if not placed:
            rows.append([det])

    ordered: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda d: d["_left"])
        for det in row:
            clean = {k: v for k, v in det.items() if not k.startswith("_")}
            ordered.append(clean)
    return ordered


def _rgba_on_white(cut: Image.Image) -> Image.Image:
    cut = cut.convert("RGBA")
    canvas = Image.new("RGB", cut.size, (255, 255, 255))
    canvas.paste(cut, mask=cut.split()[3])
    return canvas


def _largest_component(mask_bool, min_area: int = 64):
    """Keep the largest connected True region; drop stray fragments."""
    import numpy as np

    if mask_bool is None or not bool(mask_bool.any()):
        return mask_bool
    try:
        from scipy import ndimage
    except Exception:
        return mask_bool

    labeled, num = ndimage.label(mask_bool)
    if num <= 1:
        return mask_bool

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    best = int(counts.argmax())
    if counts[best] < min_area:
        return mask_bool
    return labeled == best


def _mask_bbox_and_fill(mask_bool) -> Tuple[Optional[Tuple[int, int, int, int]], float, int]:
    """Return (x0,y0,x1,y1), fill_ratio inside that bbox, and pixel count."""
    import numpy as np

    if mask_bool is None or not bool(mask_bool.any()):
        return None, 0.0, 0
    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    area = max(1, (y1 - y0) * (x1 - x0))
    count = int(mask_bool.sum())
    return (x0, y0, x1, y1), count / float(area), count


def _head_band_mask(mask_bool, head_frac: float = 0.48):
    """
    Restrict a person mask to the upper head/shoulders band.

    Uses the mask's own vertical extent so short/tall subjects both work.
    """
    import numpy as np

    if mask_bool is None or not bool(mask_bool.any()):
        return mask_bool
    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = max(1, y1 - y0)
    cut_y = y0 + int(span * max(0.28, min(0.65, head_frac)))
    head = mask_bool.copy()
    head[cut_y:, :] = False
    # Widen slightly downward for chin/neck once we know the head mass.
    if bool(head.any()):
        hys = np.where(head)[0]
        neck = min(mask_bool.shape[0], int(hys.max()) + max(4, int(span * 0.12)))
        band = mask_bool.copy()
        band[: int(hys.min()), :] = False
        band[neck:, :] = False
        head = band
    return _largest_component(head)


def _should_prefer_face_crop(
    original_mask,
    exclusive_mask,
) -> bool:
    """
    True when the body is mostly occluded / swiss-cheesed by neighbors.

    Art generation needs a clear face more than a hollow body silhouette.
    """
    import numpy as np

    orig = int(original_mask.sum()) if original_mask is not None else 0
    excl = int(exclusive_mask.sum()) if exclusive_mask is not None else 0
    if orig <= 0 or excl <= 0:
        return True

    retention = excl / float(orig)
    main = _largest_component(exclusive_mask)
    _, fill, main_count = _mask_bbox_and_fill(main)
    # Fragments left after overlap resolution: low retention or sparse bbox fill.
    if retention < 0.68:
        return True
    if fill < 0.45:
        return True
    # Side-bitten masks (neighbor silhouette) — face is safer for art.
    if main is not None and bool(main.any()):
        import numpy as np

        ys, xs = np.where(main)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        band = main[y0:y1, x0:x1]
        if band.shape[1] >= 12:
            left_f = float(band[:, : band.shape[1] // 2].mean())
            right_f = float(band[:, band.shape[1] // 2 :].mean())
            if min(left_f, right_f) < 0.42 and abs(left_f - right_f) > 0.16:
                return True
    # Multiple large components → body punched through; face is safer.
    try:
        from scipy import ndimage

        labeled, num = ndimage.label(exclusive_mask)
        if num >= 2:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            top = sorted(counts.tolist(), reverse=True)
            if len(top) >= 2 and top[1] >= max(80, int(main_count * 0.18)):
                return True
    except Exception:
        pass
    return False


def _face_mask_for_person(
    original_mask,
    exclusive_mask,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
):
    """
    Build a head/face mask when the full body cutout would be incomplete.

    Prefers the original head band for a complete face (overlap resolution often
    shreds cheeks/hair). Falls back to exclusive head, then geometric top box.
    """
    import numpy as np

    min_px = max(48, int(width * height * 0.0004))
    head_orig = _head_band_mask(original_mask, head_frac=0.52)
    head_excl = _head_band_mask(exclusive_mask, head_frac=0.52)

    # Clear faces matter more than perfect neighbor exclusion in the head band.
    if head_orig is not None and int(head_orig.sum()) >= min_px:
        if head_excl is not None and int(head_excl.sum()) >= min_px:
            # If exclusive kept most of the head, use it (cleaner edges).
            if int(head_excl.sum()) >= int(head_orig.sum() * 0.72):
                return head_excl
        return _largest_component(head_orig)

    if head_excl is not None and int(head_excl.sum()) >= min_px:
        return head_excl

    left, top, right, bottom = bbox
    x0 = max(0, int(left * width))
    x1 = min(width, max(x0 + 1, int(right * width)))
    y0 = max(0, int(top * height))
    y1 = min(height, max(y0 + 1, int(bottom * height)))
    head_y1 = min(y1, y0 + max(8, int((y1 - y0) * 0.48)))
    geom = np.zeros((height, width), dtype=bool)
    geom[y0:head_y1, x0:x1] = True
    if original_mask is not None:
        hit = geom & original_mask
        if int(hit.sum()) >= min_px:
            return _largest_component(hit)
    if exclusive_mask is not None:
        hit = geom & exclusive_mask
        if int(hit.sum()) >= min_px:
            return _largest_component(hit)
    return geom


def _mask_to_cutout(
    rgb_arr,
    mask_bool,
    pad_px: int = 8,
    *,
    portrait_square: bool = False,
) -> Optional[Image.Image]:
    """Crop the masked person and composite onto white."""
    import numpy as np

    if mask_bool is None or not bool(mask_bool.any()):
        return None

    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = mask_bool.shape[:2]

    if portrait_square:
        # Pad into a slightly tall portrait around the head/face mass.
        bw, bh = x1 - x0, y1 - y0
        side = int(max(bw, bh) * 1.35)
        side = max(side, int(max(bw, bh) + 8))
        cx = (x0 + x1) // 2
        # Bias upward so forehead/hair stay in frame.
        cy = y0 + int(bh * 0.42)
        x0 = cx - side // 2
        x1 = x0 + side
        y0 = cy - int(side * 0.45)
        y1 = y0 + side

    y0 = max(0, y0 - pad_px)
    x0 = max(0, x0 - pad_px)
    y1 = min(h, y1 + pad_px)
    x1 = min(w, x1 + pad_px)
    if x1 <= x0 or y1 <= y0:
        return None

    alpha_mask = mask_bool[y0:y1, x0:x1]
    if portrait_square and not bool(alpha_mask.any()):
        return None
    alpha = (alpha_mask.astype(np.uint8) * 255)
    rgb_crop = rgb_arr[y0:y1, x0:x1]
    rgba = np.dstack([rgb_crop, alpha])
    return _rgba_on_white(Image.fromarray(rgba, "RGBA"))


def extract_persons_yolo_seg(
    image: Image.Image,
    max_persons: int = 12,
) -> List[Dict[str, Any]]:
    """
    Instance-segment people and return reading-order cutouts on white.

    Each item:
      {
        bbox: (l,t,r,b) fractions,
        confidence: float,
        description: str,
        cutout: PIL.Image RGB on white,
      }
    """
    if image is None:
        return []

    import numpy as np

    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return []

    model = _get_yolo_model()
    conf = _conf_threshold()
    iou = _iou_threshold()
    arr = np.asarray(rgb)

    results = model.predict(
        source=arr,
        conf=conf,
        iou=iou,
        classes=[0],  # COCO person
        verbose=False,
        max_det=max(max_persons * 2, 24),
        retina_masks=True,
    )
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy
    confs = boxes.conf
    try:
        xyxy_list = xyxy.tolist()
        conf_list = confs.tolist()
    except Exception:
        xyxy_list = [row.tolist() for row in xyxy]
        conf_list = [float(c) for c in confs]

    raw_masks = None
    if masks is not None and getattr(masks, "data", None) is not None:
        try:
            raw_masks = masks.data.cpu().numpy()
        except Exception:
            try:
                raw_masks = np.asarray(masks.data)
            except Exception:
                raw_masks = None

    if raw_masks is None or len(raw_masks) == 0:
        print("⚠️ YOLO-seg returned boxes without masks; cannot isolate people cleanly")
        return []

    # Resize masks to full image and collect candidates.
    candidates: List[Dict[str, Any]] = []
    for idx, (coords, score) in enumerate(zip(xyxy_list, conf_list)):
        if idx >= len(raw_masks):
            break
        x1, y1, x2, y2 = coords
        left = max(0.0, min(float(x1) / width, 1.0))
        top = max(0.0, min(float(y1) / height, 1.0))
        right = max(0.0, min(float(x2) / width, 1.0))
        bottom = max(0.0, min(float(y2) / height, 1.0))
        bw = right - left
        bh = bottom - top
        if bw < 0.02 or bh < 0.04 or bw * bh > 0.85:
            continue

        mask = raw_masks[idx]
        if mask.shape[0] != height or mask.shape[1] != width:
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((width, height), Image.Resampling.BILINEAR)
            mask_bool = np.asarray(mask_img) >= 128
        else:
            mask_bool = mask > 0.5

        if int(mask_bool.sum()) < max(64, int(width * height * 0.0015)):
            continue

        candidates.append({
            "bbox": (left, top, right, bottom),
            "confidence": float(score),
            "mask": mask_bool,
            "description": "",
        })

    if not candidates:
        return []

    # Drop near-duplicate boxes (keep higher confidence).
    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        if all(_box_iou(cand["bbox"], item["bbox"]) < max(iou, 0.55) for item in kept):
            kept.append(cand)
    candidates = kept

    if len(candidates) > max_persons:
        candidates = candidates[:max_persons]

    # Resolve mask overlaps: each pixel belongs to the highest-confidence person.
    owner = np.full((height, width), -1, dtype=np.int32)
    owner_conf = np.zeros((height, width), dtype=np.float32)
    for i, cand in enumerate(candidates):
        mask = cand["mask"]
        better = mask & (cand["confidence"] > owner_conf)
        owner[better] = i
        owner_conf[better] = cand["confidence"]

    detections: List[Dict[str, Any]] = []
    face_crop_count = 0
    for i, cand in enumerate(candidates):
        exclusive = owner == i
        # Keep a thin halo from the original mask so edges aren't too harsh,
        # but never steal pixels already owned by a higher-confidence person.
        halo = cand["mask"] & (owner < 0)
        final_mask = exclusive | halo
        # Drop tiny leftover islands that inflate the crop into a white void.
        final_mask = _largest_component(final_mask)

        crop_kind = "body"
        if _should_prefer_face_crop(cand["mask"], final_mask):
            face_mask = _face_mask_for_person(
                original_mask=cand["mask"],
                exclusive_mask=final_mask,
                bbox=cand["bbox"],
                width=width,
                height=height,
            )
            if face_mask is not None and bool(face_mask.any()):
                final_mask = face_mask
                crop_kind = "face"
                face_crop_count += 1

        cutout = _mask_to_cutout(
            arr,
            final_mask,
            pad_px=10 if crop_kind == "face" else 8,
            portrait_square=(crop_kind == "face"),
        )
        if cutout is None:
            continue
        # Reject empty / near-empty results.
        if min(cutout.size) < 24:
            continue
        detections.append({
            "bbox": cand["bbox"],
            "confidence": cand["confidence"],
            "cutout": cutout,
            "crop_kind": crop_kind,
            "description": "",
        })

    detections = _sort_reading_order(detections)
    for idx, det in enumerate(detections, start=1):
        left, top, right, bottom = det["bbox"]
        kind = det.get("crop_kind") or "body"
        det["description"] = (
            f"person #{idx} (YOLO-seg conf={det['confidence']:.2f}, "
            f"{kind}, box={left:.0%},{top:.0%},{right:.0%},{bottom:.0%})"
        )
    print(
        f"🟦 YOLO-seg isolated {len(detections)} person cutout(s) "
        f"({face_crop_count} face-focused for occlusion)"
    )
    return detections


def detect_persons_yolo(
    image: Image.Image,
    max_persons: int = 12,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible detector API used by planning.

    Prefer segmentation cutouts; if unavailable, return bbox-only plans.
    """
    seg = extract_persons_yolo_seg(image, max_persons=max_persons)
    if seg:
        return [
            {
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "description": item["description"],
                "cutout": item.get("cutout"),
            }
            for item in seg
        ]

    # Last-resort bbox-only path (no masks) — caller should treat carefully.
    print("⚠️ Falling back to YOLO boxes without segmentation masks")
    from ultralytics import YOLO  # noqa: F401 — model already loaded

    import numpy as np

    rgb = image.convert("RGB")
    width, height = rgb.size
    model = _get_yolo_model()
    results = model.predict(
        source=np.asarray(rgb),
        conf=_conf_threshold(),
        iou=_iou_threshold(),
        classes=[0],
        verbose=False,
        max_det=max(max_persons * 2, 20),
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    detections: List[Dict[str, Any]] = []
    for coords, score in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.conf.tolist()):
        x1, y1, x2, y2 = coords
        left = max(0.0, min(float(x1) / width, 1.0))
        top = max(0.0, min(float(y1) / height, 1.0))
        right = max(0.0, min(float(x2) / width, 1.0))
        bottom = max(0.0, min(float(y2) / height, 1.0))
        if right - left < 0.02 or bottom - top < 0.04:
            continue
        detections.append({
            "bbox": (left, top, right, bottom),
            "confidence": float(score),
            "description": "",
            "cutout": None,
        })
    detections = _sort_reading_order(detections)[:max_persons]
    for idx, det in enumerate(detections, start=1):
        left, top, right, bottom = det["bbox"]
        det["description"] = (
            f"person #{idx} (YOLO conf={det['confidence']:.2f}, "
            f"box={left:.0%},{top:.0%},{right:.0%},{bottom:.0%})"
        )
    return detections
