from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - allows running logic tests without OpenCV installed
    cv2 = None


@dataclass(frozen=True)
class CoinSpec:
    denomination: int
    series: str
    diameter_mm: float


SRI_LANKAN_COIN_SPECS: List[CoinSpec] = [
    CoinSpec(1, "new", 20.0),
    CoinSpec(1, "old", 22.0),
    CoinSpec(2, "new", 22.0),
    CoinSpec(2, "old", 28.0),
    CoinSpec(5, "new", 23.5),
    CoinSpec(5, "old", 24.0),
    CoinSpec(10, "new", 26.4),
    CoinSpec(10, "old", 28.5),
]


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image and video processing. Install opencv-python.")


def actual_radius_mm(radius_pixels: float, camera_distance_mm: float, focal_length_px: float) -> float:
    return (radius_pixels * camera_distance_mm) / focal_length_px


def estimate_camera_distance_mm(
    measured_radius_pixels: float, known_radius_mm: float, focal_length_px: float
) -> float:
    return (known_radius_mm * focal_length_px) / measured_radius_pixels


def classify_by_diameter(diameter_mm: float, tolerance_mm: float = 1.25) -> Optional[CoinSpec]:
    best_match = min(SRI_LANKAN_COIN_SPECS, key=lambda c: abs(c.diameter_mm - diameter_mm))
    if abs(best_match.diameter_mm - diameter_mm) > tolerance_mm:
        return None
    return best_match


def detect_coin_circles(frame: np.ndarray) -> np.ndarray:
    require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=150,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0]


def analyze_frame(
    frame: np.ndarray,
    focal_length_px: float,
    camera_distance_mm: Optional[float] = None,
    reference_diameter_mm: Optional[float] = None,
) -> Tuple[List[Dict[str, object]], float, Dict[int, int], int]:
    circles = detect_coin_circles(frame)

    if len(circles) == 0:
        return [], camera_distance_mm or 0.0, {}, 0

    resolved_distance = camera_distance_mm
    if resolved_distance is None and reference_diameter_mm is not None:
        reference_radius_px = float(max(circles, key=lambda c: c[2])[2])
        resolved_distance = estimate_camera_distance_mm(
            measured_radius_pixels=reference_radius_px,
            known_radius_mm=reference_diameter_mm / 2.0,
            focal_length_px=focal_length_px,
        )

    if resolved_distance is None:
        raise ValueError("Provide camera_distance_mm or reference_diameter_mm for physical size estimation")

    detections: List[Dict[str, object]] = []
    counts: Dict[int, int] = {}
    total_value = 0

    for x, y, radius_px in circles:
        radius_mm = actual_radius_mm(float(radius_px), resolved_distance, focal_length_px)
        diameter_mm = radius_mm * 2.0
        spec = classify_by_diameter(diameter_mm)
        if spec is None:
            label = "Unknown"
        else:
            label = f"Rs. {spec.denomination} ({spec.series})"
            counts[spec.denomination] = counts.get(spec.denomination, 0) + 1
            total_value += spec.denomination

        detections.append(
            {
                "center": (int(x), int(y)),
                "radius_px": float(radius_px),
                "radius_mm": radius_mm,
                "diameter_mm": diameter_mm,
                "label": label,
            }
        )

    return detections, resolved_distance, counts, total_value


def annotate_frame(
    frame: np.ndarray,
    detections: List[Dict[str, object]],
    camera_distance_mm: float,
    counts: Dict[int, int],
    total_value: int,
) -> np.ndarray:
    require_cv2()
    output = frame.copy()
    for detection in detections:
        center = detection["center"]
        radius_px = int(detection["radius_px"])
        label = str(detection["label"])
        cv2.circle(output, center, radius_px, (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (center[0] - 40, center[1] - radius_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    y = 24
    for denomination in sorted(counts):
        cv2.putText(
            output,
            f"Rs. {denomination}: {counts[denomination]}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    cv2.putText(
        output,
        f"Total: Rs. {total_value}",
        (10, y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Camera distance: {camera_distance_mm:.1f} mm",
        (10, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def run_image_mode(args: argparse.Namespace) -> None:
    require_cv2()
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    detections, distance, counts, total = analyze_frame(
        frame,
        focal_length_px=args.focal_length,
        camera_distance_mm=args.distance,
        reference_diameter_mm=args.reference_diameter,
    )
    annotated = annotate_frame(frame, detections, distance, counts, total)
    cv2.imshow("Coin Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_realtime_mode(args: argparse.Namespace) -> None:
    require_cv2()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Unable to open camera")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            detections, distance, counts, total = analyze_frame(
                frame,
                focal_length_px=args.focal_length,
                camera_distance_mm=args.distance,
                reference_diameter_mm=args.reference_diameter,
            )
            annotated = annotate_frame(frame, detections, distance, counts, total)
            cv2.imshow("Coin Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect, classify, count, and valuate Sri Lankan coins")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to an image for static detection")
    mode.add_argument("--realtime", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for realtime mode")
    parser.add_argument("--focal-length", type=float, required=True, help="Focal length in pixels")
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Camera distance in millimeters. Optional when --reference-diameter is provided.",
    )
    parser.add_argument(
        "--reference-diameter",
        type=float,
        default=None,
        help="Known reference coin diameter in millimeters to estimate distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        run_image_mode(args)
    else:
        run_realtime_mode(args)


if __name__ == "__main__":
    main()
