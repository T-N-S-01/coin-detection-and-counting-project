import statistics
from collections import deque

import cv2
import numpy as np

COIN_SPECS = [
    # Current circulation coins (2017 series)
    {"name": "Rs. 1",          "diameter_mm": 20.0, "value": 1},
    {"name": "Rs. 2",          "diameter_mm": 22.0, "value": 2},
    {"name": "Rs. 5",          "diameter_mm": 23.5, "value": 5},
    {"name": "Rs. 10",         "diameter_mm": 26.4, "value": 10},
    # Older series / cents coins (still legal tender)
    {"name": "25 Cents",       "diameter_mm": 16.0, "value": 0.25},
    {"name": "50 Cents",       "diameter_mm": 18.0, "value": 0.50},
    {"name": "Rs. 2 (Older)",  "diameter_mm": 29.0, "value": 2},
    # {"name": "Rs. 10 (Older)", "diameter_mm": 27.0, "value": 10},
]

# The coin you will show alone in frame to calibrate (must match a "name" above).
CALIBRATION_COIN = "Rs. 10"

# How close (in mm) a detected coin's diameter must be to a known coin to
# count as a match.
DIAMETER_TOLERANCE_MM = 0.8

# ---------------------------------------------------------------------------
# 2. DISTANCE GUIDANCE
#    The pixel radius range in which a coin is "close enough to measure
#    accurately, far enough to stay fully in frame". Tune this to your
#    camera/resolution: if "GOOD DISTANCE" is basically unreachable, widen
#    the range; if calibration keeps being inaccurate, narrow it.
# ---------------------------------------------------------------------------
IDEAL_RADIUS_RANGE_PX = (70, 130)

# How many "good distance" frames to average before calibration is allowed.
# Higher = more stable calibration, but takes longer to hold steady.
MIN_CALIBRATION_SAMPLES = 12

# ---------------------------------------------------------------------------
pixels_per_mm = None  # becomes a number once you calibrate with 'c'
calib_buffer = deque(maxlen=MIN_CALIBRATION_SAMPLES * 2)


def detect_circles(gray_blurred):
    """Run Hough Circle detection. Returns an array of (x, y, r) or None."""
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=40,
        minRadius=15,
        maxRadius=150,
    )
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return None


def refine_circle_radius(gray_blurred, x, y, r):

    h, w = gray_blurred.shape[:2]
    pad = int(r * 0.4) + 5
    x0, y0 = max(0, x - r - pad), max(0, y - r - pad)
    x1, y1 = min(w, x + r + pad), min(h, y + r + pad)
    roi = gray_blurred[y0:y1, x0:x1]
    if roi.size == 0:
        return float(r)

    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float(r)

    expected_area = np.pi * (r ** 2)
    best, best_diff = None, None
    for c in contours:
        area = cv2.contourArea(c)
        if area < expected_area * 0.3:   # too small to be this coin's edge
            continue
        diff = abs(area - expected_area)
        if best_diff is None or diff < best_diff:
            best, best_diff = c, diff

    if best is None:
        return float(r)

    _, refined_r = cv2.minEnclosingCircle(best)
    # Sanity check: don't accept a wildly different radius (bad contour match)
    if refined_r <= 0 or abs(refined_r - r) > r * 0.5:
        return float(r)
    return float(refined_r)


def classify_coin(diameter_mm):
    """Return the COIN_SPECS entry with the closest diameter, if within tolerance."""
    best, best_diff = None, None
    for coin in COIN_SPECS:
        diff = abs(coin["diameter_mm"] - diameter_mm)
        if best_diff is None or diff < best_diff:
            best, best_diff = coin, diff
    if best is not None and best_diff <= DIAMETER_TOLERANCE_MM:
        return best
    return None


def distance_feedback(radius_px):
    """Return (message, bgr_color) guiding the user to the ideal distance."""
    lo, hi = IDEAL_RADIUS_RANGE_PX
    if radius_px < lo:
        return "TOO FAR -> move camera CLOSER", (0, 0, 255)
    elif radius_px > hi:
        return "TOO CLOSE -> move camera FARTHER AWAY", (0, 0, 255)
    return "GOOD DISTANCE", (0, 200, 0)


def calibrate(radius_px):
    """Set pixels_per_mm from the calibration coin's measured pixel radius."""
    global pixels_per_mm
    ref = next(c for c in COIN_SPECS if c["name"] == CALIBRATION_COIN)
    measured_diameter_px = radius_px * 2
    pixels_per_mm = measured_diameter_px / ref["diameter_mm"]
    print(f"[CALIBRATED] {pixels_per_mm:.3f} px/mm using {CALIBRATION_COIN} "
          f"(avg radius {radius_px:.1f}px over {len(calib_buffer)} samples)")


def main():
    cap = cv2.VideoCapture(1)
    total_value = 0.0
    counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 5)
        circles = detect_circles(blur)

        total_value = 0.0
        counts = {}
        largest_refined_r = None

        if circles is not None:
            # Refine every detected circle's radius up front
            refined = [(x, y, refine_circle_radius(blur, x, y, r)) for (x, y, r) in circles]
            largest_refined_r = max(r for _, _, r in refined)

            if pixels_per_mm is None:
                for (x, y, r) in refined:
                    cv2.circle(frame, (x, y), int(r), (0, 255, 255), 2)
                    cv2.putText(frame, f"r={r:.0f}px", (x - 20, y - int(r) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                msg, color = distance_feedback(largest_refined_r)
                if msg == "GOOD DISTANCE":
                    calib_buffer.append(largest_refined_r)
                else:
                    calib_buffer.clear()  # distance changed, discard stale samples

                cv2.putText(frame, msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                ready = len(calib_buffer) >= MIN_CALIBRATION_SAMPLES
                hint = ("Hold steady, then press 'c' to calibrate" if ready
                        else f"Hold steady at good distance... {len(calib_buffer)}/{MIN_CALIBRATION_SAMPLES}")
                cv2.putText(frame, hint, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                for (x, y, r) in refined:
                    diameter_mm = (r * 2) / pixels_per_mm
                    coin = classify_coin(diameter_mm)
                    if coin:
                        cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)
                        cv2.putText(frame, f"{coin['name']} ({diameter_mm:.1f}mm)",
                                    (x - 55, y - int(r) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        total_value += coin["value"]
                        counts[coin["name"]] = counts.get(coin["name"], 0) + 1
                    else:
                        cv2.circle(frame, (x, y), int(r), (0, 0, 255), 2)
                        cv2.putText(frame, f"?{diameter_mm:.1f}mm", (x - 30, y - int(r) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Keep showing distance guidance so the counting distance
                # stays consistent with the calibration distance.
                msg, color = distance_feedback(largest_refined_r)
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


                y0 = 60
                cv2.putText(frame, f"Total coins: {sum(counts.values())}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Total value: Rs. {total_value:.2f}",
                            (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, (name, cnt) in enumerate(counts.items()):
                    cv2.putText(frame, f"{name}: {cnt}", (10, y0 + 60 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        else:
            calib_buffer.clear()
            cv2.putText(frame, "No coin detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Coin Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            if pixels_per_mm is None and len(calib_buffer) >= MIN_CALIBRATION_SAMPLES:
                calibrate(statistics.median(calib_buffer))
                calib_buffer.clear()
            elif pixels_per_mm is None:
                print(f"Not enough steady samples yet "
                      f"({len(calib_buffer)}/{MIN_CALIBRATION_SAMPLES}) - "
                      f"hold the coin at GOOD DISTANCE first.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- FINAL COUNT (last processed frame) ---")
    if not counts:
        print("No calibrated coins were detected.")
    else:
        for name, cnt in counts.items():
            print(f"{name}: {cnt}")
        print(f"TOTAL COINS: {sum(counts.values())}")
        print(f"TOTAL VALUE: Rs. {total_value:.2f}")


if __name__ == "__main__":
    main()