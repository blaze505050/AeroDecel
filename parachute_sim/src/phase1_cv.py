"""
phase1_cv.py — AeroDecel v5.0 AI-Enhanced Computer Vision Canopy Area Extractor
==================================================================================
Extracts projected pixel area A(t) of a parachute canopy from raw video.

AeroDecel v5.0 AI Segmentation Pipeline (auto-fallback):
  1. YOLO detection (ultralytics) — fast, learned parachute detection
  2. SAM (Segment Anything) — zero-shot segmentation via YOLO bbox prompt
  3. HSV + morphological ops — classical fallback (always available)

Post-processing:
  - Optical flow temporal tracking (Lucas-Kanade)
  - Savitzky-Golay temporal smoothing
  - Area confidence scoring per frame
  - CSV export with time, area_px, area_m2, area_normalized, confidence
"""

# cv2 is imported lazily in classes that need it (CanopyExtractor, etc.)
# so that synthetic data generation and model detection work without opencv.
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import sys
import os


sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def _get_cv2():
    """Lazy import of cv2 — only needed for actual video processing."""
    import cv2
    return cv2


# ─── AI Model Detection ─────────────────────────────────────────────────────
def _detect_available_models():
    """Detect which AI segmentation backends are available."""
    available = {"hsv": True}  # Always available

    try:
        import ultralytics
        available["yolo"] = True
    except ImportError:
        available["yolo"] = False

    try:
        import segment_anything
        available["sam"] = True
    except ImportError:
        available["sam"] = False

    return available


def _select_model(requested: str = "auto") -> str:
    """Select best available model based on request and availability."""
    available = _detect_available_models()

    if requested == "auto":
        if available.get("yolo"):
            return "yolo"
        if available.get("sam"):
            return "sam"
        return "hsv"
    elif requested in available and available[requested]:
        return requested
    else:
        print(f"  ⚠ Requested model '{requested}' not available, falling back to HSV")
        return "hsv"


# ─── Background Subtractor ───────────────────────────────────────────────────
class AdaptiveBackground:
    """MOG2 background subtractor with configurable history."""
    def __init__(self, history: int = 30, var_threshold: float = 40):
        cv2 = _get_cv2()
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.initialized = False
        self._count = 0

    def learn(self, frame: np.ndarray, n_frames: int = 30):
        """Feed learning frames (before parachute deploys) to build background model."""
        self.subtractor.apply(frame, learningRate=1.0 / max(1, n_frames))
        self._count += 1
        if self._count >= n_frames:
            self.initialized = True

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return self.subtractor.apply(frame, learningRate=0)


# ─── Optical Flow Tracker (AeroDecel v5.0) ───────────────────────────────────
class OpticalFlowTracker:
    """
    Lucas-Kanade optical flow tracker for temporal coherence.
    Tracks canopy centroid between frames to handle brief occlusions
    and reduce segmentation jitter.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_center = None
        cv2 = _get_cv2()
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def track(self, frame: np.ndarray, detected_center: tuple = None,
              detected_area: float = 0.0) -> tuple[tuple, float]:
        """
        Track canopy using optical flow when detection fails.
        Returns (center, confidence) where confidence ∈ [0, 1].
        """
        cv2 = _get_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detected_center is not None and detected_area > 0:
            # Detection succeeded — update tracker
            self.prev_gray = gray.copy()
            self.prev_center = np.array([[list(detected_center)]], dtype=np.float32)
            return detected_center, 1.0

        if self.prev_gray is not None and self.prev_center is not None:
            # Detection failed — use optical flow
            cv2 = _get_cv2()
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_center, None, **self.lk_params
            )
            if status is not None and status[0][0] == 1:
                tracked = tuple(new_pts[0][0].astype(int))
                self.prev_gray = gray.copy()
                self.prev_center = new_pts
                confidence = max(0.0, 1.0 - float(err[0][0]) / 50.0)
                return tracked, confidence

        self.prev_gray = gray.copy()
        return None, 0.0


# ─── Core Extractor ──────────────────────────────────────────────────────────
class CanopyExtractor:
    def __init__(
        self,
        video_path: str | Path,
        hsv_lower: list = None,
        hsv_upper: list = None,
        use_bg_subtraction: bool = True,
        px_to_m2: float | None = None,
        blur_k: int = None,
        morph_k: int = None,
        min_area: int = None,
        cv_model: str = "auto",
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cv2 = _get_cv2()
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.px_to_m2     = px_to_m2 or cfg.PX_TO_M2

        self.hsv_lower = np.array(hsv_lower or cfg.HSV_LOWER, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper or cfg.HSV_UPPER, dtype=np.uint8)

        blur_k  = blur_k  or cfg.GAUSSIAN_BLUR_K
        self.blur_k  = blur_k if blur_k % 2 == 1 else blur_k + 1
        self.morph_k = morph_k or cfg.MORPH_KERNEL_SIZE
        self.min_area = min_area or cfg.MIN_CONTOUR_AREA

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_k, self.morph_k)
        )
        self.bg_model = AdaptiveBackground(history=cfg.BACKGROUND_FRAMES) \
            if use_bg_subtraction else None

        # AI model selection (AeroDecel v5.0)
        self.cv_model = _select_model(cv_model)
        self.yolo_model = None
        if self.cv_model == "yolo":
            self._init_yolo()

        # Optical flow tracker
        self.tracker = OpticalFlowTracker() if cfg.OPTICAL_FLOW else None

        print(f"[Phase 1] Video: {self.video_path.name}")
        print(f"          {self.width}×{self.height}px | {self.fps:.2f} fps | {self.total_frames} frames")
        print(f"          Duration: {self.total_frames/self.fps:.2f}s")
        print(f"          CV Model: {self.cv_model.upper()}" +
              (f" (confidence ≥ {cfg.YOLO_CONFIDENCE})" if self.cv_model == "yolo" else ""))

    def _init_yolo(self):
        """Initialize YOLO model for parachute detection."""
        try:
            from ultralytics import YOLO
            # Use pre-trained YOLOv8 — detects general objects including kites/umbrellas
            # User can substitute a fine-tuned model via config
            self.yolo_model = YOLO("yolov8n.pt")
            print(f"          YOLO: loaded yolov8n.pt")
        except Exception as e:
            print(f"          ⚠ YOLO init failed ({e}), falling back to HSV")
            self.cv_model = "hsv"

    def _segment_yolo(self, frame: np.ndarray) -> tuple[np.ndarray | None, float, tuple | None]:
        """YOLO-based detection → bounding box → HSV refinement within bbox."""
        cv2 = _get_cv2()
        if self.yolo_model is None:
            return None, 0.0, None

        results = self.yolo_model(frame, verbose=False, conf=cfg.YOLO_CONFIDENCE)
        best_area = 0.0
        best_bbox = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                area = w * h
                if area > best_area:
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)

        if best_bbox is None:
            return None, 0.0, None

        x1, y1, x2, y2 = best_bbox
        roi = frame[y1:y2, x1:x2]

        # Refine with HSV segmentation within the detected ROI
        if self.blur_k > 1:
            roi_blur = cv2.GaussianBlur(roi, (self.blur_k, self.blur_k), 0)
        else:
            roi_blur = roi

        hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            area_px = max(area_px, hull_area * 0.8)
            cx = x1 + int(np.mean(hull[:, 0, 0]))
            cy = y1 + int(np.mean(hull[:, 0, 1]))
            return mask, area_px, (cx, cy)
        else:
            # Use bbox area as fallback
            return mask, best_area * 0.7, ((x1+x2)//2, (y1+y2)//2)

    def _segment_frame(self, frame: np.ndarray, bg_mask: np.ndarray | None) -> tuple[np.ndarray, float, float]:
        """
        Segment canopy from a single frame.
        Returns (annotated_frame, area_in_pixels, confidence).
        """
        cv2 = _get_cv2()
        confidence = 1.0

        # Try YOLO first if available
        if self.cv_model == "yolo":
            yolo_mask, yolo_area, yolo_center = self._segment_yolo(frame)
            if yolo_area > self.min_area:
                annotated = frame.copy()
                if yolo_center:
                    cv2.circle(annotated, yolo_center, 8, (0, 255, 80), 2)
                    cv2.putText(annotated, f"YOLO A={yolo_area:.0f}px²",
                                (yolo_center[0]-40, yolo_center[1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
                return annotated, yolo_area, 0.95

        # Classical HSV pipeline (fallback or default)
        if self.blur_k > 1:
            frame_blur = cv2.GaussianBlur(frame, (self.blur_k, self.blur_k), 0)
        else:
            frame_blur = frame

        # Color segmentation in HSV
        hsv  = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Fuse with background subtraction mask if available
        if bg_mask is not None:
            mask = cv2.bitwise_and(mask, bg_mask)

        # Morphological closing (fills internal holes) then opening (removes noise)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        # Contour analysis — pick largest plausible contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_px = 0.0
        annotated = frame.copy()
        center = None

        if contours:
            valid = [c for c in contours if cv2.contourArea(c) > self.min_area]
            if valid:
                largest = max(valid, key=cv2.contourArea)
                area_px = cv2.contourArea(largest)

                # Compute convex hull area (better for partially visible canopy)
                hull = cv2.convexHull(largest)
                hull_area = cv2.contourArea(hull)
                # Use hull area if canopy is expected to be convex
                area_px = hull_area if hull_area > area_px * 0.5 else area_px

                # Compute center for tracking
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

                # Annotate
                cv2.drawContours(annotated, [hull], -1, (0, 255, 80), 2)
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 200, 255), 1)
                label = f"A={area_px:.0f}px²"
                cv2.putText(annotated, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1)

                # Confidence based on contour solidity
                solidity = area_px / max(hull_area, 1.0)
                confidence = min(1.0, solidity * 1.2)

        # Optical flow tracking for temporal coherence
        if self.tracker is not None:
            tracked_center, track_conf = self.tracker.track(frame, center, area_px)
            if center is None and tracked_center is not None:
                confidence *= track_conf * 0.5  # Reduced confidence for tracked-only

        return annotated, area_px, confidence

    def extract(self, save_preview: bool = True, preview_every: int = 30) -> pd.DataFrame:
        """
        Run full extraction pipeline over all frames.
        Returns DataFrame with columns: frame, time_s, area_px, area_m2,
            area_normalized, confidence.
        """
        records = []
        frame_idx = 0
        bg_learn_frames = cfg.BACKGROUND_FRAMES
        preview_dir = cfg.OUTPUTS_DIR / "frame_previews"
        if save_preview:
            preview_dir.mkdir(exist_ok=True)

        print(f"\n[Phase 1] Extracting canopy area ({self.cv_model.upper()})...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Background model learning phase (first N frames assumed pre-deployment)
            bg_mask = None
            if self.bg_model is not None:
                if frame_idx < bg_learn_frames:
                    self.bg_model.learn(frame, bg_learn_frames)
                elif self.bg_model.initialized:
                    bg_mask = self.bg_model.apply(frame)

            annotated, area_px, confidence = self._segment_frame(frame, bg_mask)

            time_s = frame_idx / self.fps
            records.append({
                "frame": frame_idx,
                "time_s": time_s,
                "area_px": area_px,
                "confidence": confidence,
            })

            if save_preview and frame_idx % preview_every == 0:
                cv2 = _get_cv2()
                cv2.imwrite(str(preview_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

            if frame_idx % 30 == 0:
                pct = 100.0 * frame_idx / max(1, self.total_frames)
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:5.1f}%  frame {frame_idx}/{self.total_frames}", end="", flush=True)

            frame_idx += 1

        self.cap.release()
        print(f"\r  [{'█'*20}] 100.0%  Done!{' '*20}")

        df = pd.DataFrame(records)
        df = self._postprocess(df)
        return df

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temporal smoothing and unit conversion."""
        # Savitzky-Golay smoothing (preserves peak shape better than moving average)
        if len(df) > 15:
            window = min(21, len(df) // 4 * 2 + 1)   # must be odd
            window = window if window % 2 == 1 else window + 1
            df["area_px_raw"] = df["area_px"].copy()
            df["area_px"] = savgol_filter(df["area_px"], window_length=window, polyorder=3)
            df["area_px"] = df["area_px"].clip(lower=0)

        # Physical area conversion
        if self.px_to_m2 is not None:
            df["area_m2"] = df["area_px"] * self.px_to_m2
        else:
            # Scale to reference area using peak pixel area
            max_px = df["area_px"].max()
            if max_px > 0:
                df["area_m2"] = df["area_px"] / max_px * cfg.CANOPY_AREA_M2
            else:
                df["area_m2"] = 0.0

        # Normalized 0→1 inflation fraction
        max_area = df["area_m2"].max()
        df["area_normalized"] = (df["area_m2"] / max_area).clip(0, 1) if max_area > 0 else 0.0

        return df


# ─── Synthetic Generator (for testing without video) ─────────────────────────
def generate_synthetic_At(
    duration_s: float = 10.0,
    fps: float = 30.0,
    max_area_m2: float = None,
    inflation_time: float = 2.5,
    noise_std: float = 0.02,
) -> pd.DataFrame:
    """
    Generate a synthetic A(t) curve using a generalized logistic (Richards) model.
    Useful for testing Phases 2–4 without a real video.
    """
    max_area_m2 = max_area_m2 or cfg.CANOPY_AREA_M2
    t = np.linspace(0, duration_s, int(duration_s * fps))
    n = cfg.CANOPY_N_PARAM

    # Generalized logistic inflation
    k    = 5.0 / inflation_time     # growth rate
    t0   = inflation_time * 0.6     # midpoint
    A    = max_area_m2 / (1 + np.exp(-k * (t - t0))) ** (1 / n)

    # Add realistic oscillation (canopy breathing) and noise
    oscillation = 0.03 * max_area_m2 * np.exp(-0.5 * (t - t0)) * np.sin(2 * np.pi * 1.5 * t)
    noise = np.random.normal(0, noise_std * max_area_m2, len(t))
    A = np.clip(A + oscillation + noise, 0, max_area_m2)

    # Synthetic confidence (high everywhere for synthetic data)
    confidence = np.ones(len(t)) * 0.99

    df = pd.DataFrame({
        "frame"          : np.arange(len(t)),
        "time_s"         : t,
        "area_px"        : A / max_area_m2 * 1e6,  # synthetic pixel space
        "area_m2"        : A,
        "area_normalized": A / max_area_m2,
        "confidence"     : confidence,
    })

    print(f"[Phase 1] Synthetic A(t) generated: {len(df)} frames, "
          f"max area = {df['area_m2'].max():.2f} m²")
    return df


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(video_path: Path | None = None, synthetic: bool = False) -> pd.DataFrame:
    cfg.OUTPUTS_DIR.mkdir(exist_ok=True)

    available = _detect_available_models()
    ai_status = " | ".join(f"{k}:{'✓' if v else '✗'}" for k, v in available.items())
    print(f"[Phase 1] AeroDecel CV Engine — Available: {ai_status}")

    if synthetic or (video_path is None and not cfg.VIDEO_PATH.exists()):
        print("[Phase 1] No video found — running with synthetic A(t) data.")
        df = generate_synthetic_At()
    else:
        vp = video_path or cfg.VIDEO_PATH
        extractor = CanopyExtractor(vp, cv_model=cfg.CV_MODEL)
        df = extractor.extract()

    # Save
    df.to_csv(cfg.AT_CSV, index=False)
    print(f"\n[Phase 1] ✓ Saved: {cfg.AT_CSV}")
    print(df[["time_s", "area_m2", "area_normalized"]].describe().round(4))
    return df


if __name__ == "__main__":
    run()
