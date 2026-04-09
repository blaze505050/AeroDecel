"""
config.py — Central configuration for the AeroDecel Framework v6.0.

AeroDecel: AI-Driven Aerodynamic Deceleration Analysis Framework
All tunable parameters live here. Edit before running.
"""

from pathlib import Path

# ─── AeroDecel Identity ──────────────────────────────────────────────────────
AERODECEL_VERSION   = "6.0.0"
AERODECEL_NAME      = "AeroDecel"
AERODECEL_TAGLINE   = "AI-Driven Aerodynamic Deceleration Analysis"
AERODECEL_EQ        = "m·dv/dt = mg − ½ρ(h)·v²·Cd(t)·A(t)"

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR  = ROOT_DIR / "models"

VIDEO_PATH  = DATA_DIR / "drop_test.mp4"   # ← Insert your .mp4 file here
AT_CSV      = OUTPUTS_DIR / "At_curve.csv"
ODE_CSV     = OUTPUTS_DIR / "ode_results.csv"
PINN_CSV    = OUTPUTS_DIR / "pinn_cd_curve.csv"

# ─── Physical Constants ───────────────────────────────────────────────────────
GRAVITY         = 9.80665       # m/s²  — standard gravity
PARACHUTE_MASS  = 80.0          # kg    — total payload + canopy mass
INITIAL_ALT     = 1000.0        # m     — deployment altitude (AGL)
INITIAL_VEL     = 25.0          # m/s   — velocity at deployment trigger
CD_INITIAL      = 1.5           # —     — initial drag coefficient guess
CANOPY_AREA_M2  = 50.0          # m²    — fully open canopy reference area

# Pixel-to-meter² scale factor for canopy area conversion.
# If you know the canopy's actual max area (m²) and its pixel area at full open,
# set: PX_TO_M2 = CANOPY_AREA_M2 / max_pixel_area.
# If unknown, set to None → Phase 2 uses normalized area directly.
PX_TO_M2 = None

# ─── Advanced Physics (AeroDecel v5.0) ───────────────────────────────────────
# Added mass: virtual mass correction for accelerating body in fluid
# m_eff = m + C_a · ρ · V_canopy   (C_a ≈ 0.5 for hemisphere)
ADDED_MASS_ENABLED  = True
ADDED_MASS_COEFF    = 0.5          # C_a — added mass coefficient (0.5 for hemisphere)
CANOPY_VOLUME_M3    = 0.4          # m³  — inflated canopy volume estimate

# Reynolds-dependent Cd correction
RE_CORRECTION_ENABLED  = True
CANOPY_DIAMETER_M      = 8.0       # Nominal canopy diameter for Re calculation

# Mach-number compressibility correction (Prandtl-Glauert)
MACH_CORRECTION_ENABLED = True

# Fabric porosity drag reduction
POROSITY_ENABLED       = True
POROSITY_COEFF         = 0.012     # k_p — fabric porosity coefficient

# Buoyancy correction at high altitude
BUOYANCY_ENABLED       = True

# ─── Computer Vision (Phase 1) ───────────────────────────────────────────────
# HSV thresholds for canopy segmentation. Tune to your canopy color.
#   Orange/Yellow canopy:  lower=[10,80,80],  upper=[35,255,255]
#   Red canopy:            lower=[0,120,100], upper=[10,255,255]
#   White/Silver canopy:   lower=[0,0,200],   upper=[180,30,255]
HSV_LOWER = [10,  80,  80]
HSV_UPPER = [35, 255, 255]

MORPH_KERNEL_SIZE  = 7        # Morphological kernel (increase for porous canopies)
MIN_CONTOUR_AREA   = 300      # px²  — noise filter threshold
GAUSSIAN_BLUR_K    = 5        # Pre-blur kernel (odd number); 0 to disable
BACKGROUND_FRAMES  = 30       # Frames to sample for background subtraction

# AI Segmentation model selection (AeroDecel v5.0)
# 'auto' = try YOLO → SAM → HSV fallback
# 'hsv'  = traditional HSV only
# 'yolo' = YOLO detection + HSV refinement
# 'sam'  = Segment Anything Model
CV_MODEL           = "auto"
YOLO_CONFIDENCE    = 0.5       # YOLO detection confidence threshold
OPTICAL_FLOW       = True      # Lucas-Kanade temporal tracking between frames

# ─── ODE Simulator (Phase 2) ─────────────────────────────────────────────────
ODE_METHOD         = "RK45"   # scipy solver: RK45, DOP853, Radau, LSODA
ODE_RTOL           = 1e-7
ODE_ATOL           = 1e-9
INFLATION_MODEL    = "generalized_logistic"   # "csv_interpolated" | "generalized_logistic" | "polynomial"
CANOPY_N_PARAM     = 2.0      # Inflation exponent n (generalized logistic)

# ─── PINN (Phase 3) — AeroDecel v5.0 Research-Grade ─────────────────────────
PINN_HIDDEN_LAYERS = [128, 256, 256, 128, 64]
PINN_EPOCHS        = 8000
PINN_LR            = 3e-4
PINN_PHYSICS_WEIGHT = 10.0    # λ for physics residual loss
PINN_DATA_WEIGHT    = 1.0     # λ for data-match loss
PINN_SMOOTH_WEIGHT  = 0.1     # λ for smoothness regularization
PINN_COLLOCATION_PTS = 2000   # Interior collocation points for physics loss
PINN_ACTIVATION    = "tanh"   # "tanh" | "silu" | "gelu"
PINN_LR_SCHEDULE   = True     # Use cosine annealing LR scheduler
DEVICE             = "auto"   # "auto" | "cpu" | "cuda" | "mps"

# PINN v5.0 advanced features
PINN_FOURIER_FEATURES   = True      # Fourier feature embeddings for high-freq capture
PINN_FOURIER_SCALES     = [1.0, 2.0, 4.0, 8.0]   # σ scales for sin/cos embeddings
PINN_CURRICULUM         = True      # Curriculum training: data → data+physics
PINN_CURRICULUM_WARMUP  = 1000      # Epochs of data-only warmup before physics loss
PINN_ADAPTIVE_WEIGHTS   = True      # Self-tuning loss weights (NTK-inspired)
PINN_VALIDATION_FRAC    = 0.1       # Fraction held out for early stopping
PINN_DUAL_OUTPUT        = True      # Predict both v(t) and Cd(t) simultaneously

# ─── Visualization (Phase 4) ─────────────────────────────────────────────────
DPI            = 150
FIG_FORMAT     = "png"        # "png" | "pdf" | "svg"
DARK_THEME     = True
COLOR_THEORY   = "#00d4ff"    # Theoretical curve color
COLOR_PINN     = "#ff6b35"    # PINN/ML curve color
COLOR_RAW      = "#a8ff3e"    # Raw CV data color
COLOR_GRID     = "#1a2744"

# ─── Directories Auto-Create ─────────────────────────────────────────────────
for _d in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
