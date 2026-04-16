# AeroDecel v6.0 — Project Icarus — Configuration
# ─────────────────────────────────────────────────

# ── Planet ───────────────────────────────────────────────────────────────────
PLANET = "mars"          # mars | venus | titan

# ── Thermal Protection System ─────────────────────────────────────────────────
TPS_MATERIAL  = "nylon"  # nylon | kevlar | nomex | vectran | zylon
TPS_THICKNESS = 0.01     # metres

# ── Canopy Geometry ───────────────────────────────────────────────────────────
CANOPY_SHAPE      = "elliptical"          # elliptical | circular | rectangular
CANOPY_DIMENSIONS = {"a": 10, "b": 5}    # elliptical: a,b  circular: r  rectangular: width,height

# ── LBM ──────────────────────────────────────────────────────────────────────
LBM_RESOLUTION = (100, 100)
LBM_REYNOLDS   = 1000

# ── PINN ─────────────────────────────────────────────────────────────────────
PINN_LAYERS        = [2, 64, 64, 64, 4]   # input→hidden→output
PINN_LEARNING_RATE = 1e-3
PINN_EPOCHS        = 2000

# ── Neural Operator ───────────────────────────────────────────────────────────
OPERATOR_TYPE = "fno"        # fno | deeponet
FNO_MODES     = 16
FNO_WIDTH     = 64

# ── Demo ──────────────────────────────────────────────────────────────────────
DEMO_CASE = "mars_edl"   # mars_edl | drone_recovery | reentry_capsule | military_airdrop

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
DARK_THEME = True
