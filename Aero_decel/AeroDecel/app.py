"""
app.py — AeroDecel v6.0 Dash Web Dashboard
==========================================
Run: python app.py
Install: pip install dash plotly
"""
import sys
import numpy as np

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    raise SystemExit("Install: pip install dash plotly")

import config as cfg
from src.planetary_atm   import get_planet_atmosphere
from src.thermal_model   import ThermalProtectionSystem, MATERIAL_DB
from src.canopy_geometry import CanopyGeometry

app = dash.Dash(__name__, title="AeroDecel v6.0 — Project Icarus",
                suppress_callback_exceptions=True)

# ── Layout ────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#080c14", plot_bgcolor="#0d1526",
    font=dict(color="#c8d8f0", family="monospace"),
    gridcolor="#1a2744",
)

app.layout = html.Div(style={"background":"#080c14","minHeight":"100vh","fontFamily":"monospace"}, children=[
    html.Div(style={"background":"#0d1526","borderBottom":"1px solid #1a2744","padding":"16px 24px"}, children=[
        html.H1("🚀 AeroDecel v6.0 — Project Icarus",
                style={"color":"#00d4ff","margin":0,"fontSize":22,"fontWeight":600}),
        html.P("Planetary Entry, Descent & Landing Simulation Framework",
               style={"color":"#556688","margin":0,"fontSize":12}),
    ]),

    dcc.Tabs(id="tabs", value="atm",
             colors={"border":"#1a2744","primary":"#00d4ff","background":"#0d1526"},
             children=[
        dcc.Tab(label="🌍 Atmosphere",    value="atm",    style=_tab(), selected_style=_stab()),
        dcc.Tab(label="🔥 Thermal",       value="tps",    style=_tab(), selected_style=_stab()),
        dcc.Tab(label="🪂 Canopy",        value="canopy", style=_tab(), selected_style=_stab()),
        dcc.Tab(label="🌊 CFD (LBM)",     value="cfd",    style=_tab(), selected_style=_stab()),
        dcc.Tab(label="🧠 Neural Ops",    value="nn",     style=_tab(), selected_style=_stab()),
    ]),

    html.Div(id="tab-content", style={"padding":"20px"}),
])


def _tab():
    return {"background":"#0d1526","color":"#556688","border":"1px solid #1a2744","padding":"8px 16px"}

def _stab():
    return {"background":"#1a2744","color":"#00d4ff","border":"1px solid #00d4ff","padding":"8px 16px","fontWeight":"bold"}

def _card(*children, border="#1a2744"):
    return html.Div(style={"background":"#0d1526","border":f"1px solid {border}",
                            "borderRadius":"8px","padding":"16px","margin":"8px 0"},
                    children=list(children))

def _row(*cols):
    return html.Div(style={"display":"grid","gridTemplateColumns":" ".join(["1fr"]*len(cols)),"gap":"16px"},
                    children=list(cols))

def _label(txt):
    return html.Label(txt, style={"color":"#556688","fontSize":11,"marginBottom":4,"display":"block"})

def _dd(id_, opts, val, **kw):
    return dcc.Dropdown(id=id_, options=opts, value=val,
                        style={"background":"#1a2744","color":"#c8d8f0","border":"1px solid #2a3d6e"},
                        **kw)

def _sl(id_, lo, hi, val, step=None):
    return dcc.Slider(id=id_, min=lo, max=hi, value=val, step=step or (hi-lo)/100,
                      marks=None, tooltip={"placement":"bottom","always_visible":True})

def _metric(label, value, color="#00d4ff"):
    return html.Div(style={"background":"#080c14","borderRadius":8,"padding":"12px","textAlign":"center"}, children=[
        html.Div(label, style={"color":"#556688","fontSize":10,"marginBottom":4}),
        html.Div(value, style={"color":color,"fontSize":22,"fontWeight":600}),
    ])


# ── Callback: render tab content ──────────────────────────────────────────────

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "atm":   return layout_atmosphere()
    if tab == "tps":   return layout_thermal()
    if tab == "canopy":return layout_canopy()
    if tab == "cfd":   return layout_cfd()
    if tab == "nn":    return layout_nn()


# ════════════════════════════════════════════════════════════
# TAB 1 — Atmosphere
# ════════════════════════════════════════════════════════════

def layout_atmosphere():
    return html.Div([
        _row(
            html.Div([
                _label("Planet"); _dd("atm-planet",[{"label":p.title(),"value":p} for p in ["mars","venus","titan"]],"mars"),
                html.Br(),
                _label("Max Altitude [km]"); _sl("atm-alt",10,200,100,10),
            ]),
            html.Div([dcc.Graph(id="atm-density"), dcc.Graph(id="atm-temp")]),
        ),
        html.Div(id="atm-metrics", style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":8,"marginTop":8}),
    ])

@app.callback(
    Output("atm-density","figure"), Output("atm-temp","figure"), Output("atm-metrics","children"),
    Input("atm-planet","value"), Input("atm-alt","value"))
def update_atm(planet_name, alt_max_km):
    planet = get_planet_atmosphere(planet_name)
    alts   = np.linspace(0, alt_max_km*1000, 300)
    dens   = np.array([planet.density(h)     for h in alts])
    temps  = np.array([planet.temperature(h) for h in alts])
    pres   = np.array([planet.pressure(h)    for h in alts])

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=dens, y=alts/1000, name="Density",
                             line=dict(color="#00d4ff",width=2)))
    f1.update_layout(**DARK, title="Density Profile",
                     xaxis_title="ρ [kg/m³]", yaxis_title="Altitude [km]", height=280, margin=dict(t=40,b=40))

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=temps, y=alts/1000, name="Temperature",
                             line=dict(color="#ff6b35",width=2)))
    f2.update_layout(**DARK, title="Temperature Profile",
                     xaxis_title="T [K]", yaxis_title="Altitude [km]", height=280, margin=dict(t=40,b=40))

    metrics = [
        _metric("Surface ρ", f"{planet.density(0):.4f} kg/m³"),
        _metric("Surface T", f"{planet.temperature(0):.1f} K", "#ff6b35"),
        _metric("Surface P", f"{planet.pressure(0)/1e3:.2f} kPa", "#a8ff3e"),
        _metric("Gravity",   f"{planet.gravity_ms2:.3f} m/s²", "#9d60ff"),
    ]
    return f1, f2, metrics


# ════════════════════════════════════════════════════════════
# TAB 2 — Thermal
# ════════════════════════════════════════════════════════════

def layout_thermal():
    mats = [{"label":v.name,"value":k} for k,v in MATERIAL_DB.items()]
    return html.Div([
        _row(
            html.Div([
                _label("Material");    _dd("tps-mat", mats, "zylon"),
                _label("Thickness [mm]"); _sl("tps-thick", 1, 100, 15),
                _label("Entry velocity [km/s]"); _sl("tps-vel", 1, 12, 6, 0.5),
                _label("Density at entry [kg/m³]"); _sl("tps-rho", 1e-4, 0.1, 0.01, 1e-4),
                _label("Nose radius [m]"); _sl("tps-nose", 0.1, 5, 1.0, 0.1),
                _label("Duration [s]"); _sl("tps-dur", 10, 600, 200, 10),
            ]),
            html.Div([
                html.Div(id="tps-metrics", style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":8,"marginBottom":8}),
                dcc.Graph(id="tps-plot"),
            ]),
        )
    ])

@app.callback(
    Output("tps-plot","figure"), Output("tps-metrics","children"),
    Input("tps-mat","value"), Input("tps-thick","value"),
    Input("tps-vel","value"), Input("tps-rho","value"),
    Input("tps-nose","value"), Input("tps-dur","value"))
def update_tps(mat, thick_mm, vel_kms, rho, nose, dur):
    tps = ThermalProtectionSystem(mat, thick_mm/1000)
    v   = vel_kms * 1000
    q   = tps.sutton_graves_heating(rho, v, nose)
    t   = np.linspace(0, dur, 200)
    T   = tps.solve_1d_conduction(q, t, T_initial_K=250)
    exceeded, T_peak = tps.check_material_limit()
    sf  = tps.safety_margin()

    sf_col = "#a8ff3e" if sf >= 1.5 else "#ff4560"

    fig = go.Figure()
    for i in [0, 4, 9, 14, 19]:
        if i < T.shape[1]:
            depth = (i / max(T.shape[1]-1, 1)) * thick_mm
            fig.add_trace(go.Scatter(x=t, y=T[:,i], name=f"d={depth:.1f}mm",
                                      line=dict(width=1.5)))
    fig.add_hline(y=tps.mat.max_temperature_K, line_dash="dash",
                   line_color="#ff4560", annotation_text="T_limit")
    fig.update_layout(**DARK, title=f"TPS Temperature — {tps.mat.name}  ({thick_mm}mm)",
                      xaxis_title="Time [s]", yaxis_title="Temperature [K]", height=380)

    metrics = [
        _metric("Heat flux",  f"{q/1e6:.3f} MW/m²", "#ffd700"),
        _metric("T_peak",     f"{T_peak:.1f} K",     "#ff6b35" if exceeded else "#a8ff3e"),
        _metric("Safety Factor", f"{sf:.3f}",         sf_col),
    ]
    return fig, metrics


# ════════════════════════════════════════════════════════════
# TAB 3 — Canopy
# ════════════════════════════════════════════════════════════

def layout_canopy():
    shapes = ["elliptical","circular","rectangular","disk_gap_band","tricone"]
    return html.Div([
        _row(
            html.Div([
                _label("Shape"); _dd("can-shape",[{"label":s,"value":s} for s in shapes],"elliptical"),
                _label("Dimension A/R/W [m]"); _sl("can-a", 1, 30, 10),
                _label("Dimension B/H [m]");   _sl("can-b", 1, 30, 5),
                html.Div(id="can-metrics",
                         style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":8,"marginTop":12}),
            ]),
            html.Div([dcc.Graph(id="can-cross"), dcc.Graph(id="can-cd")]),
        )
    ])

@app.callback(
    Output("can-cross","figure"), Output("can-cd","figure"), Output("can-metrics","children"),
    Input("can-shape","value"), Input("can-a","value"), Input("can-b","value"))
def update_canopy(shape, a, b):
    dims = {"elliptical":{"a":a,"b":b}, "circular":{"r":a},
            "rectangular":{"width":a,"height":b},
            "disk_gap_band":{"r_disk":a,"r_band":a*1.2},
            "tricone":{"r_base":a}}.get(shape, {"a":a,"b":b})
    cg  = CanopyGeometry(shape, dims)
    x,y = cg.generate_cross_section(200)

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor="rgba(0,212,255,0.15)",
                             line=dict(color="#00d4ff",width=2)))
    f1.update_layout(**DARK, title="Canopy Cross-Section", xaxis_title="x [m]",
                     yaxis_title="y [m]", height=300, margin=dict(t=40,b=30))

    machs = np.linspace(0, 4, 200)
    cds   = [cg.calculate_drag_coefficient(m) for m in machs]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=machs, y=cds, line=dict(color="#ff6b35",width=2)))
    f2.add_vline(x=1.0, line_dash="dash", line_color="#888")
    f2.update_layout(**DARK, title="Cd vs Mach", xaxis_title="Mach",
                     yaxis_title="Cd", height=300, margin=dict(t=40,b=30))

    s = cg.summary()
    metrics = [
        _metric("Area",       f"{s['area_m2']:.2f} m²"),
        _metric("Diameter",   f"{s['nominal_diam_m']:.2f} m", "#a8ff3e"),
        _metric("Cd (M=0.1)", f"{s['Cd_subsonic']:.4f}", "#ff6b35"),
        _metric("Cd (M=1.0)", f"{s['Cd_mach1']:.4f}",   "#ffd700"),
    ]
    return f1, f2, metrics


# ════════════════════════════════════════════════════════════
# TAB 4 — CFD (LBM)
# ════════════════════════════════════════════════════════════

def layout_cfd():
    return html.Div([
        _card(
            html.H3("Lattice Boltzmann D2Q9 CFD", style={"color":"#00d4ff","marginTop":0}),
            _row(
                html.Div([
                    _label("Grid (Ny × Nx)");
                    dcc.RadioItems(id="lbm-res",
                        options=[{"label":"32×64","value":"32"},{"label":"48×96","value":"48"},{"label":"64×128","value":"64"}],
                        value="32", style={"color":"#c8d8f0"}),
                    _label("Reynolds number"); _sl("lbm-re", 10, 500, 100, 10),
                    _label("Steps"); _sl("lbm-steps", 100, 2000, 500, 100),
                    _label("Flow type");
                    dcc.RadioItems(id="lbm-flow",
                        options=[{"label":"Channel","value":"channel"},{"label":"Lid-driven","value":"lid_driven"}],
                        value="channel", style={"color":"#c8d8f0"}),
                    html.Br(),
                    html.Button("▶ Run LBM", id="lbm-btn",
                                style={"background":"#0d1f44","border":"1px solid #00d4ff",
                                       "color":"#00d4ff","borderRadius":6,"padding":"8px 16px","cursor":"pointer"}),
                ]),
                html.Div([
                    html.Div(id="lbm-metrics",
                             style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":8,"marginBottom":8}),
                    dcc.Graph(id="lbm-plot"),
                ]),
            ),
        )
    ])

@app.callback(
    Output("lbm-plot","figure"), Output("lbm-metrics","children"),
    Input("lbm-btn","n_clicks"),
    State("lbm-res","value"), State("lbm-re","value"),
    State("lbm-steps","value"), State("lbm-flow","value"),
    prevent_initial_call=True)
def run_lbm(n, res_str, Re, steps, flow):
    from src.lbm_solver import LBMSolver
    Ny = int(res_str); Nx = Ny * 2
    solver = LBMSolver((Ny, Nx), Re)
    solver.initialize()
    result = solver.solve(steps=int(steps), flow_type=flow, verbose=False)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=result["vorticity"],
        colorscale="RdBu", zmid=0,
        colorbar=dict(title="ω_z", tickfont=dict(color="#c8d8f0")),
    ))
    fig.update_layout(**DARK, title=f"Vorticity Field  Re={Re}  steps={result['step']}",
                      xaxis_title="x", yaxis_title="y", height=380)

    metrics = [
        _metric("Cd",        f"{result['Cd']:.4f}", "#ff6b35"),
        _metric("Cl",        f"{result['Cl']:.4f}", "#a8ff3e"),
        _metric("Converged", "Yes ✓" if result["converged"] else "No", "#a8ff3e" if result["converged"] else "#ff4560"),
    ]
    return fig, metrics


# ════════════════════════════════════════════════════════════
# TAB 5 — Neural Operators
# ════════════════════════════════════════════════════════════

def layout_nn():
    return html.Div([
        _card(
            html.H3("Neural Operator — Parametric EDL", style={"color":"#00d4ff","marginTop":0}),
            _row(
                html.Div([
                    _label("Operator type");
                    dcc.RadioItems(id="nn-type",
                        options=[{"label":"FNO","value":"fno"},{"label":"DeepONet","value":"deeponet"}],
                        value="fno", style={"color":"#c8d8f0"}),
                    _label("Training samples"); _sl("nn-n", 20, 200, 60, 10),
                    _label("Epochs"); _sl("nn-ep", 50, 500, 200, 50),
                    _label("Planet"); _dd("nn-planet",[{"label":p.title(),"value":p} for p in ["mars","venus","titan"]],"mars"),
                    html.Br(),
                    html.Button("▶ Train Operator", id="nn-btn",
                                style={"background":"#0d1f44","border":"1px solid #00d4ff",
                                       "color":"#00d4ff","borderRadius":6,"padding":"8px 16px","cursor":"pointer"}),
                ]),
                html.Div([
                    html.Div(id="nn-metrics",
                             style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":8,"marginBottom":8}),
                    dcc.Graph(id="nn-plot"),
                ]),
            ),
        )
    ])

@app.callback(
    Output("nn-plot","figure"), Output("nn-metrics","children"),
    Input("nn-btn","n_clicks"),
    State("nn-type","value"), State("nn-n","value"),
    State("nn-ep","value"), State("nn-planet","value"),
    prevent_initial_call=True)
def train_nn(n, op_type, n_samp, epochs, planet_name):
    from src.operator_dataset import OperatorDataset
    from src.neural_operator   import NeuralOperator

    ds   = OperatorDataset(output_resolution=50, planet_name=planet_name)
    data = ds.generate(n_samples=int(n_samp), verbose=False)

    X = data["inputs"].astype(np.float32)
    Y = data["outputs"].astype(np.float32)
    tr, te = ds.train_test_split(data, test_frac=0.2)

    op = NeuralOperator(op_type, n_in=1, n_out=1, modes=8, width=16)
    losses = op.train(tr["inputs"], tr["outputs"], epochs=int(epochs), verbose=False)

    y_pred = op.predict(te["inputs"])
    rmse = float(np.sqrt(((y_pred.flatten() - te["outputs"].flatten())**2).mean()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, name="Train loss", line=dict(color="#00d4ff",width=1.5)))
    fig.update_layout(**DARK, title=f"{op_type.upper()} Training Loss",
                      xaxis_title="Epoch", yaxis_title="MSE Loss (log)",
                      yaxis_type="log", height=380)

    metrics = [
        _metric("Test RMSE", f"{rmse:.4f}", "#ffd700" if rmse < 0.1 else "#ff6b35"),
        _metric("Backend", op._backend.upper(), "#a8ff3e"),
    ]
    return fig, metrics


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AeroDecel v6.0 Dashboard → http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
