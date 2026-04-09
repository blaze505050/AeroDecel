"""
app.py — AeroDecel v5.0 Streamlit Interactive Web Dashboard
Run: streamlit run app.py
Install: pip install streamlit
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Install: pip install streamlit")

st.set_page_config(page_title=f"AeroDecel v5.0", page_icon="🪂",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background:#080c14;color:#c8d8f0}
.mc{background:#0d1526;border:1px solid #1a2744;border-radius:10px;padding:14px;text-align:center;margin:3px}
.ml{font-size:10px;color:#556688;margin-bottom:3px}
.mv{font-size:24px;font-weight:600;color:#00d4ff}
.mu{font-size:10px;color:#445566}
h1,h2,h3{color:#c8d8f0!important}
.stButton>button{background:#0d1f44;border:1px solid #00d4ff;color:#00d4ff;border-radius:8px}
.aerodecel-badge{display:inline-block;background:linear-gradient(135deg,#00d4ff22,#9d60ff22);border:1px solid #00d4ff44;border-radius:6px;padding:4px 12px;font-size:11px;color:#00d4ff;margin-right:6px}
</style>""", unsafe_allow_html=True)

import config as cfg
from src.atmosphere import density, temperature, pressure, speed_of_sound
from src.calibrate_cd import _simulate, _logistic_A

with st.sidebar:
    st.markdown(f"## 🪂 AeroDecel v{cfg.AERODECEL_VERSION}")
    st.markdown(f"<span class='aerodecel-badge'>AI-Driven Aerodynamic Deceleration</span>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Mission Control", "📐 Design Calculator",
        "⚡ Opening Shock",   "🌡️ Atmosphere",
        "📊 Scenario Compare","🧮 Quick Calibrate",
        "🔬 Advanced Physics",
    ])
    st.markdown("---")
    st.markdown("### Physics Parameters")
    mass = st.slider("Mass [kg]",   10.0, 300.0, 80.0, 1.0)
    alt0 = st.slider("Alt₀ [m]",  100.0,5000.0,1000.0,50.0)
    v0   = st.slider("v₀ [m/s]",   5.0, 100.0,  25.0, 1.0)
    Cd   = st.slider("Cd",          0.3,   2.5,   1.35,0.05)
    Am   = st.slider("A_max [m²]",  5.0, 200.0,  50.0, 1.0)
    ti   = st.slider("t_infl [s]",  0.3,   8.0,   2.5, 0.1)

@st.cache_data
def sim(mass,alt0,v0,Cd,Am,ti):
    dt=0.05; ts,vs,hs,As,Ds=[],[],[],[],[]
    v_,h_,t_=float(v0),float(alt0),0.0
    while h_>0 and t_<600:
        A=_logistic_A(t_,Am,ti); rho=density(max(0,h_))
        drag=0.5*rho*v_**2*Cd*A
        dv=cfg.GRAVITY-drag/mass; dh=-v_
        v_=max(0,v_+dt*dv); h_=max(0,h_+dt*dh); t_+=dt
        ts.append(t_);vs.append(v_);hs.append(h_);As.append(A);Ds.append(drag)
        if h_<=0: break
    return np.array(ts),np.array(vs),np.array(hs),np.array(As),np.array(Ds)

def dark_fig(nrows=1,ncols=1,figsize=(14,5)):
    plt.rcParams.update({"figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0","axes.labelcolor":"#c8d8f0",
        "xtick.color":"#c8d8f0","ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9})
    fig,axes=plt.subplots(nrows,ncols,figsize=figsize,facecolor="#080c14")
    if nrows==1 and ncols==1: axes=[axes]
    for ax in np.array(axes).flat:
        ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3)
        ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
    return fig, axes

def mcard(label,val,unit=""):
    return f'<div class="mc"><div class="ml">{label}</div><div class="mv">{val}</div><div class="mu">{unit}</div></div>'

# ── MISSION CONTROL ────────────────────────────────────────────────────────────
if "Mission" in page:
    st.markdown("# 🪂 AeroDecel — Mission Control")
    ts,vs,hs,As,Ds = sim(mass,alt0,v0,Cd,Am,ti)
    v_t=float(vs[-1]); t_l=float(ts[-1]); pk=float(Ds.max())

    cols=st.columns(5)
    for col,(lbl,v,u) in zip(cols,[("Terminal v",f"{v_t:.2f}","m/s"),
        ("Descent time",f"{t_l:.1f}","s"),("Peak drag",f"{pk:.0f}","N"),
        ("Weight",f"{mass*cfg.GRAVITY:.0f}","N"),("v_term analytic",
        f"{float(np.sqrt(2*mass*cfg.GRAVITY/(density(0)*Cd*Am))):.2f}","m/s")]):
        col.markdown(mcard(lbl,v,u),unsafe_allow_html=True)

    st.markdown("---")
    fig,axes=dark_fig(2,3,(16,8))
    axes=np.array(axes).flat
    ax=next(axes); ax.fill_between(ts,vs,alpha=0.15,color="#00d4ff"); ax.plot(ts,vs,color="#00d4ff",lw=2); ax.axhline(v_t,color="#a8ff3e",lw=1,ls="--"); ax.set_title("Velocity v(t)",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("v [m/s]")
    ax=next(axes); ax.fill_between(ts,hs,alpha=0.15,color="#9d60ff"); ax.plot(ts,hs,color="#9d60ff",lw=2); ax.set_title("Altitude h(t)",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("h [m]")
    ax=next(axes); ax.fill_between(ts,As,alpha=0.15,color="#ffd700"); ax.plot(ts,As,color="#ffd700",lw=2); ax.axhline(Am,color="#888",lw=0.7,ls=":"); ax.set_title("Canopy Area A(t)",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("A [m²]")
    ax=next(axes); ax.fill_between(ts,Ds/1e3,alpha=0.15,color="#ff4560"); ax.plot(ts,Ds/1e3,color="#ff4560",lw=2); ax.axhline(mass*cfg.GRAVITY/1e3,color="#888",lw=0.8,ls=":"); ax.set_title("Drag Force [kN]",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("F [kN]")
    ax=next(axes); rho_a=np.array([density(max(0,h)) for h in hs]); q=0.5*rho_a*vs**2; ax.plot(ts,q,color="#ff6b35",lw=2); ax.fill_between(ts,q,alpha=0.15,color="#ff6b35"); ax.set_title("Dynamic Pressure",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("q [Pa]")
    ax=next(axes); KE=0.5*mass*vs**2/1e3; PE=mass*cfg.GRAVITY*hs/1e3; ax.fill_between(ts,KE,alpha=0.2,color="#00d4ff",label="KE"); ax.fill_between(ts,PE,alpha=0.2,color="#9d60ff",label="PE"); ax.plot(ts,KE,color="#00d4ff",lw=1.5); ax.plot(ts,PE,color="#9d60ff",lw=1.5); ax.plot(ts,KE+PE,color="#ffd700",lw=1.2,ls="--",label="Total"); ax.set_title("Energy Budget [kJ]",fontweight="bold"); ax.set_xlabel("Time [s]"); ax.set_ylabel("E [kJ]"); ax.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("#### Phase Portrait")
    fig2,ax2=plt.subplots(figsize=(6,4),facecolor="#080c14")
    ax2.set_facecolor("#0d1526"); ax2.tick_params(colors="#c8d8f0"); ax2.spines[:].set_color("#2a3d6e"); ax2.grid(True,alpha=0.3)
    sc=ax2.scatter(vs,hs,c=ts,cmap="plasma",s=3,alpha=0.8)
    fig2.colorbar(sc,ax=ax2,label="t [s]")
    ax2.set_xlabel("v [m/s]"); ax2.set_ylabel("h [m]"); ax2.set_title("Phase Portrait",fontweight="bold")
    st.pyplot(fig2,use_container_width=False); plt.close(fig2)

# ── DESIGN CALCULATOR ─────────────────────────────────────────────────────────
elif "Design" in page:
    st.markdown("# 📐 AeroDecel Design Calculator")
    from scipy.optimize import brentq
    c1,c2=st.columns([1,2])
    with c1:
        tv=st.slider("Target v [m/s]",2.0,15.0,5.0,0.1)
        dm=st.slider("Mass [kg]",10.0,300.0,mass,1.0)
        da=st.slider("Alt₀ [m]",100.0,5000.0,alt0,50.0)
        dc=st.slider("Cd",0.3,2.5,Cd,0.05)
        dt_=st.slider("t_infl [s]",0.3,8.0,ti,0.1)
    with c2:
        def obj(A): return _simulate(Cd=dc,mass=dm,alt0=da,v0=v0,Am=A,ti=dt_,dt=0.1)["landing_velocity"]-tv
        try:
            vlo,vhi=obj(1)+tv,obj(400)+tv
            if vlo>tv>vhi:
                Asol=brentq(obj,1,400,xtol=0.05,maxiter=80)
                rv=_simulate(Cd=dc,mass=dm,alt0=da,v0=v0,Am=Asol,ti=dt_,dt=0.05)
                D=float(np.sqrt(4*Asol/np.pi))
                st.markdown(f"""<div style='background:#0d1526;border:1px solid #00d4ff;border-radius:10px;padding:20px'>
<h3 style='color:#00d4ff'>Solution ✓</h3>
<table style='width:100%;font-family:monospace;font-size:13px'>
<tr><td style='color:#556688'>A_inf</td><td style='color:#c8d8f0;text-align:right'><b>{Asol:.2f} m²</b></td></tr>
<tr><td style='color:#556688'>Diameter D₀</td><td style='color:#c8d8f0;text-align:right'>{D:.2f} m</td></tr>
<tr><td style='color:#556688'>Actual v_land</td><td style='color:#00d4ff;text-align:right'><b>{rv["landing_velocity"]:.4f} m/s</b></td></tr>
<tr><td style='color:#556688'>Pack volume</td><td style='color:#c8d8f0;text-align:right'>{9e-3*Asol**1.5*1000:.2f} L</td></tr>
<tr><td style='color:#556688'>Pack weight</td><td style='color:#c8d8f0;text-align:right'>{Asol*44/1000*1.15:.2f} kg</td></tr>
</table></div>""",unsafe_allow_html=True)
                Aarr=np.linspace(max(1,Asol*0.3),Asol*2.5,40)
                varr=[_simulate(Cd=dc,mass=dm,alt0=da,v0=v0,Am=A_,ti=dt_,dt=0.15)["landing_velocity"] for A_ in Aarr]
                fig,ax=plt.subplots(figsize=(8,4),facecolor="#080c14"); ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
                ax.plot(Aarr,varr,color="#00d4ff",lw=2); ax.axhline(tv,color="#a8ff3e",lw=1.5,ls="--",label=f"Target {tv}m/s"); ax.axvline(Asol,color="#ff6b35",lw=1.5,ls="--",label=f"A={Asol:.1f}m²"); ax.scatter([Asol],[rv["landing_velocity"]],s=100,color="#ffd700",zorder=5); ax.set_xlabel("A_inf [m²]"); ax.set_ylabel("v_land [m/s]"); ax.set_title("v_land vs area",fontweight="bold"); ax.legend(fontsize=9)
                st.pyplot(fig,use_container_width=True); plt.close(fig)
            else:
                st.warning(f"v_land range [{min(vlo,vhi):.2f},{max(vlo,vhi):.2f}] m/s — adjust params")
        except Exception as e: st.error(str(e))

# ── OPENING SHOCK ─────────────────────────────────────────────────────────────
elif "Shock" in page:
    st.markdown("# ⚡ Opening Shock — MIL-HDBK-1791")
    c1,c2=st.columns([1,2])
    with c1:
        sv=st.slider("v_deploy [m/s]",5.0,80.0,float(v0),0.5)
        sh=st.slider("h_deploy [m]",100.0,5000.0,float(alt0),50.0)
        sm=st.slider("Mass [kg]",10.0,300.0,float(mass),1.0)
        sA=st.slider("A_inf [m²]",1.0,200.0,float(Am),1.0)
        sCd=st.slider("Cd",0.3,2.5,float(Cd),0.05)
        sti=st.slider("t_infl [s]",0.3,8.0,float(ti),0.1)
        sct=st.selectbox("Canopy type",["flat_circular","ribbon","drogue","ram_air","conical"])
    with c2:
        from src.opening_shock import analyse as sa
        r=sa(v_deploy=sv,h_deploy=sh,mass=sm,A_inf=sA,Cd=sCd,t_infl=sti,canopy_type=sct,verbose=False)
        d=r.to_dict(); cc="#a8ff3e" if d["compliant"] else "#ff4560"; ct="COMPLIANT ✓" if d["compliant"] else "NON-COMPLIANT ✗"
        st.markdown(f"""<div style='background:#0d1526;border:1px solid {cc};border-radius:10px;padding:20px;margin-bottom:12px'>
<h3 style='color:{cc};margin:0 0 10px'>{ct}</h3>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-family:monospace;font-size:13px'>
<div><span style='color:#556688'>F_steady </span><span style='color:#c8d8f0'>{d["F_steady_N"]/1e3:.3f} kN</span></div>
<div><span style='color:#556688'>F_peak </span><span style='color:{cc};font-weight:bold'>{d["F_peak_N"]/1e3:.3f} kN</span></div>
<div><span style='color:#556688'>CLA used </span><span style='color:#ffd700;font-weight:bold'>{d["CLA_used"]:.4f}</span></div>
<div><span style='color:#556688'>Min SF </span><span style='color:{cc};font-weight:bold'>{d["minimum_sf"]:.3f}</span></div>
</div></div>""",unsafe_allow_html=True)
        rows=[{"": "✓" if c["status"]=="OK" else "⚠" if c["status"]=="WARNING" else "✗",
               "Component":c["component"],"Rated [N]":f'{c["rated_N"]:,.0f}',
               "SF":f'{c["safety_factor"]:.2f}',"Status":c["status"]}
              for c in d["structural_components"]]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        df_s=r.df; fig,ax=plt.subplots(figsize=(9,4),facecolor="#080c14"); ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
        ax.fill_between(df_s.time_s,df_s.F_shock_N/1e3,alpha=0.2,color="#ff4560"); ax.plot(df_s.time_s,df_s.F_shock_N/1e3,color="#ff4560",lw=2,label="F_shock"); ax.plot(df_s.time_s,df_s.F_steady_N/1e3,color="#00d4ff",lw=1.2,ls="--",label="F_steady"); ax.axvline(sti,color="#ffd700",lw=0.9,ls=":"); ax.set_xlabel("Time [s]"); ax.set_ylabel("Force [kN]"); ax.set_title("Opening shock force history",fontweight="bold"); ax.legend(fontsize=9)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

# ── ATMOSPHERE ────────────────────────────────────────────────────────────────
elif "Atmosphere" in page:
    st.markdown("# 🌡️ ISA Atmosphere Explorer")
    alts=np.linspace(0,20000,400)
    T_a=np.array([temperature(h) for h in alts]); P_a=np.array([pressure(h) for h in alts])/1000; R_a=np.array([density(h) for h in alts])
    fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor="#080c14")
    for ax,(data,xl,color,title) in zip(axes,[
        (T_a,"T [K]","#ff6b35","Temperature"),(P_a,"P [kPa]","#00d4ff","Pressure"),(R_a,"ρ [kg/m³]","#a8ff3e","Density")]):
        ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
        ax.plot(data,alts/1000,color=color,lw=2); ax.fill_betweenx(alts/1000,data,alpha=0.1,color=color)
        ax.axhline(11,color="#888",lw=0.7,ls="--",alpha=0.6); ax.text(data.min()*1.02,11.3,"Tropopause",fontsize=7,color="#888")
        ax.set_xlabel(xl); ax.set_ylabel("Alt [km]"); ax.set_title(title,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.dataframe(pd.DataFrame([{"Alt[m]":h,"T[K]":round(temperature(h),2),"P[Pa]":round(pressure(h),1),"ρ[kg/m³]":round(density(h),5),"a[m/s]":round(speed_of_sound(h),2)} for h in [0,500,1000,2000,5000,8000,11000,15000]]),use_container_width=True,hide_index=True)

# ── SCENARIO COMPARE ──────────────────────────────────────────────────────────
elif "Scenario" in page:
    st.markdown("# 📊 Scenario Comparator")
    n=st.slider("Scenarios",2,4,3)
    defaults=[(80,1.35,50),(100,1.0,40),(60,1.6,60),(120,0.8,35)]
    scens=[]; cols=st.columns(n)
    for i,col in enumerate(cols[:n]):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            m_=st.number_input(f"Mass {i+1}",10.0,300.0,float(defaults[i][0]),1.0,key=f"m{i}")
            c_=st.number_input(f"Cd {i+1}",0.3,2.5,float(defaults[i][1]),0.05,key=f"c{i}")
            a_=st.number_input(f"Area {i+1}",5.0,200.0,float(defaults[i][2]),1.0,key=f"a{i}")
            scens.append((m_,c_,a_,f"S{i+1}: m={m_:.0f} Cd={c_:.2f} A={a_:.0f}m²"))
    colors=["#00d4ff","#ff6b35","#a8ff3e","#ffd700"]
    fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor="#080c14")
    for ax in axes:
        ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
    rows=[]
    for i,(m_,c_,a_,lbl) in enumerate(scens[:n]):
        ts_,vs_,hs_,As_,Ds_=sim(m_,alt0,v0,c_,a_,ti); col=colors[i]
        axes[0].plot(ts_,vs_,color=col,lw=1.8,label=lbl); axes[1].plot(ts_,hs_,color=col,lw=1.8,label=lbl); axes[2].plot(vs_,hs_,color=col,lw=1.8,label=lbl)
        rows.append({"Scenario":lbl,"v_term[m/s]":round(float(vs_[-1]),2),"t[s]":round(float(ts_[-1]),1),"Peak F[N]":round(float(Ds_.max()),0)})
    axes[0].set_xlabel("t[s]"); axes[0].set_ylabel("v[m/s]"); axes[0].set_title("v(t)",fontweight="bold"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("t[s]"); axes[1].set_ylabel("h[m]"); axes[1].set_title("h(t)",fontweight="bold"); axes[1].legend(fontsize=7)
    axes[2].set_xlabel("v[m/s]"); axes[2].set_ylabel("h[m]"); axes[2].set_title("Phase Portrait",fontweight="bold"); axes[2].legend(fontsize=7)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ── QUICK CALIBRATE ───────────────────────────────────────────────────────────
elif "Calibrate" in page:
    st.markdown("# 🧮 Quick Cd Calibration")
    c1,c2=st.columns(2)
    with c1:
        ov=st.number_input("Observed v_land [m/s]",1.0,30.0,6.2,0.1)
        cm=st.slider("Mass [kg]",10.0,300.0,float(mass),1.0)
        ca=st.slider("Alt₀ [m]",100.0,5000.0,float(alt0),50.0)
        cA=st.slider("A_max [m²]",5.0,200.0,float(Am),1.0)
        ct=st.slider("t_infl [s]",0.3,8.0,float(ti),0.1)
        go=st.button("🔍 Back-solve Cd",use_container_width=True)
    with c2:
        if go:
            from src.calibrate_cd import calibrate_from_landing_velocity
            try:
                r=calibrate_from_landing_velocity(ov,mass=cm,alt0=ca,Am=cA,ti=ct,n_bootstrap=150,verbose=False)
                st.markdown(f"""<div style='background:#0a1830;border:1px solid #00d4ff;border-radius:10px;padding:20px'>
<h3 style='color:#00d4ff'>Cd = {r["Cd_eff"]:.5f}</h3>
<p style='color:#c8d8f0'>95% CI: [{r["Cd_ci_low"]:.4f}, {r["Cd_ci_high"]:.4f}]</p>
<p style='color:#c8d8f0'>± {r["Cd_ci_half_width"]:.4f}</p>
<p style='color:#556688'>Solve: {r["solve_time_ms"]:.1f} ms · {r["n_brent_evals"]} evals · residual {r["residual_ms"]:+.2e} m/s</p>
</div>""",unsafe_allow_html=True)
                boot=np.array(r["bootstrap_samples"])
                fig,ax=plt.subplots(figsize=(7,4),facecolor="#080c14"); ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
                ax.hist(boot,bins=30,color="#00d4ff",alpha=0.7,edgecolor="none",density=True)
                ax.axvline(r["Cd_eff"],color="#a8ff3e",lw=2,label=f"Cd={r['Cd_eff']:.4f}"); ax.axvline(r["Cd_ci_low"],color="#ff6b35",lw=1.2,ls="--"); ax.axvline(r["Cd_ci_high"],color="#ff6b35",lw=1.2,ls="--",label="95% CI"); ax.axvspan(r["Cd_ci_low"],r["Cd_ci_high"],alpha=0.12,color="#00d4ff")
                ax.set_xlabel("Cd"); ax.set_ylabel("Density"); ax.set_title("Bootstrap posterior",fontweight="bold"); ax.legend(fontsize=9)
                st.pyplot(fig,use_container_width=True); plt.close(fig)
            except Exception as e: st.error(str(e))

# ── ADVANCED PHYSICS (AeroDecel v5.0) ─────────────────────────────────────────
elif "Advanced" in page:
    st.markdown("# 🔬 Advanced Physics — AeroDecel v5.0")
    st.markdown("""
    <div style='background:#0d1526;border:1px solid #00d4ff44;border-radius:10px;padding:16px;margin-bottom:16px'>
    <p style='color:#00d4ff;font-size:13px;margin-bottom:8px'><b>AeroDecel Correction Pipeline</b></p>
    <p style='color:#8899bb;font-size:12px;line-height:1.6'>
    Cd_eff = Cd₀ × f_Mach(M) × f_Re(Re) × f_porosity(v)<br>
    m_eff = m + C_a · ρ · V_canopy  (added mass)<br>
    F_buoy = ρ · g · V_canopy  (buoyancy)
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        adv_v = st.slider("Velocity [m/s]", 2.0, 80.0, 25.0, 0.5, key="adv_v")
        adv_h = st.slider("Altitude [m]", 0.0, 5000.0, 1000.0, 50.0, key="adv_h")
        adv_Cd0 = st.slider("Cd₀ (baseline)", 0.3, 2.5, 1.35, 0.05, key="adv_cd")
        adv_D = st.slider("Canopy diameter [m]", 2.0, 15.0, 8.0, 0.5, key="adv_D")
        adv_kp = st.slider("Porosity k_p", 0.0, 0.03, 0.012, 0.001, key="adv_kp")
        do_mach = st.checkbox("Mach correction (Prandtl-Glauert)", True)
        do_re = st.checkbox("Reynolds correction (Knacke 1992)", True)
        do_por = st.checkbox("Porosity correction (Pflanz 1952)", True)

    with c2:
        from src.atmosphere import density as rho_fn, speed_of_sound as a_fn, dynamic_viscosity as mu_fn

        rho = rho_fn(adv_h)
        a = a_fn(adv_h)
        mu = mu_fn(adv_h)
        M = adv_v / max(a, 1.0)
        Re = rho * adv_v * adv_D / max(mu, 1e-10)

        Cd = adv_Cd0
        corr_m = corr_r = corr_p = 1.0

        if do_mach and M > 0.05:
            M_c = min(M, 0.79)
            corr_m = 1.0 / max(np.sqrt(1.0 - M_c**2), 0.01)
            Cd *= corr_m

        if do_re and Re > 0:
            log_Re = np.log10(max(Re, 1e3))
            if log_Re < 4.5: corr_r = 1.05
            elif log_Re < 5.0: corr_r = 1.0 + 0.05 * (5.0 - log_Re) / 0.5
            elif log_Re < 5.5: corr_r = 1.0 - 0.08 * (log_Re - 5.0) / 0.5
            elif log_Re < 6.0: corr_r = 0.92 + 0.06 * (log_Re - 5.5) / 0.5
            else: corr_r = 0.98
            Cd *= corr_r

        if do_por and adv_kp > 0:
            corr_p = max(0.05, 1.0 - adv_kp * adv_v)
            Cd *= corr_p

        total_corr = corr_m * corr_r * corr_p

        cols = st.columns(4)
        cols[0].markdown(mcard("Cd_effective", f"{Cd:.5f}", ""), unsafe_allow_html=True)
        cols[1].markdown(mcard("Mach", f"{M:.5f}", f"a={a:.1f} m/s"), unsafe_allow_html=True)
        cols[2].markdown(mcard("Reynolds", f"{Re:.0f}", f"D={adv_D}m"), unsafe_allow_html=True)
        cols[3].markdown(mcard("Total corr.", f"{total_corr:.5f}", "multiplier"), unsafe_allow_html=True)

        # Correction breakdown bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#080c14")

        ax = axes[0]
        ax.set_facecolor("#0d1526"); ax.grid(True, alpha=0.3); ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
        labels = ["Mach (P-G)", "Reynolds", "Porosity", "Total"]
        values = [corr_m, corr_r, corr_p, total_corr]
        colors_bar = ["#ff6b35", "#a8ff3e", "#9d60ff", "#ffd700"]
        bars = ax.bar(labels, values, color=colors_bar, alpha=0.8, edgecolor="none")
        ax.axhline(1.0, color="#888", lw=0.8, ls=":")
        ax.set_ylabel("Correction factor")
        ax.set_title("Individual correction factors", fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, color="#c8d8f0")

        # Cd vs velocity sweep
        ax2 = axes[1]
        ax2.set_facecolor("#0d1526"); ax2.grid(True, alpha=0.3); ax2.tick_params(colors="#c8d8f0"); ax2.spines[:].set_color("#2a3d6e")
        vs_sweep = np.linspace(2, 80, 100)
        Cd_sweep = []
        for v_s in vs_sweep:
            cd_s = adv_Cd0
            M_s = v_s / max(a, 1.0)
            Re_s = rho * v_s * adv_D / max(mu, 1e-10)
            if do_mach and M_s > 0.05:
                cd_s *= 1.0 / max(np.sqrt(1.0 - min(M_s, 0.79)**2), 0.01)
            if do_re:
                lr = np.log10(max(Re_s, 1e3))
                if lr < 4.5: cr = 1.05
                elif lr < 5.0: cr = 1.0 + 0.05*(5.0-lr)/0.5
                elif lr < 5.5: cr = 1.0 - 0.08*(lr-5.0)/0.5
                elif lr < 6.0: cr = 0.92 + 0.06*(lr-5.5)/0.5
                else: cr = 0.98
                cd_s *= cr
            if do_por: cd_s *= max(0.05, 1.0 - adv_kp * v_s)
            Cd_sweep.append(cd_s)
        ax2.plot(vs_sweep, Cd_sweep, color="#00d4ff", lw=2, label="Cd_eff(v)")
        ax2.axhline(adv_Cd0, color="#888", lw=0.9, ls="--", label=f"Cd₀={adv_Cd0}")
        ax2.axvline(adv_v, color="#ffd700", lw=1.2, ls=":", alpha=0.7, label=f"v={adv_v} m/s")
        ax2.scatter([adv_v], [Cd], s=80, color="#ffd700", zorder=5)
        ax2.set_xlabel("Velocity [m/s]"); ax2.set_ylabel("Cd_eff")
        ax2.set_title("Corrected Cd vs velocity", fontweight="bold")
        ax2.legend(fontsize=8)

        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

st.markdown("---")
st.markdown(f"<p style='text-align:center;color:#334455;font-size:10px'>AeroDecel v{cfg.AERODECEL_VERSION} · {cfg.AERODECEL_EQ} · Zero cost · All open source</p>",unsafe_allow_html=True)
