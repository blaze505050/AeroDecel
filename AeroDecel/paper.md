---
title: 'AeroDecel: An Open-Source Multi-Fidelity Framework for Planetary Entry, Descent, and Landing Simulation'
tags:
  - Python
  - aerospace engineering
  - entry descent and landing
  - computational fluid dynamics
  - Monte Carlo simulation
  - hypersonic aerothermodynamics
  - physics-informed machine learning
authors:
  - name: AeroDecel Contributors
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 16 April 2026
bibliography: paper.bib
---

# Summary

AeroDecel is a comprehensive, open-source Python framework for simulating
planetary Entry, Descent, and Landing (EDL) trajectories with support for
multi-body 6-DOF dynamics, real-gas thermochemistry, ablative thermal
protection, aeroelastic flutter analysis, and lattice Boltzmann turbulence
modelling. The framework couples high-fidelity physics modules — including
Park 1993 CO$_2$ dissociation kinetics, Amar ablation modelling, and
modified Newtonian aerodynamics — with machine-learning surrogates such as
normalising flows, Fourier neural operators, graph neural networks, and
Gaussian process emulators. AeroDecel enables researchers and engineers to
perform end-to-end EDL mission analysis, from atmospheric entry through
parachute deployment to powered terminal descent, entirely in Python with
zero proprietary dependencies.

# Statement of Need

High-fidelity EDL simulation has traditionally been confined to proprietary
tools developed by major aerospace contractors and national space agencies.
Codes such as POST II (NASA), DSENDS (JPL), and LAURA (Langley) are either
export-controlled, require expensive licences, or lack the modularity
needed for rapid prototyping of novel mission concepts. Open-source
alternatives do exist, but they are typically fragmented: a 3-DOF
trajectory integrator here, a standalone ablation solver there, with no
unified framework that spans the full EDL sequence from hypersonic entry
interface to touchdown.

AeroDecel addresses this gap by providing a single, pip-installable Python
package that integrates 28 features across five tiers — physics, machine
learning, systems engineering, visualisation, and software infrastructure
— under a consistent API. The framework is designed for three primary
audiences:

1. **Academic researchers** who need a transparent, modifiable codebase for
   validating new EDL guidance, aerothermodynamic, or TPS models against
   heritage mission data (e.g., Mars 2020 Perseverance).
2. **Aerospace engineering students** who require an accessible tool for
   coursework and thesis projects without navigating export-control
   barriers.
3. **Small-satellite and commercial teams** who need rapid EDL feasibility
   assessment without committing to heavyweight proprietary toolchains.

The framework has been validated against reconstructed Perseverance EDL
flight data, achieving $R^2 > 0.999$ on the velocity–altitude profile
after automated drag-coefficient calibration. Its modular architecture
— documented in `CONTRIBUTING.md` with clear extension patterns for
adding planets, TPS materials, and ML modules — lowers the barrier to
community contribution and reproducible EDL research.

# Core Physics and Computational Models

AeroDecel implements the following validated physics and computational
models:

## Trajectory Dynamics

- **13-state quaternion 6-DOF propagator** with Baumgarte constraint
  stabilisation and aerodynamic stability derivatives ($C_{m\alpha}$,
  $C_{mq}$, $C_{n\beta}$, $C_{lp}$) for coupled translational–rotational
  dynamics [@zipfel2007modeling].
- **3-DOF point-mass integrator** used for Monte Carlo ensemble
  propagation with Latin Hypercube Sampling across 11 uncertain parameters
  [@giordano2005monte].

## Aerothermodynamics

- **Real-gas thermochemistry** using Park 1993 two-temperature CO$_2$
  dissociation kinetics with Gibbs free-energy equilibrium composition,
  yielding effective specific heat ratio $\gamma_{\mathrm{eff}}$ that
  drops from 1.28 to approximately 1.05 at peak heating
  [@park1993review].
- **Convective heating** via the Sutton–Graves engineering correlation and
  the Fay–Riddell stagnation-point formulation [@sutton1971; @fay1958].
- **Modified Newtonian aerodynamics** for generating $C_D$, $C_L$, and
  $C_m$ databases over angle-of-attack and Mach number grids
  [@anderson2006hypersonic].

## Thermal Protection

- **Amar ablation model** coupling Arrhenius pyrolysis kinetics with
  surface thermochemistry, Mickley–Davis blowing-correction ($B'$ tables),
  and a moving-boundary finite-difference thermal solver
  [@amar2006one].
- **1-D transient conduction** through multi-material TPS stackups with
  surface re-radiation boundary conditions.

## Structural and Fluid Dynamics

- **Aeroelastic flutter analysis** using CST-parameterised triangular
  finite-element membrane models with eigensolution-based modal analysis
  and Newmark-$\beta$ time-domain integration [@dowell2015modern].
- **D3Q19 lattice Boltzmann method** with Smagorinsky sub-grid-scale
  turbulence modelling for three-dimensional wake-flow simulation
  [@kruger2017lattice].

## Machine Learning Surrogates

- **Real-NVP normalising flows** with physics-informed ODE loss penalties
  for full posterior trajectory prediction $P(\mathbf{x}|\theta)$
  [@rezende2015variational].
- **Fourier neural operator** trained across Mars, Venus, and Titan
  atmospheres with planet-embedding vectors enabling zero-shot
  generalisation [@li2021fourier].
- **Message-passing graph neural network** for per-panel canopy stress
  prediction and safety-factor estimation [@gilmer2017neural].
- **Gaussian process emulator** with Expected Improvement Bayesian
  optimisation for minimum-mass TPS design in approximately 50 function
  evaluations [@rasmussen2006gaussian].
- **Extended Kalman filter** coupled with a forward ODE model for
  real-time streaming estimation of drag coefficient with Joseph
  covariance update [@simon2006optimal].

## Systems Engineering

- **NSGA-II multi-objective optimiser** for Pareto-optimal TPS design
  trading mass, safety factor, and cost [@deb2002fast].
- **Fault tree analysis** with AND/OR gates, Birnbaum and
  Fussell–Vesely importance measures, minimal cut-set enumeration, and
  Monte Carlo uncertainty quantification [@vesely1981fault].
- **Multi-stage EDL sequencer** implementing the four-phase Perseverance
  profile: guided entry, heatshield jettison, disk-gap-band parachute
  deployment, and throttled powered descent [@way2006mars].

# Acknowledgements

The authors thank the open-source scientific Python community — in
particular the developers of NumPy, SciPy, and Matplotlib — whose tools
form the computational backbone of AeroDecel.

# References
