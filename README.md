<p align="center">
  <img src="Banner-7.PNG"
       alt="Directed Energy Engagement Digital Twin"
       width="100%">
</p>

<h1 align="center">Directed Energy Engagement Digital Twin</h1>

<p align="center">
  <strong>Physics-Informed Aerospace Simulation • Digital Engineering • Autonomous Systems • State Estimation • Uncertainty Quantification • V&V</strong>
</p>

<p align="center">
  <strong>MODEL. SIMULATE. VALIDATE. DOMINATE.</strong>
</p>

---

## Software Stack

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="52" alt="Python" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="52" alt="NumPy" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="52" alt="Pandas" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" height="52" alt="Matplotlib" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/plotly/plotly-original.svg" height="52" alt="Plotly" />
</p>

<p align="center">
  <strong>Python • Streamlit • NumPy • Pandas • Matplotlib • Plotly</strong>
</p>

<p align="center">
  Aerospace Dynamics • Digital Twin • EKF State Estimation • Monte Carlo UQ •
  Thermal/Power Modeling • Systems Engineering • Verification & Validation
</p>

---

## Overview

The **Directed Energy Engagement Digital Twin** is a Python-based aerospace digital engineering application for modeling, simulating, visualizing, and evaluating a complex closed-loop engagement system.

The application brings together **aerospace dynamics, state estimation, atmospheric modeling, beam propagation, power and thermal behavior, Monte Carlo uncertainty quantification, requirements traceability, verification, and reference benchmarking** within a single interactive engineering environment.

Rather than treating these disciplines as independent calculations, the digital twin connects them through a common system state so that changes in the environment, target trajectory, sensor performance, pointing uncertainty, available power, or thermal condition can propagate through the model and influence downstream engineering outputs.

The result is a lightweight example of how **physics-informed modeling, digital engineering, autonomous-system architecture, and systems engineering** can be integrated into an interactive Python application.

---

## Why This Project Matters

Modern aerospace and autonomous systems operate at the intersection of **physics, sensing, computation, uncertainty, resource constraints, and decision-making**.

Engineering these systems requires more than optimizing individual components. Engineers must understand how subsystem behavior propagates through the larger system and affects mission-level performance.

This application explores that problem through a closed-loop digital twin:

**Sense → Estimate → Decide → Act → Assess → Re-engage**

The architecture is relevant to engineering problems involving:

- Uncrewed aircraft systems (UAS)
- Autonomous aerospace systems
- Aerospace digital twins
- Mission and engagement simulation
- Guidance, navigation, and control
- Sensor fusion and state estimation
- Tracking and perception systems
- Model-based systems engineering
- Power and thermal management
- Uncertainty quantification
- Simulation-based V&V
- Resilient autonomous systems
- AI-enabled engineering decision support

The directed-energy scenario provides a demanding multidisciplinary problem through which these capabilities can be developed and demonstrated.

---

## Engineering Value

The primary value of the project is **multidisciplinary model integration**.

A change in one part of the simulated system can influence multiple downstream engineering quantities.

For example:

**Environment**

↓  

**Atmospheric transmission**

↓  

**Delivered optical power**

↓  

**Target irradiance**

↓  

**Thermal response**

↓  

**Estimated Thermal Effect Index**

At the same time:

**Target motion**

↓  

**Sensor geometry**

↓  

**State estimation**

↓  

**Track quality and uncertainty**

↓  

**Pointing solution**

↓  

**Engagement decision support**

This provides a computational foundation for:

- architecture evaluation
- concept exploration
- requirements analysis
- engineering trade studies
- sensitivity analysis
- uncertainty characterization
- subsystem interaction analysis
- model verification
- reference benchmarking
- digital engineering experimentation
- autonomy research
- rapid engineering prototyping

The architecture can also be adapted to aerospace and robotic applications well beyond the directed-energy scenario.

---

# Closed-Loop Engagement Architecture

The application models an eight-stage engagement process:

### 1. Detect
Search sensors identify potential targets.

### 2. Identify
Measurements support classification and prioritization.

### 3. Track
State estimation maintains a continuously updated target track.

### 4. Decide
Engagement logic evaluates geometry, timing, uncertainty, system state, and constraints.

### 5. Point / Aim
Beam geometry and pointing uncertainty are evaluated.

### 6. Deliver Energy
Available power, atmospheric transmission, and beam spreading determine delivered energy.

### 7. Assess Effect
Reduced-order thermal response and engineering effect metrics are evaluated.

### 8. Re-engage
Updated system state feeds the next decision cycle.

This architecture provides a compact implementation of the:

**Sense → Estimate → Decide → Act → Assess**

pattern found throughout autonomous aerospace and robotic systems.

---

# Core Engineering Capabilities

## ✈️ Aerospace Dynamics

The application includes both **3-DOF kinematic** and **generic 6-DOF rigid-body** modeling capabilities.

The higher-order aerospace model incorporates:

- translational dynamics
- rigid-body rotational dynamics
- quaternion attitude representation
- body/inertial coordinate transformations
- trajectory propagation
- forces and moments
- reduced-order aerodynamic effects

NASA Generic Transport Model data can be used for **model-to-model flight-dynamics reference benchmarking**.

The aerospace models are intentionally generic and reduced-order rather than representations of a specific operational aircraft or missile.

---

## 🎯 State Estimation and Tracking

The application implements measurement-driven state estimation using Kalman-filter methods.

Engineering outputs include:

- estimated target state
- state covariance
- measurement residuals
- track-quality metrics
- line-of-sight geometry
- angular uncertainty
- closest-point-of-approach calculations
- NEES/NIS consistency diagnostics

These capabilities are directly transferable to engineering problems involving:

**UAS autonomy • robotics • GNSS-denied navigation • sensor fusion • target tracking • perception pipelines**

---

## 🌎 Atmospheric Modeling

Environmental modeling includes:

- standard-atmosphere relationships
- visibility-dependent aerosol extinction
- Rayleigh scaling
- generic humidity effects
- reduced-order atmospheric transmission
- configurable turbulence/spreading effects

**NOAA IGRA radiosonde observations** provide independent atmospheric-state data for comparison with modeled environmental conditions.

IGRA data are not presented as validation of optical propagation.

---

## 🔭 Beam Propagation and Pointing

The reduced-order optical model evaluates:

- diffraction
- additional optical spreading
- reduced-order turbulence effects
- pointing uncertainty
- atmospheric transmission
- beam spot size
- target irradiance
- delivered optical power

The model is intended for engineering analysis, sensitivity studies, and system interaction analysis rather than high-fidelity wave-optics prediction.

---

## ⚡ Power and Thermal Digital Twin

The engagement model is coupled to finite system resources.

Modeled quantities include:

- electrical input
- optical output
- conversion losses
- stored energy
- generator contribution
- cooling capacity
- waste heat
- coolant temperature
- thermal margin

The optional **Thermal & Power Digital Twin** extends this capability with a generic multi-node thermal network and transient **1-D Fourier conduction analysis**.

This allows the application to investigate a fundamental aerospace systems question:

> **Can the system perform the required mission while remaining inside its energy and thermal constraints?**

---

## 🌡️ Target Thermal Response

The primary target-response model uses a reduced-order areal lumped-capacitance formulation.

An optional 1-D conduction model provides higher-resolution investigation of transient thermal behavior through a representative material slab.

Outputs can include:

- front-surface temperature
- internal temperature
- back-surface temperature
- thermal margins
- time-to-limit behavior
- Estimated Thermal Effect Index

The Estimated Thermal Effect Index is an **engineering indicator**, not a probability of kill or validated lethality prediction.

---

## 🎲 Monte Carlo Uncertainty Quantification

Monte Carlo simulation propagates uncertain inputs through the modeled system.

Capabilities include:

- configurable trial counts
- uncertain input sampling
- output distributions
- percentile analysis
- sensitivity analysis
- threshold exceedance statistics
- uncertainty-informed engineering assessment

This allows system behavior to be evaluated as a **distribution of possible outcomes rather than a single deterministic result**.

---

## 🛰️ 3-D Digital Twin

The interactive 3-D environment visualizes:

- target trajectories
- moving-platform state
- measurement reconstruction
- engagement geometry
- beam pointing
- line-of-sight relationships
- closest-point-of-approach geometry

The visualization connects numerical model state with an intuitive spatial representation of the simulated system.

---

# Systems Engineering and Digital Engineering

The application includes an **Engineering Lab** that connects model execution with systems-engineering artifacts.

Capabilities include:

- stakeholder-need registry
- system requirements
- requirement evaluation
- requirements traceability
- verification methods
- verification evidence
- trade studies
- model provenance
- regression checks
- engineering-data export

The current application includes a defined engineering baseline and checks whether traceability identifiers resolve to registered stakeholder needs and system requirements.

This creates a lightweight bridge between:

**Requirements → Architecture → Models → Simulation → Evidence → Engineering Decision**

The objective is not simply to produce simulation outputs, but to maintain a connection between **why a capability exists, what requirement it supports, how it is evaluated, and what evidence the model produces**.

---

# Verification, Validation, and Reference Benchmarking

The project deliberately distinguishes **verification** from **validation**.

## Verification

Verification activities include:

- numerical convergence checks
- regression testing
- covariance consistency checks
- conservation/invariant checks
- model sanity checks
- requirements traceability checks

The objective is to determine whether the implemented model behaves consistently with its mathematical and software definition.

## NASA GTM Reference Benchmarking

The generic aerospace model can be compared with **NASA Generic Transport Model (GTM)** reference data.

This capability is treated as:

**model-to-model flight-dynamics reference benchmarking**

and not as empirical flight-test validation of the application.

## NOAA IGRA Atmospheric Comparison

**NOAA IGRA** radiosonde observations provide independent atmospheric-state data that can be compared with modeled environmental conditions.

The comparison applies only to the atmospheric variables evaluated.

It does not constitute empirical validation of the directed-energy propagation or target-effect models.

---

# Application Architecture

The Streamlit application currently contains eleven primary engineering workspaces:

| # | Module | Engineering Purpose |
|---:|---|---|
| 1 | **Engagement Loop** | Closed-loop engagement analysis |
| 2 | **State Estimation** | Tracking and estimator diagnostics |
| 3 | **Time History** | Dynamic simulation outputs |
| 4 | **Monte Carlo** | Uncertainty quantification |
| 5 | **Model State** | Internal model-state inspection |
| 6 | **3D Digital Twin** | Spatial visualization |
| 7 | **Export** | Engineering-data export |
| 8 | **Advanced Twin / V&V** | Higher-order aerospace modeling and verification |
| 9 | **Engineering Lab** | Requirements, trades, provenance, and regression |
| 10 | **Validation & Reference Benchmarking** | NASA GTM and NOAA IGRA comparisons |
| 11 | **Thermal & Power Digital Twin** | Component thermal and power analysis |

---

# Technology Stack

| Technology | Engineering Role |
|---|---|
| 🐍 **Python** | Core simulation, numerical models, and application logic |
| 🔺 **Streamlit** | Interactive digital-twin engineering interface |
| 🔢 **NumPy** | Numerical computation and matrix operations |
| 🐼 **Pandas** | Engineering-data structures and analysis |
| 📊 **Matplotlib** | Scientific plotting and V&V visualization |
| 📈 **Plotly** | Interactive 2-D/3-D visualization |
| 🚀 **NASA GTM** | Flight-dynamics reference benchmarking |
| 🌎 **NOAA IGRA** | Atmospheric-state observational comparison |
| 🎲 **Monte Carlo** | Uncertainty quantification |
| `{ }` **JSON / CSV** | Portable engineering-data export |

---

# Relevance to Aerospace and Autonomous Systems

Although this demonstrator uses a directed-energy engagement scenario, its underlying computational architecture is much broader.

The same engineering patterns apply to:

### Uncrewed Aircraft Systems
- autonomous mission management
- sensor fusion
- detect-and-avoid
- navigation
- trajectory analysis
- power-aware mission planning

### Autonomous Systems
- perception-to-decision pipelines
- state estimation
- environment modeling
- closed-loop decision architectures
- uncertainty-aware autonomy

### Space Systems
- spacecraft proximity operations
- rendezvous modeling
- planetary surface autonomy
- landing-zone assessment
- resource-constrained mission planning

### Robotics
- localization
- sensor fusion
- robotic inspection
- environment-aware decision making
- resource management

### Digital Engineering
- multidisciplinary simulation
- requirements traceability
- model provenance
- verification
- trade studies
- uncertainty quantification
- digital-thread development

These applications share a common engineering problem:

> **Estimate the state of a dynamic environment, propagate physical models, quantify uncertainty, enforce system constraints, and convert the result into an actionable engineering decision.**

---

# AI and Physics-Informed Autonomy

The current application is intentionally **physics-informed rather than AI-dependent**.

Physics-based models provide interpretable system behavior, dimensional consistency, explicit assumptions, and engineering constraints.

AI/ML can then be introduced where it provides measurable value, including:

- perception and classification
- anomaly detection
- surrogate modeling
- adaptive parameter estimation
- predictive maintenance
- mission planning
- intelligent decision support
- engineering-model interrogation
- natural-language interaction with digital twins

A particularly promising architecture for future development is:

> **Physics-Based Digital Twin + State Estimation + Uncertainty Quantification + AI Decision Support**

This approach allows machine intelligence to operate against a structured engineering model instead of replacing the underlying physics with an opaque prediction pipeline.

---

# Potential Industry Applications

## Aerospace

Digital twins, flight simulation, mission engineering, GNC, modeling and simulation, multidisciplinary engineering analysis, and V&V.

## Uncrewed and Autonomous Systems

UAS autonomy, sensor fusion, navigation, tracking, mission management, autonomous decision architectures, and resilient navigation.

## Defense Technology

Mission simulation, tracking, resource allocation, decision support, uncertainty analysis, and digital engineering.

## AI-Enabled Engineering

Physics-informed AI, surrogate modeling, engineering copilots, intelligent model interrogation, anomaly detection, and AI-assisted systems engineering.

## Robotics

State estimation, sensor fusion, closed-loop autonomy, environment modeling, resource management, and decision support.

---

# Engineering Scope and Limitations

This repository is a **physics-informed, reduced-order digital engineering and research prototype**.

It is **not**:

- a validated operational weapon model
- a high-fidelity missile simulation
- a lethality model
- a probability-of-kill model
- an operational fire-control system
- a substitute for hardware testing

Aerodynamics, atmospheric propagation, beam behavior, thermal response, power behavior, and subsystem interactions use deliberately reduced-order representations appropriate for engineering exploration.

Results should therefore be interpreted as **scenario-dependent engineering outputs**, not predictions of real-world operational performance.

---

# Engineering Competencies Demonstrated

This project demonstrates hands-on work across:

**Python Engineering Software Development**

**Aerospace Modeling & Simulation**

**Digital Twin Architecture**

**3-DOF / 6-DOF Flight Dynamics**

**Rigid-Body Dynamics**

**Coordinate-Frame Transformations**

**State Estimation & Kalman Filtering**

**Sensor Fusion & Tracking**

**Monte Carlo Simulation**

**Uncertainty Quantification**

**Power & Thermal Modeling**

**Transient Heat Transfer**

**Scientific Computing**

**Interactive 3-D Visualization**

**Requirements Traceability**

**Engineering Trade Studies**

**Model Provenance**

**Verification & Validation**

**Reference Benchmarking**

**Systems Engineering**

**Autonomous-System Architecture**

---

# Project Direction

This project is part of a broader effort to explore how **aerospace physics, autonomous systems, digital engineering, systems engineering, and AI-enabled analysis** can be integrated into lightweight and accessible engineering software.

The longer-term objective is to shorten the engineering path from:

> **Requirements → Models → Simulation → Evidence → Engineering Decision**

while maintaining explicit assumptions, traceability, uncertainty awareness, model provenance, and human engineering judgment.

---

<p align="center">
  <strong>MODEL. SIMULATE. VALIDATE. DOMINATE.</strong>
</p>

<p align="center">
  <em>Physics-informed digital engineering for complex aerospace and autonomous systems.</em>
</p>
