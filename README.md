# NeuroForecast AI: Longitudinal Vestibular Schwannoma Growth Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Medical Imaging](https://img.shields.io/badge/Domain-Neuro--Oncology%20MRI-0284C7?style=flat)]()

A clinical decision support platform for neuro-oncology teams managing patients with Vestibular Schwannoma (acoustic neuroma) under conservative surveillance. The system combines multi-session 3D magnetic resonance imaging (MRI), spatiotemporal deep learning, and volumetric trajectory modeling to estimate future tumor progression and support clinical intervention planning.

---

## Overview

Vestibular Schwannoma is a benign, slow-growing neoplasm originating from the myelin sheath of the eighth cranial nerve within the internal auditory canal and cerebellopontine angle. Standard clinical practice often adopts a conservative "wait-and-scan" protocol with periodic follow-up MRI sessions before committing to microsurgical resection or stereotactic radiosurgery.

Current routine assessments rely heavily on manual one-dimensional or two-dimensional caliper measurements on axial MRI slices. However, acoustic neuromas exhibit asymmetric, non-linear three-dimensional volumetric expansion that single-axis measurements can mischaracterize.

NeuroForecast AI provides an automated, reproducible workflow:
1. **Longitudinal Volumetric Segmentation:** Measures true three-dimensional tumor volume ($mm^3$) across sequential scan sessions.
2. **Spatiotemporal Forecasting (`TempVSNet`):** Ingests baseline and follow-up scan pairs to forecast subsequent tumor boundaries using a 5-fold ensemble 3D architecture.
3. **Tumor Kinetic Analysis:** Computes interval volumetric growth rate and tumor doubling time (TDT) to differentiate indolent lesions from rapidly progressive cases.
4. **Clinical Reporting:** Generates structured multi-page PDF summaries containing patient demographics, cross-sectional radiographic slices, trajectory curves, and Multi-Disciplinary Team (MDT) triage indicators.

---

## System Architecture

```
  Session 1 (Baseline MRI)       Session 2 (Current MRI)
             │                              │
             └──────────────┬───────────────┘
                            ▼
              ┌───────────────────────────┐
              │    Longitudinal Pair      │
              │    Alignment & Voxel Prep │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │  TempVSNet 3D Ensemble    │
              │  (5-Fold Cross-Validated) │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │  Predicted Mask & Volume  │
              │  at Future Follow-Up (t3) │
              └─────────────┬─────────────┘
                            ▼
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│  Clinical Web Dashboard     │   │  Automated PDF Report       │
│  - Slice-by-Slice Viewer    │   │  - Radiographic Snapshots   │
│  - Growth Trajectory Curves │   │  - Growth Kinetics (TDT)    │
│  - High-Risk MDT Flagging   │   │  - Digital Verification QR  │
└─────────────────────────────┘   └─────────────────────────────┘
```

---

## Technical Stack

* **Backend & API:** Python 3.10+, Flask
* **Deep Learning Engine:** PyTorch (3D Temporal Convolutional Networks with 5-fold ensemble inference)
* **Image Processing & Math:** NumPy, SciPy (multidimensional image processing and cubic spline interpolation), Pandas
* **Data Visualization:** Matplotlib (vector plots and kinetic trajectory charts)
* **Documentation & Audit:** fpdf2, qrcode
* **Web Interface:** HTML5, CSS3, JavaScript, Jinja2 templates

---

## Directory Structure

```
Brain-Tumor-FYP/
├── app.py                  # Primary Flask server and API endpoints
├── requirements.txt        # Python dependency manifest
├── .gitattributes          # Linguist language detection overrides
├── models/                 # Pretrained PyTorch ensemble weights (fold0 to fold4)
│   ├── fold0_best.pth
│   ├── fold1_best.pth
│   ├── fold2_best.pth
│   ├── fold3_best.pth
│   └── fold4_best.pth
├── utils/
│   ├── model_arch.py       # TempVSNet 3D neural network definition
│   └── pdf_generator.py    # Multi-page clinical report synthesizer
├── templates/              # Clinical dashboard HTML views
│   ├── login.html          # Authentication portal
│   ├── database.html       # Patient directory with progress flags
│   ├── patient_profile.html# Longitudinal session history
│   ├── mri_viewer.html     # Interactive cross-sectional slice viewer
│   ├── analysis.html       # Volumetric analysis & model execution
│   └── add_patient.html    # Patient registration portal
├── static/                 # Styles, client assets, and generated trajectory plots
└── local_data/             # De-identified longitudinal MRI arrays (.npz) and demographics
    ├── patients_db.json
    ├── VS-MC-RC2_demographic_Dec2025.csv
    └── mris/               # Multi-session voxel volumes
```

---

## Local Setup & Quick Start

### 1. Prerequisites
* Python 3.10 or 3.11 installed
* CUDA-compatible GPU recommended for accelerated 3D volumetric inference (CPU fallback supported)

### 2. Environment Configuration
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Rohith-S-15/Brain-Tumor-FYP.git
cd Brain-Tumor-FYP
python -m venv venv
```

Activate the environment:
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Clinical Portal
```bash
python app.py
```

The application runs locally at `http://127.0.0.1:5000`.

### 5. Authentication & Demo Access
* **Default Password:** `admin`
* **Role Selection:** Select any clinical role (Radiologist, Neurosurgeon, or Administrator) from the portal dropdown.

---

## Demo Patient Cohort

The repository comes packaged with anonymized longitudinal scan pairs under `local_data/mris/` for immediate testing:

| Patient ID | Baseline Session | Follow-Up Session | Surveillance Status |
| :--- | :--- | :--- | :--- |
| **VS_MC_RC2_028** | 1992-09-19 | 1994-05-28 | Conservative Observation |
| **VS_MC_RC2_040** | 1993-02-14 | 1995-02-17 | Slow Progressor |
| **VS_MC_RC2_063** | 1992-06-12 | 1993-06-08 | Stable / Indolent |
| **VS_MC_RC2_066** | 1992-04-16 | 1992-11-27 | Active Growth (MDT Flagged) |
| **VS_MC_RC2_082** | 1992-04-26 | 1993-05-14 | Conservative Observation |
| **VS_MC_RC2_192** | 1993-01-09 | 1993-05-27 | Longitudinal Multi-Epoch |

---

## Evaluation Metrics & Clinical Utility

The ensemble model evaluates volumetric consistency across multiple quantitative axes:
* **Volumetric Discrepancy:** Measured in cubic millimeters ($mm^3$) against ground-truth contour segmentations.
* **Tumor Doubling Time (TDT):** Derived using exponential and logarithmic kinetic modeling to estimate tumor doubling velocity over annual intervals.
* **Triage Categorization:** Patients exhibiting positive interval volumetric acceleration over threshold cutoffs are flagged for prioritized Multi-Disciplinary Team review.

---

## Ethical Statement & Clinical Disclaimer

This software and its predictive models were developed as an academic Final Year Project for investigational and research purposes only. The volumetric predictions, automated contour estimates, and generated clinical documentation do not constitute a definitive medical diagnosis. All algorithmic outputs must be independently reviewed and verified by a board-certified neuroradiologist or neurosurgeon before formulating treatment decisions.
