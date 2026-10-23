# AgriPredict AI — Smart Agriculture Analytics & Crop Prediction Platform

An enterprise-grade, interactive agricultural analytics and predictive intelligence platform engineered for third-year Artificial Intelligence and Data Science coursework. The platform combines multiple real-world Indian agricultural datasets spanning historical crop production, multi-city meteorological climatology, laboratory soil nutrient profiles, chemical fertilizer classifications, and agricultural mandi market prices.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [Key Features & Dashboard Modules](#-key-features--dashboard-modules)
4. [Datasets & Data Governance](#-datasets--data-governance)
5. [Architecture & Folder Structure](#-architecture--folder-structure)
6. [Technology Stack](#-technology-stack)
7. [Installation & Setup](#-installation--setup)
8. [Running the Application](#-running-the-application)
9. [Machine Learning Pipelines & Evaluation](#-machine-learning-pipelines--evaluation)
10. [Analytical Limitations & Disclaimers](#-analytical-limitations--disclaimers)
11. [Future Enhancements](#-future-enhancements)

---

## 🌾 Project Overview

Modern agricultural decision-making demands multi-dimensional intelligence: matching local soil chemistry with optimal crop varieties, selecting balanced chemical fertilizer formulations, understanding meteorological exposure and monsoon timing, and aligning crop production volumes with Mandi market prices and Government Minimum Support Prices (MSP).

**AgriPredict AI** delivers a unified web-based analytics studio built with **Python 3**, **Streamlit**, **Scikit-learn**, and **Plotly**. It ingests five distinct real-world agricultural datasets without synthetic fabrication, cleans and harmonizes them through robust pipelines, and provides both predictive ML models and interactive visual dashboards.

---

## 🎯 Objectives

1. **Multi-Dataset Synthesis:** Ingest and integrate 5 distinct agricultural datasets with strict schema validation.
2. **Data Cleaning & Quality Engineering:** Eliminate duplicates, handle missing values via crop-specific median imputation, and detect statistical outliers (IQR) without blindly purging valid agricultural extremes.
3. **Exploratory Data Analysis (EDA):** Provide an interactive chart builder with dynamic Plotly visualizations (Heatmaps, Bar, Scatter, Box, Donut, Line).
4. **Machine Learning Benchmarking:** Train, evaluate, and persist Random Forest, Decision Tree, Gradient Boosting, and Linear Regression models using reproducible train/test splits.
5. **Prescriptive Recommendations:** Enable real-time prediction for crop suitability and fertilizer formulations based on user-supplied inputs with model confidence scores.
6. **Market Intelligence & Policy Audit:** Evaluate mandi modal prices against official Government Minimum Support Prices (MSP).
7. **Automated Executive Reporting:** Synthesize downloadable publication-grade PDF summaries via ReportLab alongside CSV data packages.

---

## 🚀 Key Features & Dashboard Modules

| Module / Page | Primary Functionality |
| :--- | :--- |
| **Main Landing (`app.py`)** | Platform overview, verified KPI indicators, dataset health status cards, and architectural matrix. |
| **1. Overview (`1_Overview.py`)** | Macro-level executive insights, top crops by production, seasonal distribution, and key empirical findings. |
| **2. Dataset Manager (`2_Dataset_Manager.py`)** | Dataset inventory, column schemas, null profiling, temporary CSV uploader, and **Multi-Dataset Integration Studio**. |
| **3. Data Preprocessing (`3_Data_Preprocessing.py`)** | Interactive preprocessing pipeline, missing value imputation, IQR outlier profiling, and persistence to `data/processed/`. |
| **4. Exploratory Analysis (`4_Exploratory_Analysis.py`)** | Custom chart studio for correlation heatmaps, scatter plots with OLS trendlines, box plots, and histograms. |
| **5. Crop Production (`5_Crop_Production.py`)** | Filterable spatio-temporal production analysis across states, crops, seasons, and years (1997–2015), plus state-crop heatmap. |
| **6. Weather Analytics (`6_Weather_Analytics.py`)** | Meteorological climatology across 8 Indian cities (2009–2020), heatwave and heavy rainfall day counts, and seasonal patterns. |
| **7. Soil & Crop Rec. (`7_Soil_and_Crop_Recommendation.py`)** | Soil nutrient profiling (N, P, K, pH, EC, micronutrients) and real-time ML crop recommendation with radar charts. |
| **8. Fertilizer Analytics (`8_Fertilizer_Analytics.py`)** | Chemical fertilizer formulation analytics, cross-tabulations, and interactive fertilizer grade predictor. |
| **9. Market Intelligence (`9_Market_Intelligence.py`)** | APMC Mandi modal prices across 122 commodities, price spread analysis, and Government MSP policy audit. |
| **10. ML Model Lab (`10_ML_Model_Lab.py`)** | Unified laboratory for benchmarking classification, regression, and unsupervised K-Means clustering with 2D PCA. |
| **11. Reports (`11_Reports.py`)** | One-click ReportLab PDF executive report synthesis and downloadable CSV analytical data packages. |

---

## 📂 Datasets & Data Governance

All datasets are sourced from real-world agricultural records:

1. **Crop Production (`crop_production.csv`):** 246,091 historical records across Indian states, districts, crops, and seasons (1997–2015).
2. **Fertilizer Recommendation (`Fertilizer Prediction.csv`):** 99 records documenting soil type, crop type, moisture, N-P-K levels, and recommended fertilizer grades.
3. **Agricultural Market Prices (`Indian Agricultural Market Prices.csv`):** Cross-sectional snapshot across 122 commodities detailing modal, min, max prices (Rs/Quintal) and Government MSP.
4. **Soil Nutrients & Crop (`Soil Nutrients and Crop Recommendation Dataset.csv`):** 620 laboratory soil samples mapping 11 chemical properties to 6 crop classes (pomegranate, mango, grapes, mulberry, ragi, potato).
5. **Historical Weather Observatories (`WEATHER DATASET/`):** 771,456 hourly records (96,432 per city) from 2009 to 2020 across 8 major hubs: Bengaluru, Bombay, Delhi, Hyderabad, Jaipur, Kanpur, Nagpur, and Pune.

*Raw datasets are strictly preserved in `data/raw/` and the workspace root. Cleaned artifacts are exported to `data/processed/`.*

---

## 🏗️ Architecture & Folder Structure

```
dvp2/ (Workspace Root)
├── app.py                             # Main Streamlit landing dashboard
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
├── train_all.py                       # Automated model pre-training script
│
├── config/
│   ├── __init__.py
│   └── settings.py                    # Dynamic paths, UI design tokens, dataset specs
│
├── data/
│   ├── raw/                           # Original immutable CSV files & Weather directory
│   └── processed/                     # Cleaned, standardized datasets (.gitkeep)
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py                 # Cached loaders with encoding fallbacks (UTF-8, Latin-1)
│   ├── data_cleaning.py               # Missing imputation, duplicate purging, IQR outliers
│   ├── weather_processing.py          # Temporal feature extraction & daily/monthly aggregation
│   └── dataset_integration.py         # Multi-dataset joins, canonical crop mapping, diagnostics
│
├── analysis/
│   ├── __init__.py
│   ├── crop_analysis.py               # Crop rankings, temporal trends, seasonal splits
│   ├── weather_analysis.py            # City climatology, rainfall distributions, extreme events
│   ├── soil_analysis.py               # Macronutrient/micronutrient stats and pH categorization
│   ├── fertilizer_analysis.py         # Formulation breakdown and soil/crop cross-tabulations
│   └── market_analysis.py             # Mandi modal prices, price spreads, MSP policy audit
│
├── models/
│   ├── __init__.py
│   ├── crop_recommendation.py         # Multi-class crop classification & inference
│   ├── fertilizer_model.py            # Fertilizer ColumnTransformer pipeline & inference
│   ├── yield_prediction.py            # Crop yield continuous regression (Tonnes/Ha)
│   ├── clustering.py                  # K-Means clustering, Elbow curve, 2D PCA projection
│   └── model_evaluation.py            # Cross-model evaluation tables and metric comparators
│
├── utils/
│   ├── __init__.py
│   ├── charts.py                      # Reusable, themed Plotly visual components
│   ├── metrics.py                     # Platform KPI and ML scorecard computation
│   ├── report_generator.py            # ReportLab PDF compiler & CSV export utilities
│   └── helpers.py                     # Modern CSS components, UI badges, domain disclaimers
│
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Dataset_Manager.py
│   ├── 3_Data_Preprocessing.py
│   ├── 4_Exploratory_Analysis.py
│   ├── 5_Crop_Production.py
│   ├── 6_Weather_Analytics.py
│   ├── 7_Soil_and_Crop_Recommendation.py
│   ├── 8_Fertilizer_Analytics.py
│   ├── 9_Market_Intelligence.py
│   ├── 10_ML_Model_Lab.py
│   └── 11_Reports.py
│
├── saved_models/                      # Trained model joblib pipelines (.gitkeep)
├── reports/                           # Generated ReportLab PDF executive summaries (.gitkeep)
│
└── tests/
    ├── __init__.py
    ├── test_data_loading.py           # Pytest suite for loaders and inspection
    └── test_preprocessing.py          # Pytest suite for cleaning, joins, and ML inference
```

---

## 💻 Technology Stack

- **Core Language:** Python 3.10+ (Verified on Python 3.14)
- **Web Application Framework:** Streamlit (v1.35+)
- **Machine Learning & Preprocessing:** Scikit-learn, Joblib
- **Data Manipulation & Ingestion:** Pandas, NumPy
- **Interactive Visualization:** Plotly Express, Plotly Graph Objects
- **Document Synthesis:** ReportLab (PDF Engine)
- **Automated Testing:** Pytest

---

## ⚙️ Installation & Setup

Follow these steps to set up the project locally on Windows through Antigravity IDE, Visual Studio Code, or PowerShell:

### 1. Clone or Open Workspace
Open the workspace directory in your terminal:
```powershell
cd "c:\Users\Asus\OneDrive\Desktop\SEM\Sem05\Miniproject\dvp2"
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## ▶️ Running the Application

### 1. Run Automated Unit Tests
Verify that all data loading, preprocessing, and model pipelines pass:
```powershell
python -m pytest tests/
```
*(All 12 unit tests should execute and pass.)*

### 2. Pre-Train Model Artifacts (Optional)
To verify or retrain all saved model pipelines:
```powershell
python train_all.py
```

### 3. Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🤖 Machine Learning Pipelines & Evaluation

### 1. Soil Nutrient Crop Recommendation
- **Task:** Multi-Class Classification (6 crops: pomegranate, mango, grapes, mulberry, ragi, potato).
- **Features:** Nitrogen, Phosphorus, Potassium, pH, Electrical Conductivity, Sulfur, Copper, Iron, Manganese, Zinc, Boron.
- **Top Algorithm:** **Random Forest Classifier** (Accuracy: **95.97%**, Weighted F1: **0.9598**).

### 2. Chemical Fertilizer Recommendation
- **Task:** Multi-Class Classification (7 formulations: Urea, DAP, 28-28, 14-35-14, 20-20, 17-17-17, 10-26-26).
- **Features:** Categorical (`Soil Type`, `Crop Type`) via `OneHotEncoder` + Numerical (`Temperature`, `Humidity`, `Moisture`, `N`, `P`, `K`) via `StandardScaler`.
- **Top Algorithm:** **Random Forest Classifier** (Accuracy: **100%**, Weighted F1: **1.0000**).

### 3. Crop Yield Prediction
- **Task:** Continuous Regression (`Yield = Production / Area` in Tonnes/Hectare).
- **Features:** State, Crop, Season, Area.
- **Top Algorithm:** **Random Forest Regressor** ($R^2$: **0.8756**, RMSE: **3.406 Tonnes/Ha**).

### 4. Unsupervised K-Means Soil & Climatology Clustering
- **Task:** Exploratory cluster segmentation.
- **Diagnostics:** Silhouette Score, Davies-Bouldin Index, Elbow Inertia Curve, 2D PCA projection.

---

## ⚠️ Analytical Limitations & Disclaimers

1. **Empirical Market Snapshot:** The market dataset contains 122 commodity rows without longitudinal daily date timestamps. Longitudinal time-series price forecasting was deliberately excluded to avoid spurious projections.
2. **Weather-District Integration:** City weather files originate from 8 major urban/metropolitan stations. Multi-dataset integration with district crop production is restricted to verified geographical overlaps and overlapping years (2009–2015).
3. **Agronomic Decision Support:** AI model recommendations provide decision support and do not substitute for official on-field soil testing and agronomist consultation.

---

## 🔮 Future Enhancements

- **Satellite Remote Sensing:** Integrate Sentinel-2 / Landsat NDVI vegetation index imagery.
- **Live Mandi API:** Connect to the Government of India `data.gov.in` Agmarknet real-time price feeds.
- **Hyper-Local IoT Sensors:** Stream live soil moisture and temperature telemetry from Arduino/ESP32 microcontrollers.
