# optimized_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# -----------------------------
# Config (update paths if needed)
# -----------------------------
MODEL_PATH = "sih_model.pkl"
SCALER_PATH = "sih_scaler.pkl"
ENCODER_PATH = "sih_label_encoder.pkl"
DATASET_PATH = "SIH_water_treatment_generated_2000_each_fixedA.csv"  # local path used earlier

st.set_page_config(page_title="SIH — Fast Water Treatment Advisor", layout="wide")

# -----------------------------
# Caching: load artifacts once
# -----------------------------
@st.cache_resource
def load_model_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH, encoder_path=ENCODER_PATH):
    """Load model, scaler and label encoder once per session (cached)."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        return model, scaler, le
    except Exception as e:
        st.error("Error loading model artifacts: " + str(e))
        st.stop()

@st.cache_data
def load_dataset(path=DATASET_PATH):
    """Cache dataset (used only for download / optional EDA)."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

# -----------------------------
# Lazy heavy imports (only when needed)
# -----------------------------
def plot_bar(df):
    import plotly.express as px
    fig = px.bar(df, x='param', y='value', title='Input Parameter Values', height=300)
    return fig

def create_pdf_bytes(report):
    # import inside function to avoid startup cost
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "SIH Water Treatment Report", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, "Inputs:", ln=True)
    for k, v in report['inputs'].items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 6, f"Predicted stage: {report['predicted_stage']}", ln=True)
    pdf.cell(0, 6, f"Final stage (use-case {report['use_case']}): {report['final_stage']}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0, 6, "Sequence:")
    for s in report['sequence']:
        pdf.multi_cell(0, 6, "- " + s)
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# Utility functions (fast)
# -----------------------------
def compute_score(vals):
    # light-weight score similar to earlier, but simpler for speed
    ideal = {
        'pH': 7.0, 'TSS': 50, 'Turbidity': 10, 'BOD': 10, 'COD': 30,
        'NH4_N': 2, 'Total_Nitrogen': 6, 'Phosphate': 1,
        'Fecal_Coliform': 500, 'Oil_Grease': 5, 'TDS': 1200, 'Heavy_Metals': 0.1
    }
    max_vals = {
        'pH': 3.0, 'TSS': 2000, 'Turbidity': 500, 'BOD': 800, 'COD': 1500,
        'NH4_N': 80, 'Total_Nitrogen': 200, 'Phosphate': 50,
        'Fecal_Coliform': 1e6, 'Oil_Grease': 300, 'TDS': 6000, 'Heavy_Metals': 50
    }
    score_components = []
    for k, v in vals.items():
        if k not in ideal: continue
        dist = min(abs(v - ideal[k]) / (max_vals[k] - ideal[k] + 1e-9), 1.0)
        score_components.append(1.0 - dist)
    return int(np.clip(np.mean(score_components) * 100, 0, 100))

def classify_badness(param, val):
    thresholds = {
        'TSS': (50, 200),
        'Turbidity': (10, 100),
        'BOD': (10, 100),
        'COD': (30, 200),
        'NH4_N': (2, 10),
        'Total_Nitrogen': (5, 30),
        'Phosphate': (1, 5),
        'Fecal_Coliform': (500, 10000),
        'Oil_Grease': (5, 50),
        'TDS': (1000, 3000),
        'Heavy_Metals': (0.1, 1.0),
        'pH': (6.5, 8.5)
    }
    if param not in thresholds:
        return "Unknown"
    low, mid = thresholds[param]
    if param == 'pH':
        return "Good" if (low <= val <= mid) else "Poor"
    if val <= low:
        return "Good"
    elif val <= mid:
        return "Moderate"
    else:
        return "Poor"

def build_sequence_up_to(stage):
    order = ["Primary", "Secondary", "Tertiary", "Advanced"]
    seq_map = {
        "Primary": ["Screening", "Grit removal", "Primary sedimentation"],
        "Secondary": ["Aeration / activated sludge", "Secondary clarification"],
        "Tertiary": ["Nutrient removal (N&P)", "Filtration", "Disinfection"],
        "Advanced": ["UF/RO", "Activated carbon", "Heavy metal removal", "AOP"]
    }
    steps = []
    for s in order[: order.index(stage) + 1]:
        steps.extend(seq_map[s])
    return steps

# -----------------------------
# Load artifacts (cached)
# -----------------------------
model, scaler, le = load_model_artifacts()

# -----------------------------
# Sidebar utilities (fast, cached dataset)
# -----------------------------
with st.sidebar:
    st.header("Utilities")
    st.write("Dataset (for download):")
    ds = load_dataset()
    if ds is not None:
        st.write(f"Rows: {len(ds):,}")
        csv_bytes = ds.to_csv(index=False).encode("utf-8")
        st.download_button("Download dataset CSV", data=csv_bytes, file_name="sih_dataset.csv", mime="text/csv")
    else:
        st.info("Dataset not found. You can still use prediction if artifacts exist.")
    st.markdown("---")
    st.caption("Model artifacts loaded and cached for speed.")

# -----------------------------
# Main UI
# -----------------------------
st.title("💧 SIH — Fast Water Treatment Advisor (Optimized)")
st.write("Enter the 12 water-quality parameters and pick a use-case. Model & heavy libs are cached for fast startup.")

# Input fields in 3 columns
cols = st.columns(3)
inputs = {}
field_specs = [
    ("pH", 7.0, 0.0, 14.0),
    ("TSS", 100.0, 0.0, 5000.0),
    ("Turbidity", 10.0, 0.0, 1000.0),
    ("BOD", 50.0, 0.0, 2000.0),
    ("COD", 120.0, 0.0, 3000.0),
    ("NH4_N", 5.0, 0.0, 200.0),
    ("Total_Nitrogen", 20.0, 0.0, 500.0),
    ("Phosphate", 2.0, 0.0, 100.0),
    ("Fecal_Coliform", 500.0, 0.0, 1_000_000.0),
    ("Oil_Grease", 5.0, 0.0, 200.0),
    ("TDS", 1200.0, 0.0, 10000.0),
    ("Heavy_Metals", 0.5, 0.0, 50.0)
]
for i, (name, default, lo, hi) in enumerate(field_specs):
    c = cols[i % 3]
    inputs[name] = c.number_input(name, min_value=lo, max_value=hi, value=float(default))

use_case = st.selectbox("Use case", ["Irrigation", "Domestic Use", "Industrial Use", "Drinking Water", "Aquaculture"])

# Only perform heavy ops on button click
if st.button("Predict & Recommend"):
    with st.spinner("Predicting and preparing recommendations..."):
        # prepare vector and predict
        feature_order = ['pH','TSS','Turbidity','BOD','COD','NH4_N','Total_Nitrogen','Phosphate','Fecal_Coliform','Oil_Grease','TDS','Heavy_Metals']
        X = np.array([[inputs[f] for f in feature_order]])
        Xs = scaler.transform(X)
        pred_enc = model.predict(Xs)[0]
        pred_stage = le.inverse_transform([pred_enc])[0]

        # resolve final stage by use case
        ORDER = ["Primary","Secondary","Tertiary","Advanced"]
        USECASE_MIN = {"Irrigation":"Secondary","Domestic Use":"Tertiary","Industrial Use":"Tertiary","Drinking Water":"Advanced","Aquaculture":"Tertiary"}
        final_stage = ORDER[max(ORDER.index(pred_stage), ORDER.index(USECASE_MIN[use_case]))]

        # score & risk (fast)
        score = compute_score(inputs)
        # simple risk reasons
        risk_reasons = []
        risk_level = "Low"
        if inputs['Fecal_Coliform'] > 50000:
            risk_reasons.append("High fecal coliform")
            risk_level = "Medium"
        if inputs['Heavy_Metals'] > 1.0:
            risk_reasons.append("High heavy metals")
            risk_level = "High"

        # build sequence
        seq = build_sequence_up_to(final_stage)
        # chemical suggestions (light)
        chlorine = 4.0 if inputs['Fecal_Coliform'] > 100000 else (2.5 if inputs['Fecal_Coliform'] > 10000 else 1.0)
        alum = 40 if inputs['TSS'] > 500 else (20 if inputs['TSS'] > 200 else 8)

    # Display summary quickly (two columns)
    left, right = st.columns([2,1])
    with left:
        st.subheader("Prediction")
        st.write(f"Model predicted: **{pred_stage}**")
        st.write(f"Final for use-case **{use_case}**: **{final_stage}**")

        st.markdown("**Recommended treatment sequence:**")
        for s in seq:
            st.write("• " + s)

        st.markdown("**Chemical suggestions (approx)**")
        st.write(f"- Chlorine dose (mg/L): {chlorine}")
        st.write(f"- Alum (mg/L): {alum}")
        st.write(f"- Recommend RO: {'Yes' if inputs['TDS'] > 3000 or final_stage=='Advanced' else 'No'}")

    with right:
        # score badge + risk
        if score >= 75:
            st.success(f"Score: {score}/100 (Good)")
        elif score >= 50:
            st.warning(f"Score: {score}/100 (Moderate)")
        else:
            st.error(f"Score: {score}/100 (Poor)")

        st.markdown(f"**Risk**: {risk_level}")
        if risk_reasons:
            for r in risk_reasons:
                st.write("- " + r)

    # Parameter analysis table (fast)
    analysis = [(k, inputs[k], classify_badness(k, inputs[k])) for k in feature_order]
    df_analysis = pd.DataFrame(analysis, columns=["Parameter", "Value", "Status"])
    st.subheader("Parameter Analysis")
    st.dataframe(df_analysis, width=900, height=220)

    # Plotly bar (lazy import)
    try:
        plot_df = pd.DataFrame({'param':[r[0] for r in analysis], 'value':[r[1] for r in analysis]})
        fig = plot_bar(plot_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write("Charting library not available.")

    # Prepare PDF report (lazy)
    report = {
        "inputs": inputs,
        "predicted_stage": pred_stage,
        "use_case": use_case,
        "final_stage": final_stage,
        "sequence": seq,
        "score": score,
        "risk": risk_level
    }
    try:
        pdf_bytes = create_pdf_bytes(report)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="sih_report.pdf", mime="application/pdf")
    except Exception:
        st.info("PDF library not available; install fpdf for PDF export.")

    st.success("Done — recommendations generated.")
