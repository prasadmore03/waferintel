"""
AI-Powered Wafer Defect Detection + RAG Assistant
Enhanced Streamlit app with RAG-based AI assistant for defect explanation
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rag_wafer_assistant import WaferRAGAssistant, DefectExplanation
import time

st.set_page_config(page_title="AI-Powered Wafer Defect Detection + RAG Assistant", layout="wide")

MODEL_DIR = "models"

# MixedWM38 (external) 8 basic defect types, in label index order.
MIXEDWM38_LABELS = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]

def friendly_external_label(raw: str) -> str:
    # Our extracted folders are `class_0`..`class_7`. Map them to names.
    if isinstance(raw, str) and raw.startswith("class_"):
        try:
            idx = int(raw.split("_", 1)[1])
            if 0 <= idx < len(MIXEDWM38_LABELS):
                return f"{MIXEDWM38_LABELS[idx]} ({raw})"
        except Exception:
            return raw
    return raw

def friendly_external_labels(arr) -> np.ndarray:
    return np.array([friendly_external_label(str(x)) for x in arr], dtype=object)

# --- Enhanced UI styling with RAG theme ---
st.markdown(
    """
    <style>
      :root{
        --bg0:#060716;
        --bg1:#0b1030;
        --panel:rgba(18,23,54,0.78);
        --panel2:rgba(16,20,44,0.78);
        --border:rgba(86,104,160,0.35);
        --text:#f3f6ff;
        --muted:#b7c3ea;
        --accent:#8b5cf6;   /* violet */
        --accent2:#22d3ee;  /* cyan */
        --good:#22c55e;
        --warn:#fbbf24;
        --danger:#fb7185;
        --ai:#f59e0b;      /* AI assistant color */
      }
      .stApp{
        background: radial-gradient(1200px circle at 18% -10%, rgba(139,92,246,0.22), transparent 45%),
                    radial-gradient(900px circle at 88% 10%, rgba(34,211,238,0.18), transparent 40%),
                    radial-gradient(600px circle at 50% 50%, rgba(245,158,11,0.15), transparent 50%),
                    linear-gradient(180deg, var(--bg0), var(--bg1) 55%, var(--bg0));
        color: var(--text);
      }
      /* Hide Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4{
        color: var(--text) !important;
      }
      .cardish, .st-expander{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
      }
      .metric-card{
        background: var(--panel2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px 14px;
      }
      .ai-card{
        background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(139,92,246,0.1));
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 14px;
        padding: 16px;
        margin: 10px 0;
      }
      .app-title{
        font-size: 2.05rem;
        font-weight: 900;
        letter-spacing: .2px;
        margin-bottom: 4px;
      }
      .app-subtitle{
        color: var(--muted);
        margin-top: 0px;
      }
      .stButton>button{
        background: linear-gradient(90deg, rgba(139,92,246,0.25), rgba(34,211,238,0.18));
        border: 1px solid rgba(139,92,246,0.45);
        color: var(--text);
        border-radius: 10px;
      }
      .stButton>button:hover{
        background: linear-gradient(90deg, rgba(139,92,246,0.35), rgba(34,211,238,0.26));
        border-color: rgba(34,211,238,0.55);
      }
      .ai-button{
        background: linear-gradient(90deg, rgba(245,158,11,0.25), rgba(139,92,246,0.18));
        border: 1px solid rgba(245,158,11,0.45);
        color: var(--text);
        border-radius: 10px;
      }
      .ai-button:hover{
        background: linear-gradient(90deg, rgba(245,158,11,0.35), rgba(139,92,246,0.26));
        border-color: rgba(245,158,11,0.55);
      }
      /* Tabs */
      button[data-baseweb="tab"]{
        color: var(--muted);
      }
      button[data-baseweb="tab"][aria-selected="true"]{
        color: var(--text) !important;
        border-bottom: 2px solid rgba(34,211,238,0.9);
      }
      /* Dataframe header tint */
      .stDataFrame thead th{
        background: rgba(34,211,238,0.10) !important;
      }
      /* Chat message styling */
      .chat-message{
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 80%;
      }
      .user-message{
        background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(34,211,238,0.1));
        border: 1px solid rgba(139,92,246,0.3);
        margin-left: auto;
        text-align: right;
      }
      .ai-message{
        background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(139,92,246,0.1));
        border: 1px solid rgba(245,158,11,0.3);
        margin-right: auto;
      }
      /* Severity indicators */
      .severity-critical{ color: #ef4444; font-weight: bold; }
      .severity-high{ color: #f59e0b; font-weight: bold; }
      .severity-medium{ color: #22d3ee; font-weight: bold; }
      .severity-low{ color: #22c55e; font-weight: bold; }
      .severity-none{ color: #94a3b8; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_wm_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "wafer_model.h5"))
    classes = np.load(os.path.join(MODEL_DIR, "wafer_class_names.npy"), allow_pickle=True)
    return model, classes

@st.cache_resource
def load_external_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "external_wafer_model.h5"))
    classes = np.load(os.path.join(MODEL_DIR, "external_wafer_classes.npy"), allow_pickle=True)
    return model, classes

@st.cache_resource
def load_external_rf():
    rf_path = os.path.join(MODEL_DIR, "external_baseline_rf.pkl")
    if not os.path.exists(rf_path):
        return None
    return joblib.load(rf_path)

def preprocess_img(uploaded, size):
    img = Image.open(uploaded).convert("L").resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return img, arr[np.newaxis, ..., np.newaxis]

def is_multilabel_external_model(model: tf.keras.Model, classes: np.ndarray) -> bool:
    """Check if model is multi-label"""
    try:
        out_units = int(model.output_shape[-1])
    except Exception:
        return False
    if out_units != len(classes):
        return False
    return not all(str(c).startswith("class_") for c in classes.tolist())

@st.cache_data
def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div style="font-size:.86rem;color:#b7c3ea;margin-bottom:2px;">{title}</div>
          <div style="font-size:1.55rem;font-weight:900;line-height:1.2;">{value}</div>
          <div style="font-size:.86rem;color:#b7c3ea;margin-top:2px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _list_image_files(image_root: str):
    """List image files recursively"""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_root, "**", e), recursive=True))
    files = [p for p in files if os.path.basename(os.path.dirname(p)).startswith("class_")]
    return files

def get_severity_color(severity: str) -> str:
    """Get color class for severity level"""
    severity_colors = {
        "Critical": "severity-critical",
        "High": "severity-high", 
        "Medium": "severity-medium",
        "Low": "severity-low",
        "None": "severity-none"
    }
    return severity_colors.get(severity, "")

# --- Initialize RAG Assistant ---
@st.cache_resource
def initialize_rag_assistant():
    """Initialize RAG Assistant"""
    try:
        assistant = WaferRAGAssistant()  # Use simplified rule-based assistant
        return assistant
    except Exception as e:
        st.error(f"Error initializing RAG Assistant: {e}")
        return None

# --- Header ---
st.markdown(
    """
    <div>
      <div class="app-title">🧠 AI-Powered Wafer Defect Detection + RAG Assistant</div>
      <div class="app-subtitle">Advanced defect analysis with AI-powered explanation system</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="cardish"><b>🔧 Model Status</b><br><span style="color:#9fb0d1;font-size:0.95em;">Ready check for live inference</span></div>', unsafe_allow_html=True)
    
    # Check model availability
    wm_model_ok = os.path.exists(os.path.join(MODEL_DIR, "wafer_model.h5"))
    wm_classes_ok = os.path.exists(os.path.join(MODEL_DIR, "wafer_class_names.npy"))
    ext_model_ok = os.path.exists(os.path.join(MODEL_DIR, "external_wafer_model.h5"))
    ext_classes_ok = os.path.exists(os.path.join(MODEL_DIR, "external_wafer_classes.npy"))
    
    if wm_model_ok and wm_classes_ok:
        st.success("WM-811K model: Ready")
    else:
        st.warning("WM-811K model: Not found")
    
    if ext_model_ok and ext_classes_ok:
        st.success("External model: Ready")
    else:
        st.warning("External model: Not found")
    
    # RAG Assistant Status
    st.markdown('<div class="cardish"><b>🤖 AI Assistant Status</b><br><span style="color:#9fb0d1;font-size:0.95em;">RAG-based defect explanation</span></div>', unsafe_allow_html=True)
    
    if 'rag_assistant' not in st.session_state:
        st.session_state.rag_assistant = initialize_rag_assistant()
    
    if st.session_state.rag_assistant:
        st.success("RAG Assistant: Ready")
    else:
        st.error("RAG Assistant: Not available")

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["🔍 Defect Detection", "📊 Model Insights", "🤖 AI Assistant"])

with tab1:
    st.subheader("🔍 Wafer Defect Detection")
    subtab_inf, subtab_card = st.tabs(["Live Inspection", "Model Card"])

    with subtab_inf:
        st.write("Upload a wafer map image to classify the defect pattern and get AI-powered explanations.")
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            uploaded = st.file_uploader("Upload wafer map image", type=["png", "jpg", "jpeg"], key="wm_rag")
            
            if uploaded is not None:
                preview, x = preprocess_img(uploaded, 32)
                
                if preview is not None and x is not None:
                    st.markdown('<div class="cardish">', unsafe_allow_html=True)
                    st.image(preview, caption="Input wafer map", width=260)
                    
                    conf_threshold = st.slider(
                        "Decision gate (confidence threshold)",
                        0.0,
                        1.0,
                        0.6,
                        0.01,
                        key="wm_conf_threshold_rag",
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if st.button("🔍 Analyze Defect + Get AI Explanation", key="wm_predict_rag", type="primary"):
                        try:
                            wm_model, wm_classes = load_wm_model()
                            
                            if wm_model is not None and wm_classes is not None:
                                with st.spinner("🔍 Analyzing wafer image..."):
                                    probs = wm_model.predict(x, verbose=0)[0]
                                    idx = int(np.argmax(probs))
                                    max_prob = float(probs[idx])
                                    status = "PASS" if max_prob >= conf_threshold else "REVIEW"
                                    
                                    # Store prediction for AI assistant
                                    st.session_state.last_prediction = {
                                        "defect_type": wm_classes[idx],
                                        "confidence": max_prob,
                                        "dataset": "WM811K",
                                        "status": status,
                                        "all_probabilities": dict(zip(wm_classes, probs))
                                    }
                                    
                                    # Display prediction result
                                    st.success(f"Defect: {wm_classes[idx]}  |  Confidence: {max_prob*100:.2f}%  |  Gate: {status}")
                                    
                                    # Generate AI explanation
                                    if st.session_state.rag_assistant:
                                        with st.spinner("🤖 Generating AI explanation..."):
                                            explanation = st.session_state.rag_assistant.generate_explanation(
                                                wm_classes[idx], "WM811K", max_prob
                                            )
                                            st.session_state.last_explanation = explanation
                                            
                                            # Display explanation in AI card
                                            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                                            st.markdown("### 🤖 AI Assistant Analysis")
                                            st.markdown(st.session_state.rag_assistant.format_explanation(explanation), unsafe_allow_html=True)
                                            st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Probability chart
                                    df = (
                                        pd.DataFrame({"class": wm_classes, "probability": probs})
                                        .sort_values("probability", ascending=False)
                                        .reset_index(drop=True)
                                    )
                                    st.markdown("**📊 Top probabilities**")
                                    st.bar_chart(df.set_index("class")["probability"].head(8))
                                    st.dataframe(df, use_container_width=True)
                            else:
                                st.error("WM-811K model not available. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")

        with col2:
            if 'last_prediction' in st.session_state:
                st.markdown('<div class="cardish">', unsafe_allow_html=True)
                st.markdown("### 📋 Recent Analysis")
                
                pred = st.session_state.last_prediction
                st.markdown(f"**Defect:** {pred['defect_type']}")
                st.markdown(f"**Confidence:** {pred['confidence']:.1%}")
                st.markdown(f"**Dataset:** {pred['dataset']}")
                st.markdown(f"**Status:** {pred['status']}")
                
                if 'last_explanation' in st.session_state:
                    exp = st.session_state.last_explanation
                    severity_class = get_severity_color(exp.severity_level)
                    st.markdown(f"**Severity:** <span class='{severity_class}'>{exp.severity_level}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Recommended:** {exp.recommended_action}")
                
                st.markdown('</div>', unsafe_allow_html=True)

    with subtab_card:
        metrics = load_json(os.path.join(MODEL_DIR, "wm811k_metrics.json"))
        if metrics is None:
            st.warning("No saved WM-811K metrics found. Retrain once to generate `models/wm811k_metrics.json`.")
        else:
            c1, c2, c3 = st.columns(3, gap="large")
            macro_auc = metrics.get("macro_auc_ovr", None)
            report = metrics.get("report", {})
            acc = report.get("accuracy", None)
            f1w = report.get("weighted avg", {}).get("f1-score", None)

            with c1:
                metric_card("Macro AUC (OvR)", f"{macro_auc:.3f}" if macro_auc is not None else "—", "multi-class separability")
            with c2:
                metric_card("Accuracy", f"{acc:.3f}" if acc is not None else "—", "overall hit rate")
            with c3:
                metric_card("Weighted F1", f"{f1w:.3f}" if f1w is not None else "—", "imbalance-aware score")

            st.markdown("### Confusion matrix")
            cm_path = os.path.join(MODEL_DIR, metrics.get("confusion_matrix_png", "wm811k_confusion_matrix.png"))
            if os.path.exists(cm_path):
                st.image(cm_path, caption="WM-811K confusion matrix (saved from training)", use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Retrain once to generate it.")

            # Per-class table
            rows = []
            for k, v in report.items():
                if isinstance(v, dict) and "f1-score" in v and k not in ["macro avg", "weighted avg"]:
                    rows.append(
                        {
                            "class": k,
                            "precision": v.get("precision"),
                            "recall": v.get("recall"),
                            "f1": v.get("f1-score"),
                            "support": v.get("support"),
                        }
                    )
            if rows:
                dfc = pd.DataFrame(rows).sort_values("f1")
                st.markdown("### Per-class performance (quantifiable)")
                st.dataframe(dfc, use_container_width=True)

with tab2:
    st.subheader("📊 External Wafer Dataset Insights")
    subtab_inf2, subtab_card2 = st.tabs(["Live Inspection", "Model Card"])

    with subtab_inf2:
        st.write("Dataset: **MixedWM38**. This model can be **multi-label** (one wafer may contain multiple defects).")

        with st.expander("Defect legend (what each class means)", expanded=False):
            st.markdown(
                """
                MixedWM38 uses 8 basic wafer defect types (and can include mixed combinations in original dataset):
                - **Center**
                - **Donut**
                - **Edge-Loc**
                - **Edge-Ring**
                - **Loc**
                - **Near-full**
                - **Random**
                - **Scratch**
                """
            )
            
            # Show one example image per available class folder (if dataset folder exists)
            img_root = "external_wafer_images"
            if os.path.exists(img_root):
                cols = st.columns(4)
                for i in range(8):
                    folder = os.path.join(img_root, f"class_{i}")
                    if not os.path.exists(folder):
                        continue
                    ex_files = _list_image_files(folder)
                    if not ex_files:
                        continue
                    p = sorted(ex_files)[0]
                    img = Image.open(p).convert("L").resize((128, 128))
                    with cols[i % 4]:
                        st.image(img, caption=MIXEDWM38_LABELS[i], use_container_width=True)
            else:
                st.info("Add `external_wafer_images/` to see example images here.")
        
        uploaded2 = st.file_uploader("Upload external wafer map", type=["png", "jpg", "jpeg"], key="ext_rag")

        if uploaded2 is not None:
            left, right = st.columns([1, 1.6], gap="large")
            preview2, x2 = preprocess_img(uploaded2, 128)

            if preview2 is not None and x2 is not None:
                with left:
                    st.markdown('<div class="cardish">', unsafe_allow_html=True)
                    st.image(preview2, caption="Input wafer map (MixedWM38)", width=260)
                    conf_threshold_ext = st.slider(
                        "Decision gate (confidence threshold)",
                        0.0,
                        1.0,
                        0.6,
                        0.01,
                        key="ext_conf_threshold_rag",
                    )
                    defect_threshold = st.slider(
                        "Defect threshold (multi-label)",
                        0.05,
                        0.95,
                        0.35,
                        0.01,
                        key="ext_defect_threshold_rag",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with right:
                    if st.button("🔍 Analyze MixedWM38 + Get AI Explanation", key="ext_predict_rag", type="primary"):
                        try:
                            ext_model, ext_classes = load_external_model()
                            
                            if ext_model is not None and ext_classes is not None:
                                with st.spinner("🔍 Analyzing wafer image..."):
                                    multilabel = is_multilabel_external_model(ext_model, ext_classes)
                                    ext_classes_friendly = (
                                        ext_classes.astype(object) if multilabel else friendly_external_labels(ext_classes)
                                    )
                                    probs2 = ext_model.predict(x2, verbose=0)[0].astype("float32")

                                    if multilabel:
                                        present = np.where(probs2 >= defect_threshold)[0].tolist()
                                        if present:
                                            st.success(
                                                "Detected defects: "
                                                + " • ".join([f"{ext_classes_friendly[i]} ({probs2[i]*100:.1f}%)" for i in present])
                                            )
                                            # Use first detected defect for AI explanation
                                            primary_defect = ext_classes_friendly[present[0]]
                                        else:
                                            top_i = int(np.argmax(probs2))
                                            st.warning(
                                                f"No defect crossed threshold. Top: {ext_classes_friendly[top_i]} ({probs2[top_i]*100:.1f}%)"
                                            )
                                            primary_defect = ext_classes_friendly[top_i]
                                        max_prob2 = float(np.max(probs2))
                                        status2 = "PASS" if max_prob2 >= conf_threshold_ext else "REVIEW"
                                        st.info(f"Decision gate (max-prob): {max_prob2*100:.2f}% -> {status2}")
                                        
                                        # Store multi-label prediction for AI assistant
                                        detected_defects = [ext_classes_friendly[i] for i in present] if present else [ext_classes_friendly[top_i]]
                                        st.session_state.last_prediction = {
                                            "defect_type": primary_defect,
                                            "confidence": max_prob2,
                                            "dataset": "MixedWM38 (Multi-label)",
                                            "status": status2,
                                            "detected_defects": detected_defects,
                                            "all_probabilities": dict(zip(ext_classes_friendly, probs2))
                                        }
                                    else:
                                        idx2 = int(np.argmax(probs2))
                                        max_prob2 = float(probs2[idx2])
                                        status2 = "PASS" if max_prob2 >= conf_threshold_ext else "REVIEW"
                                        st.success(f"Prediction: {ext_classes_friendly[idx2]} ({max_prob2*100:.2f}%) -> {status2}")
                                        
                                        # Store prediction for AI assistant
                                        st.session_state.last_prediction = {
                                            "defect_type": ext_classes_friendly[idx2],
                                            "confidence": max_prob2,
                                            "dataset": "MixedWM38",
                                            "status": status2,
                                            "all_probabilities": dict(zip(ext_classes_friendly, probs2))
                                        }
                                        primary_defect = ext_classes_friendly[idx2]

                                    # Generate AI explanation
                                    if st.session_state.rag_assistant:
                                        with st.spinner("🤖 Generating AI explanation..."):
                                            explanation = st.session_state.rag_assistant.generate_explanation(
                                                primary_defect, "MixedWM38", max_prob2
                                            )
                                            st.session_state.last_explanation = explanation
                                            
                                            # Display explanation in AI card
                                            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                                            st.markdown("### 🤖 AI Assistant Analysis")
                                            st.markdown(st.session_state.rag_assistant.format_explanation(explanation), unsafe_allow_html=True)
                                            st.markdown('</div>', unsafe_allow_html=True)

                                    # Probability distribution
                                    df2 = (
                                        pd.DataFrame({"class": ext_classes_friendly, "probability": probs2})
                                        .sort_values("probability", ascending=False)
                                        .reset_index(drop=True)
                                    )
                                    st.markdown("**Defect probabilities**")
                                    st.bar_chart(df2.set_index("class")["probability"].head(8))
                                    st.dataframe(df2, use_container_width=True)

                                    # CNN vs RF comparison
                                    ext_rf = load_external_rf()
                                    if ext_rf is not None:
                                        x2_flat = x2.reshape(1, -1)
                                        rf_pred_idx = int(ext_rf.predict(x2_flat)[0])
                                        rf_pred_name = (
                                            ext_classes_friendly[rf_pred_idx]
                                            if rf_pred_idx < len(ext_classes_friendly)
                                            else rf_pred_idx
                                        )
                                        agree = False if multilabel else (rf_pred_idx == int(np.argmax(probs2)))
                                        st.info(f"Baseline check (RF dominant label): {rf_pred_name}  |  Agreement: {agree}")
                            else:
                                st.error("External model not available. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error during external prediction: {str(e)}")

    with subtab_card2:
        metrics = load_json(os.path.join(MODEL_DIR, "mixedwm38_metrics.json"))
        if metrics is None:
            st.warning("No saved MixedWM38 metrics found. Retrain once to generate `models/mixedwm38_metrics.json`.")
        else:
            c1, c2, c3 = st.columns(3, gap="large")
            macro_auc = metrics.get("macro_auc", None)
            with c1:
                metric_card("Macro AUC", f"{macro_auc:.3f}" if macro_auc is not None else "—", "multi-label separability")
            with c2:
                metric_card("Task type", "Multi-label", "sigmoid outputs (8 defects)")
            with c3:
                metric_card("Labels", "8 defects", "Center…Scratch")

            st.markdown("### Per-defect AUC (quantifiable)")
            auc_map = metrics.get("auc_per_label", {})
            prev_map = metrics.get("label_prevalence", {})
            rows = []
            for k, v in auc_map.items():
                rows.append({"defect": k, "auc": v, "prevalence": prev_map.get(k)})
            if rows:
                dfm = pd.DataFrame(rows).sort_values("auc")
                st.bar_chart(dfm.set_index("defect")["auc"])
                st.dataframe(dfm, use_container_width=True)

with tab3:
    st.subheader("🤖 AI Assistant - RAG-based Defect Explanation")
    
    if st.session_state.rag_assistant:
        # Chat interface
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Chat with AI Assistant")
        st.markdown("Ask questions about wafer defects, process issues, or get detailed explanations of detected defects.")
        
        # Display chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-message"><strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask about wafer defects...", key="chat_input")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💬 Send", key="send_chat"):
                if user_input.strip():
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Get AI response
                    with st.spinner("🤖 Thinking..."):
                        context = st.session_state.get('last_prediction', None)
                        response = st.session_state.rag_assistant.chat_with_assistant(user_input, context)
                    
                    # Add AI response
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Clear input and refresh
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick question templates
        st.markdown("### 🎯 Quick Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 What is Center defect?", key="quick_center"):
                st.session_state.chat_history.append({"role": "user", "content": "What is Center defect?"})
                response = st.session_state.rag_assistant.chat_with_assistant("What is Center defect?", None)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col2:
            if st.button("🔍 What is Donut defect?", key="quick_donut"):
                st.session_state.chat_history.append({"role": "user", "content": "What is Donut defect?"})
                response = st.session_state.rag_assistant.chat_with_assistant("What is Donut defect?", None)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col3:
            if st.button("🔍 What is Scratch defect?", key="quick_scratch"):
                st.session_state.chat_history.append({"role": "user", "content": "What is Scratch defect?"})
                response = st.session_state.rag_assistant.chat_with_assistant("What is Scratch defect?", None)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Context-aware explanation
        if 'last_prediction' in st.session_state and 'last_explanation' not in st.session_state:
            st.markdown("### 📋 Current Prediction Context")
            st.info("Upload and analyze a wafer image to get AI-powered defect explanation.")
        
        if 'last_explanation' in st.session_state:
            st.markdown("### 📋 Last AI Explanation")
            exp = st.session_state.last_explanation
            st.markdown(f"**Defect:** {exp.defect_name}")
            st.markdown(f"**Severity:** <span class='{get_severity_color(exp.severity_level)}'>{exp.severity_level}</span>", unsafe_allow_html=True)
            st.markdown(f"**Recommended Action:** {exp.recommended_action}")
            
            if st.button("🔄 Get New Explanation", key="new_explanation"):
                if 'last_prediction' in st.session_state:
                    pred = st.session_state.last_prediction
                    explanation = st.session_state.rag_assistant.generate_explanation(
                        pred['defect_type'], pred['dataset'], pred['confidence']
                    )
                    st.session_state.last_explanation = explanation
                    st.rerun()
    
    else:
        st.error("RAG Assistant is not available. Please check the installation.")

# --- Footer ---
st.markdown("""
<div style="background: rgba(30,41,59,0.9); padding: 20px; border-radius: 15px; margin-top: 30px; text-align: center; border: 1px solid rgba(245,158,11,0.3);">
    <p style="margin: 0; color: #94a3b8;">
        🧠 AI-Powered Wafer Defect Detection + RAG Assistant | 
        Built with ❤️ using TensorFlow, Streamlit & RAG Technology
    </p>
</div>
""", unsafe_allow_html=True)
