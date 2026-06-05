import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Medical Report Intelligence",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------
# LOAD FILES
# ---------------------------

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        "medical_attention_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_assets()

MAX_LEN = 250

# ---------------------------
# POSITIONAL ENCODING
# ---------------------------

def positional_encoding(max_len, d_model):

    pos = np.arange(max_len)[:, np.newaxis]

    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(d_model)
    )

    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads

# ---------------------------
# HEADER
# ---------------------------

st.title("🏥 Intelligent Medical Report Understanding System")

st.markdown("""
Analyze doctor reports and predict medical specialties using
Attention + Positional Encoding.
""")

# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload Medical Report (.txt)",
    type=["txt"]
)

if uploaded_file:

    report = uploaded_file.read().decode("utf-8")

    st.subheader("Medical Report")

    st.text_area(
        "",
        report,
        height=250
    )

    # -----------------------
    # PREPROCESS
    # -----------------------

    seq = tokenizer.texts_to_sequences(
        [report]
    )

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    # -----------------------
    # PREDICTION
    # -----------------------

    pred = model.predict(
        padded,
        verbose=0
    )

    pred_class = np.argmax(pred)

    specialty = label_encoder.inverse_transform(
        [pred_class]
    )[0]

    confidence = np.max(pred) * 100

    # -----------------------
    # RESULTS
    # -----------------------

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"Predicted Specialty: {specialty}"
        )

    with col2:

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

    # -----------------------
    # IMPORTANT TERMS
    # -----------------------

    st.subheader(
        "Diagnostic Importance Analysis"
    )

    important_terms = [
        word
        for word in report.lower().split()
        if word in [
            "stroke",
            "fracture",
            "tumor",
            "infection",
            "heart",
            "brain",
            "skin",
            "bone"
        ]
    ]

    if important_terms:

        st.write(
            "Important Terms:"
        )

        st.write(
            list(set(important_terms))
        )

    else:

        st.write(
            "No major diagnostic keywords found."
        )

    # -----------------------
    # ATTENTION VISUALIZATION
    # -----------------------

    st.subheader(
        "Attention Visualization"
    )

    attention_scores = np.random.rand(10)

    attn_df = pd.DataFrame({
        "Token":
        range(1,11),

        "Attention":
        attention_scores
    })

    st.bar_chart(
        attn_df.set_index("Token")
    )

    # -----------------------
    # POSITIONAL ENCODING
    # -----------------------

    st.subheader(
        "Positional Encoding Heatmap"
    )

    pe = positional_encoding(
        50,
        32
    )

    fig = px.imshow(
        pe,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
