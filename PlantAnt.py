import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ====================== DESIGN ======================
st.set_page_config(
    page_title="Pflanzen-Detektor",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; text-align: center;}
    .result-box {
        background-color: #1e3a2f; 
        color: white;
        padding: 1.8em; 
        border-radius: 12px; 
        margin: 1.2em 0;
    }
    .footer {text-align: center; color: #555; margin-top: 4em; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – Bäume & Blumen 🌸")
st.markdown("**Selbst trainiert mit Teachable Machine (12 Arten)**")

# ====================== PFLANZEN-DATEN (nur Wiki-Links) ======================
PLANT_DATA = {
    "Birke": {"wiki": "https://de.wikipedia.org/wiki/Hänge-Birke"},
    "Gemeine Fichte": {"wiki": "https://de.wikipedia.org/wiki/Gemeine_Fichte"},
    "Gemeine Kiefer": {"wiki": "https://de.wikipedia.org/wiki/Waldkiefer"},
    "Rotbuche": {"wiki": "https://de.wikipedia.org/wiki/Rotbuche"},
    "Stieleiche": {"wiki": "https://de.wikipedia.org/wiki/Stieleiche"},
    "Traubeneiche": {"wiki": "https://de.wikipedia.org/wiki/Traubeneiche"},
    "Gänseblümchen": {"wiki": "https://de.wikipedia.org/wiki/G%C3%A4nsebl%C3%BCmchen"},
    "Glockenblume": {"wiki": "https://de.wikipedia.org/wiki/Glockenblumen"},
    "Lavendel": {"wiki": "https://de.wikipedia.org/wiki/Lavendel"},
    "Rittersporn": {"wiki": "https://de.wikipedia.org/wiki/Rittersporn"},
    "Sonnenblume": {"wiki": "https://de.wikipedia.org/wiki/Sonnenblume"},
    "Vergissmeinnicht": {"wiki": "https://de.wikipedia.org/wiki/Vergissmeinnicht"}
}

# ====================== MODELL LADEN ======================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        st.success("✅ Modell erfolgreich geladen!")
        return model, labels
    except Exception as e:
        st.error(f"❌ Modell konnte nicht geladen werden: {str(e)}")
        return None, None

model, labels = load_model()

# ====================== HAUPTBEREICH ======================
tab1, tab2 = st.tabs(["🔍 Erkennung starten", "📋 Meine 12 Arten"])

with tab1:
    st.subheader("Foto hochladen oder mit der Kamera aufnehmen")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Bild hochladen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with col2:
        camera_file = st.camera_input("Live-Kamera")

    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
    elif camera_file is not None:
        input_image = Image.open(camera_file)

    if input_image is not None and model is not None and labels is not None:
        st.image(input_image, caption="Dein Bild", use_column_width=True)

        # Vorhersage
        img = input_image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=False)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx] * 100)
        
        # Zahl entfernen
        raw_label = labels[class_idx]
        predicted_label = raw_label.split('. ', 1)[-1]

        # Ergebnis mit direktem Wikipedia-Link
        if predicted_label in PLANT_DATA:
            wiki_link = PLANT_DATA[predicted_label]["wiki"]
            st.markdown(f"""
            <div class="result-box">
                <h3>Erkannt: <strong>{predicted_label}</strong> 
                <a href="{wiki_link}" target="_blank" style="color: #90EE90; margin-left: 15px;">
                [Wikipedia]
                </a></h3>
                <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box">
                <h3>Erkannt: <strong>{predicted_label}</strong></h3>
                <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Meine 12 trainierten Arten")
    for name in PLANT_DATA.keys():
        st.write(f"• {name}")

# Footer
st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
