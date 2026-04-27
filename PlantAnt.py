import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ====================== DESIGN ======================
st.set_page_config(page_title="Pflanzen-Detektor", page_icon="🌳", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; text-align: center;}
    h2, h3 {color: #2E8B57;}
    .footer {text-align: center; color: #555; margin-top: 3em;}
    .result-box {background-color: #e8f5e9; padding: 1.5em; border-radius: 10px; border-left: 6px solid #228B22;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – Bäume & Blumen 🌸")
st.markdown("**Selbst trainiert mit Teachable Machine (12 Arten)**")

# ====================== DEINE 12 ARTEN + DATEN ======================
PLANT_DATA = {
    "Birke": {"de": "Birke", "bot": "Betula pendula", "pflege": "Boden: sandig bis lehmig. Licht: vollsonnig. Wasser: mäßig. Standort: Pionierbaum auf offenen Flächen.", "wiki": "https://de.wikipedia.org/wiki/Hänge-Birke"},
    "Gemeine Fichte": {"de": "Gemeine Fichte", "bot": "Picea abies", "pflege": "Boden: frisch, nährstoffreich. Licht: halbschattig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Gemeine_Fichte"},
    "Gemeine Kiefer": {"de": "Gemeine Kiefer", "bot": "Pinus sylvestris", "pflege": "Boden: sandig, nährstoffarm. Licht: vollsonnig. Wasser: sehr gering.", "wiki": "https://de.wikipedia.org/wiki/Waldkiefer"},
    "Rotbuche": {"de": "Rotbuche", "bot": "Fagus sylvatica", "pflege": "Boden: fruchtbar, leicht sauer bis neutral. Licht: halbschattig bis sonnig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Rotbuche"},
    "Stieleiche": {"de": "Stieleiche", "bot": "Quercus robur", "pflege": "Boden: tiefgründig, feucht bis frisch. Licht: sonnig bis halbschattig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Stieleiche"},
    "Traubeneiche": {"de": "Traubeneiche", "bot": "Quercus petraea", "pflege": "Boden: eher trocken, sauer. Licht: sonnig. Wasser: gering.", "wiki": "https://de.wikipedia.org/wiki/Traubeneiche"},
    "Gänseblümchen": {"de": "Gänseblümchen", "bot": "Bellis perennis", "pflege": "Boden: normaler Gartenboden. Licht: sonnig bis halbschattig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/G%C3%A4nsebl%C3%BCmchen"},
    "Glockenblume": {"de": "Glockenblume", "bot": "Campanula spec.", "pflege": "Boden: durchlässig. Licht: sonnig bis halbschattig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Glockenblumen"},
    "Lavendel": {"de": "Lavendel", "bot": "Lavandula angustifolia", "pflege": "Boden: trocken, sandig. Licht: vollsonnig. Wasser: sehr gering.", "wiki": "https://de.wikipedia.org/wiki/Lavendel"},
    "Rittersporn": {"de": "Rittersporn", "bot": "Delphinium spec.", "pflege": "Boden: nährstoffreich. Licht: sonnig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Rittersporn"},
    "Sonnenblume": {"de": "Sonnenblume", "bot": "Helianthus annuus", "pflege": "Boden: nährstoffreich. Licht: vollsonnig. Wasser: mäßig.", "wiki": "https://de.wikipedia.org/wiki/Sonnenblume"},
    "Vergissmeinnicht": {"de": "Vergissmeinnicht", "bot": "Myosotis spec.", "pflege": "Boden: feucht. Licht: halbschattig. Wasser: hoch.", "wiki": "https://de.wikipedia.org/wiki/Vergissmeinnicht"}
}

# ====================== MODELL LADEN ======================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5")
        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        st.info("Bitte lege 'keras_model.h5' und 'labels.txt' ins Root-Verzeichnis.")
        return None, None

model, labels = load_model()

# ====================== HAUPT-APP ======================
tab1, tab2, tab3 = st.tabs(["🔍 Erkennung", "📋 Meine 12 Arten", "ℹ️ Hinweise"])

with tab1:
    st.subheader("Bild hochladen oder Kamera")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])
    with col2:
        camera = st.camera_input("Live-Kamera")

    input_image = None
    if uploaded is not None:
        input_image = Image.open(uploaded)
    elif camera is not None:
        input_image = Image.open(camera)

    if input_image is not None and model is not None:
        st.image(input_image, caption="Dein Bild", use_column_width=True)

        # Vorverarbeitung
        img = input_image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Vorhersage
        pred = model.predict(img_array, verbose=False)
        idx = np.argmax(pred[0])
        confidence = float(pred[0][idx] * 100)
        predicted = labels[idx]

        st.markdown(f"""
        <div class="result-box">
            <h3>Erkannt: <strong>{predicted}</strong></h3>
            <p>Sicherheit: {confidence:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        if predicted in PLANT_DATA:
            d = PLANT_DATA[predicted]
            st.write(f"**Botanischer Name:** {d['bot']}")
            st.write(f"**Pflegetipps:** {d['pflege']}")
            st.markdown(f"[→ Wikipedia]({d['wiki']})")

with tab2:
    st.subheader("Deine trainierten 12 Arten")
    for name in PLANT_DATA.keys():
        st.write(f"• {name}")

with tab3:
    st.info("""
    **Tipp:** Mit nur 12 Arten sollte die Erkennungsgenauigkeit schon recht gut sein.  
    Teste das Modell mit Bildern, die du **nicht** zum Training verwendet hast.
    """)

st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
