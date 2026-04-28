import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ====================== STREAMLIT CONFIG & DESIGN ======================
st.set_page_config(
    page_title="Pflanzen-Detektor",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; text-align: center; font-family: 'Helvetica Neue', Arial;}
    h2, h3 {color: #2E8B57;}
    .result-box {
        background-color: #e8f5e9; 
        padding: 1.8em; 
        border-radius: 12px; 
        border-left: 8px solid #228B22;
        margin: 1em 0;
    }
    .info-box {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 10px;
        border: 1px solid #90EE90;
    }
    .footer {text-align: center; color: #555; margin-top: 4em; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – Bäume & Blumen 🌸")
st.markdown("**Selbst trainiert mit Teachable Machine (12 Arten)**")

# ====================== DEINE 12 ARTEN + DATEN ======================
PLANT_DATA = {
    "Birke": {"de": "Birke", "bot": "Betula pendula", "pflege": "Boden: sandig bis lehmig. Licht: vollsonnig. Wasser: mäßig. Standort: Pionierbaum auf offenen Flächen.", "wiki": "https://de.wikipedia.org/wiki/Hänge-Birke"},
    "Gemeine Fichte": {"de": "Gemeine Fichte", "bot": "Picea abies", "pflege": "Boden: frisch, nährstoffreich. Licht: halbschattig. Wasser: mäßig. Standort: Berg- und Hügelländer.", "wiki": "https://de.wikipedia.org/wiki/Gemeine_Fichte"},
    "Gemeine Kiefer": {"de": "Gemeine Kiefer", "bot": "Pinus sylvestris", "pflege": "Boden: sandig, nährstoffarm. Licht: vollsonnig. Wasser: sehr gering. Standort: Kiefernheiden.", "wiki": "https://de.wikipedia.org/wiki/Waldkiefer"},
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
        # Wichtig: compile=False verhindert viele Deserialisierungs-Probleme
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        st.success("✅ Teachable Machine Modell erfolgreich geladen!")
        return model, labels
    except Exception as e:
        st.error(f"❌ Modell konnte nicht geladen werden: {e}")
        st.info("Tipp: Verwende tensorflow==2.15.0 oder tensorflow-cpu==2.15.0 in requirements.txt")
        return None, None

model, labels = load_model()

# ====================== HAUPTBEREICH ======================
tab1, tab2, tab3 = st.tabs(["🔍 Erkennung starten", "📋 Meine 12 Arten", "ℹ️ Hinweise"])

with tab1:
    st.subheader("Foto hochladen oder Kamera nutzen")

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

        # Vorverarbeitung
        img = input_image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Vorhersage
        prediction = model.predict(img_array, verbose=False)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx] * 100)
        predicted_label = labels[class_idx]

        # Ergebnis – jetzt mit besserem Kontrast
        st.markdown(f"""
        <div class="result-box">
            <h3>Erkannt: <strong>{predicted_label}</strong></h3>
            <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        # Pflanzen-Informationen anzeigen
        if predicted_label in PLANT_DATA:
            data = PLANT_DATA[predicted_label]
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader(f"ℹ️ Informationen zu {data['de']}")
            st.write(f"**Botanischer Name:** {data['bot']}")
            st.write(f"**Pflegetipps (deutsches Klima):** {data['pflege']}")
            st.markdown(f"[→ Mehr auf Wikipedia erfahren]({data['wiki']})")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Keine zusätzlichen Informationen zu dieser Art hinterlegt.")

with tab2:
    st.subheader("Deine trainierten 12 Arten")
    for name in PLANT_DATA.keys():
        st.write(f"• **{name}**")

with tab3:
    st.info("""
    **Hinweis:**  
    Diese App erkennt aktuell 12 Arten.  
    Für bessere Ergebnisse Fotos mit klarem Hintergrund und guter Beleuchtung verwenden.
    """)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
