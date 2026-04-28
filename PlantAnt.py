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
    h2, h3 {color: #2E8B57;}
    
    .result-box {
        background-color: #1e3a2f; 
        color: white;
        padding: 1.8em; 
        border-radius: 12px; 
        margin: 1.2em 0;
    }
    .info-box {
        background-color: #ffffff;
        padding: 1.8em;
        border-radius: 10px;
        border: 2px solid #228B22;
        margin-top: 1.5em;
    }
    .footer {text-align: center; color: #555; margin-top: 4em; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – Bäume & Blumen 🌸")
st.markdown("**Selbst trainiert mit Teachable Machine (12 Arten)**")

# ====================== PFLANZEN-DATEN (mit deutschen Tipps) ======================
PLANT_DATA = {
    "Birke": {
        "de": "Birke",
        "bot": "Betula pendula",
        "pflege": "Birken bevorzugen sonnige bis halbschattige Standorte mit durchlässigem, eher sandigem Boden. Sie sind anspruchslos und vertragen Trockenheit gut. Ideal als Pionierbaum auf offenen Flächen oder in Gärten.",
        "wiki": "https://de.wikipedia.org/wiki/Hänge-Birke"
    },
    "Gemeine Fichte": {
        "de": "Gemeine Fichte",
        "bot": "Picea abies",
        "pflege": "Die Gemeine Fichte liebt frische, nährstoffreiche und leicht saure Böden. Sie bevorzugt halbschattige bis schattige Lagen und braucht regelmäßige Feuchtigkeit, besonders in den ersten Jahren.",
        "wiki": "https://de.wikipedia.org/wiki/Gemeine_Fichte"
    },
    "Gemeine Kiefer": {
        "de": "Gemeine Kiefer",
        "bot": "Pinus sylvestris",
        "pflege": "Kiefern sind sehr genügsam und wachsen am besten auf sandigen, nährstoffarmen und trockenen Böden in voller Sonne. Sie sind extrem trockenheitsverträglich und winterhart.",
        "wiki": "https://de.wikipedia.org/wiki/Waldkiefer"
    },
    "Rotbuche": {
        "de": "Rotbuche",
        "bot": "Fagus sylvatica",
        "pflege": "Rotbuchen gedeihen am besten auf fruchtbaren, leicht feuchten und humusreichen Böden. Sie vertragen Halbschatten gut und sind in ganz Deutschland weit verbreitet. Sehr schattentolerant im Unterwuchs.",
        "wiki": "https://de.wikipedia.org/wiki/Rotbuche"
    },
    "Stieleiche": {
        "de": "Stieleiche",
        "bot": "Quercus robur",
        "pflege": "Stieleichen lieben tiefgründige, frische bis feuchte Böden und sonnige bis halbschattige Standorte. Sie sind sehr langlebig und gut für offene Landschaften und Parks geeignet.",
        "wiki": "https://de.wikipedia.org/wiki/Stieleiche"
    },
    "Traubeneiche": {
        "de": "Traubeneiche",
        "bot": "Quercus petraea",
        "pflege": "Die Traubeneiche bevorzugt eher trockene, saure und durchlässige Böden auf Hügeln und in Mittelgebirgen. Sie ist lichtbedürftig und sehr robust.",
        "wiki": "https://de.wikipedia.org/wiki/Traubeneiche"
    },
    "Gänseblümchen": {
        "de": "Gänseblümchen",
        "bot": "Bellis perennis",
        "pflege": "Gänseblümchen wachsen fast überall auf normalem Garten- oder Rasenboden. Sie mögen sonnige bis halbschattige Plätze und brauchen nur mäßige Feuchtigkeit.",
        "wiki": "https://de.wikipedia.org/wiki/G%C3%A4nsebl%C3%BCmchen"
    },
    "Glockenblume": {
        "de": "Glockenblume",
        "bot": "Campanula spec.",
        "pflege": "Glockenblumen bevorzugen durchlässigen, nährstoffreichen Boden und sonnige bis halbschattige Standorte. Sie eignen sich gut für Staudenbeete.",
        "wiki": "https://de.wikipedia.org/wiki/Glockenblumen"
    },
    "Lavendel": {
        "de": "Lavendel",
        "bot": "Lavandula angustifolia",
        "pflege": "Lavendel braucht viel Sonne und einen trockenen, sandigen bis kiesigen Boden. Staunässe ist unbedingt zu vermeiden. Sehr bienenfreundlich.",
        "wiki": "https://de.wikipedia.org/wiki/Lavendel"
    },
    "Rittersporn": {
        "de": "Rittersporn",
        "bot": "Delphinium spec.",
        "pflege": "Rittersporn liebt nährstoffreiche, tiefgründige Böden und volle Sonne. Er braucht regelmäßige Feuchtigkeit und sollte vor Wind geschützt stehen.",
        "wiki": "https://de.wikipedia.org/wiki/Rittersporn"
    },
    "Sonnenblume": {
        "de": "Sonnenblume",
        "bot": "Helianthus annuus",
        "pflege": "Sonnenblumen brauchen viel Sonne und einen nährstoffreichen, lockeren Boden. Regelmäßiges Gießen ist besonders während des Wachstums wichtig.",
        "wiki": "https://de.wikipedia.org/wiki/Sonnenblume"
    },
    "Vergissmeinnicht": {
        "de": "Vergissmeinnicht",
        "bot": "Myosotis spec.",
        "pflege": "Vergissmeinnicht bevorzugen feuchte, humusreiche Böden und halbschattige Standorte. Sie eignen sich gut für Beet- und Teichränder.",
        "wiki": "https://de.wikipedia.org/wiki/Vergissmeinnicht"
    }
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
        predicted_label = labels[class_idx]

        # Ergebnis-Box (dunkel mit weißem Text)
        st.markdown(f"""
        <div class="result-box">
            <h3>Erkannt: <strong>{predicted_label}</strong></h3>
            <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        # Pflanzen-Informationen
        if predicted_label in PLANT_DATA:
            data = PLANT_DATA[predicted_label]
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader(f"{data['de']} ({data['bot']})")
            st.write(f"**Pflegetipps für das deutsche Klima:**")
            st.write(data['pflege'])
            st.markdown(f"🔗 [Mehr auf Wikipedia erfahren]({data['wiki']})")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Deine trainierten 12 Arten")
    for name in PLANT_DATA.keys():
        data = PLANT_DATA[name]
        st.write(f"• **{name}** ({data['bot']})")

# Footer
st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
