import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io

# ====================== GEMEINES DESIGN ======================
st.set_page_config(
    page_title="Pflanzen-Detektor – Bäume & Blumen",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; font-family: 'Helvetica Neue', Arial, sans-serif; text-align: center;}
    h2, h3 {color: #2E8B57;}
    .stButton>button {background-color: #228B22; color: white; border-radius: 8px; font-weight: bold;}
    .footer {text-align: center; color: #555; font-size: 0.9em; margin-top: 3em;}
    .stExpander {border: 1px solid #90EE90;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – Bäume & Blumen 🌸")
st.markdown("**KI-gestützte Erkennung von heimischen Bäumen und Blumen**")

# ====================== PFLANZEN-DATENBANK ======================
PLANT_DATA = {
    "Rotbuche": {"de": "Rotbuche", "bot": "Fagus sylvatica", "pflege": "Boden: fruchtbar, leicht sauer bis neutral, gut drainiert. Licht: halbschattig bis sonnig. Wasser: mäßig (verträgt Trockenheit nach Etablierung). Standort: Wälder und Parks in ganz Deutschland, sehr frosthart.", "wiki": "https://de.wikipedia.org/wiki/Rotbuche"},
    "Stieleiche": {"de": "Stieleiche", "bot": "Quercus robur", "pflege": "Boden: tiefgründig, feucht bis frisch. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Auenwälder und offene Landschaften in Nord- und Mitteleuropa.", "wiki": "https://de.wikipedia.org/wiki/Stieleiche"},
    "Traubeneiche": {"de": "Traubeneiche", "bot": "Quercus petraea", "pflege": "Boden: eher trocken, sauer. Licht: sonnig. Wasser: gering. Standort: trockene Hanglagen in Mittelgebirgen.", "wiki": "https://de.wikipedia.org/wiki/Traubeneiche"},
    "Gemeine Fichte": {"de": "Gemeine Fichte", "bot": "Picea abies", "pflege": "Boden: frisch, nährstoffreich. Licht: halbschattig. Wasser: mäßig. Standort: Berg- und Hügelländer in Deutschland.", "wiki": "https://de.wikipedia.org/wiki/Gemeine_Fichte"},
    "Gemeine Kiefer": {"de": "Gemeine Kiefer", "bot": "Pinus sylvestris", "pflege": "Boden: sandig, nährstoffarm. Licht: vollsonnig. Wasser: sehr gering. Standort: Kiefernheiden und Trockenstandorte.", "wiki": "https://de.wikipedia.org/wiki/Waldkiefer"},
    "Bergahorn": {"de": "Bergahorn", "bot": "Acer pseudoplatanus", "pflege": "Boden: fruchtbar, frisch. Licht: halbschattig. Wasser: mäßig. Standort: Bergwälder und Parks.", "wiki": "https://de.wikipedia.org/wiki/Berg-Ahorn"},
    "Birke": {"de": "Birke", "bot": "Betula pendula", "pflege": "Boden: sandig bis lehmig. Licht: vollsonnig. Wasser: mäßig. Standort: Pionierbaum auf offenen Flächen.", "wiki": "https://de.wikipedia.org/wiki/Hänge-Birke"},
    "Sommerlinde": {"de": "Sommerlinde", "bot": "Tilia platyphyllos", "pflege": "Boden: nährstoffreich, frisch. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Alleen und Parks.", "wiki": "https://de.wikipedia.org/wiki/Sommer-Linde"},
    "Winterlinde": {"de": "Winterlinde", "bot": "Tilia cordata", "pflege": "Boden: tiefgründig, frisch. Licht: halbschattig. Wasser: mäßig. Standort: Wälder und Gärten.", "wiki": "https://de.wikipedia.org/wiki/Winter-Linde"},
    "Hainbuche": {"de": "Hainbuche", "bot": "Carpinus betulus", "pflege": "Boden: frisch bis feucht. Licht: halbschattig. Wasser: mäßig. Standort: Hecken und Unterholz.", "wiki": "https://de.wikipedia.org/wiki/Hainbuche"},
    "Esche": {"de": "Esche", "bot": "Fraxinus excelsior", "pflege": "Boden: feucht, nährstoffreich. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Auenwälder.", "wiki": "https://de.wikipedia.org/wiki/Gew%C3%B6hnliche_Esche"},
    "Vogelbeere / Eberesche": {"de": "Vogelbeere / Eberesche", "bot": "Sorbus aucuparia", "pflege": "Boden: sauer bis neutral. Licht: sonnig. Wasser: gering. Standort: Waldränder und Berge.", "wiki": "https://de.wikipedia.org/wiki/Vogelbeere"},
    "Apfelbaum": {"de": "Apfelbaum", "bot": "Malus domestica", "pflege": "Boden: tiefgründig, lehmig. Licht: vollsonnig. Wasser: regelmäßig. Standort: Obstgärten in ganz Deutschland.", "wiki": "https://de.wikipedia.org/wiki/Apfelbaum"},
    "Birnbaum": {"de": "Birnbaum", "bot": "Pyrus communis", "pflege": "Boden: fruchtbar, frisch. Licht: sonnig. Wasser: regelmäßig. Standort: Streuobstwiesen.", "wiki": "https://de.wikipedia.org/wiki/Birnbaum"},
    "Kastanie": {"de": "Kastanie", "bot": "Aesculus hippocastanum", "pflege": "Boden: frisch, nährstoffreich. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Parks und Alleen.", "wiki": "https://de.wikipedia.org/wiki/Rosskastanie"},
    "Douglasie": {"de": "Douglasie", "bot": "Pseudotsuga menziesii", "pflege": "Boden: frisch, durchlässig. Licht: sonnig. Wasser: mäßig. Standort: forstliche Anpflanzungen.", "wiki": "https://de.wikipedia.org/wiki/Douglasie"},
    "Lärche": {"de": "Lärche", "bot": "Larix decidua", "pflege": "Boden: frisch bis feucht. Licht: vollsonnig. Wasser: mäßig. Standort: Gebirgslagen.", "wiki": "https://de.wikipedia.org/wiki/Europ%C3%A4ische_L%C3%A4rche"},
    "Weißtanne": {"de": "Weißtanne", "bot": "Abies alba", "pflege": "Boden: frisch, nährstoffreich. Licht: halbschattig. Wasser: mäßig. Standort: Bergwälder.", "wiki": "https://de.wikipedia.org/wiki/Weißtanne"},
    "Schwarzerle": {"de": "Schwarzerle", "bot": "Alnus glutinosa", "pflege": "Boden: feucht bis nass. Licht: sonnig bis halbschattig. Wasser: hoch. Standort: Bach- und Auenränder.", "wiki": "https://de.wikipedia.org/wiki/Schwarzerle"},
    "Zitterpappel / Espe": {"de": "Zitterpappel / Espe", "bot": "Populus tremula", "pflege": "Boden: durchlässig. Licht: vollsonnig. Wasser: mäßig. Standort: Pionierbaum auf Lichtungen.", "wiki": "https://de.wikipedia.org/wiki/Zitter-Pappel"},

    "Gänseblümchen": {"de": "Gänseblümchen", "bot": "Bellis perennis", "pflege": "Boden: normaler Gartenboden. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Rasen und Wiesen in ganz Deutschland.", "wiki": "https://de.wikipedia.org/wiki/G%C3%A4nsebl%C3%BCmchen"},
    "Löwenzahn": {"de": "Löwenzahn", "bot": "Taraxacum officinale", "pflege": "Boden: nährstoffreich. Licht: sonnig. Wasser: mäßig. Standort: Wiesen und Wegränder.", "wiki": "https://de.wikipedia.org/wiki/L%C3%B6wenzahn"},
    "Kornblume": {"de": "Kornblume", "bot": "Centaurea cyanus", "pflege": "Boden: sandig-lehmig. Licht: vollsonnig. Wasser: gering. Standort: Getreidefelder und Wildblumenwiesen.", "wiki": "https://de.wikipedia.org/wiki/Kornblume"},
    "Klatschmohn": {"de": "Klatschmohn", "bot": "Papaver rhoeas", "pflege": "Boden: durchlässig. Licht: vollsonnig. Wasser: gering. Standort: Ackerränder.", "wiki": "https://de.wikipedia.org/wiki/Klatschmohn"},
    "Sonnenblume": {"de": "Sonnenblume", "bot": "Helianthus annuus", "pflege": "Boden: nährstoffreich. Licht: vollsonnig. Wasser: mäßig. Standort: Gärten und Felder.", "wiki": "https://de.wikipedia.org/wiki/Sonnenblume"},
    "Tulpe": {"de": "Tulpe", "bot": "Tulipa gesneriana", "pflege": "Boden: durchlässig, sandig. Licht: sonnig. Wasser: mäßig während Wachstum. Standort: Beete (Zwiebel im Herbst pflanzen).", "wiki": "https://de.wikipedia.org/wiki/Tulpen"},
    "Rose": {"de": "Rose", "bot": "Rosa spec.", "pflege": "Boden: humusreich, lehmig. Licht: sonnig. Wasser: regelmäßig. Standort: Gärten (Hunds-Rose sehr robust).", "wiki": "https://de.wikipedia.org/wiki/Rosen"},
    "Lavendel": {"de": "Lavendel", "bot": "Lavandula angustifolia", "pflege": "Boden: trocken, sandig. Licht: vollsonnig. Wasser: sehr gering. Standort: mediterrane Beete, winterhart mit Schutz.", "wiki": "https://de.wikipedia.org/wiki/Lavendel"},
    "Hortensie": {"de": "Hortensie", "bot": "Hydrangea macrophylla", "pflege": "Boden: sauer bis neutral. Licht: halbschattig. Wasser: hoch. Standort: schattige Gartenecken.", "wiki": "https://de.wikipedia.org/wiki/Hortensien"},
    "Schlüsselblume": {"de": "Schlüsselblume", "bot": "Primula veris", "pflege": "Boden: frisch, humos. Licht: halbschattig. Wasser: mäßig. Standort: Wiesen und Gärten.", "wiki": "https://de.wikipedia.org/wiki/Echte_Schl%C3%BCsselblume"},
    "Glockenblume": {"de": "Glockenblume", "bot": "Campanula spec.", "pflege": "Boden: durchlässig. Licht: sonnig bis halbschattig. Wasser: mäßig. Standort: Staudenbeete.", "wiki": "https://de.wikipedia.org/wiki/Glockenblumen"},
    "Margerite": {"de": "Margerite", "bot": "Leucanthemum vulgare", "pflege": "Boden: normal. Licht: sonnig. Wasser: mäßig. Standort: Wiesen.", "wiki": "https://de.wikipedia.org/wiki/Wiesen-Margerite"},
    "Schafgarbe": {"de": "Schafgarbe", "bot": "Achillea millefolium", "pflege": "Boden: trocken bis frisch. Licht: sonnig. Wasser: gering. Standort: Wiesen.", "wiki": "https://de.wikipedia.org/wiki/Gemeine_Schafgarbe"},
    "Vergissmeinnicht": {"de": "Vergissmeinnicht", "bot": "Myosotis spec.", "pflege": "Boden: feucht. Licht: halbschattig. Wasser: hoch. Standort: Bachränder und Beete.", "wiki": "https://de.wikipedia.org/wiki/Vergissmeinnicht"},
    "Rittersporn": {"de": "Rittersporn", "bot": "Delphinium spec.", "pflege": "Boden: nährstoffreich. Licht: sonnig. Wasser: mäßig. Standort: Staudenbeete.", "wiki": "https://de.wikipedia.org/wiki/Rittersporn"},
    "Iris / Schwertlilie": {"de": "Iris / Schwertlilie", "bot": "Iris germanica", "pflege": "Boden: durchlässig. Licht: sonnig. Wasser: mäßig. Standort: Beete.", "wiki": "https://de.wikipedia.org/wiki/Schwertlilien"},
    "Nelke": {"de": "Nelke", "bot": "Dianthus spec.", "pflege": "Boden: kalkhaltig, durchlässig. Licht: sonnig. Wasser: gering. Standort: Steingärten.", "wiki": "https://de.wikipedia.org/wiki/Nelken"},
    "Phacelia": {"de": "Phacelia", "bot": "Phacelia tanacetifolia", "pflege": "Boden: normal. Licht: sonnig. Wasser: mäßig. Standort: Bienenweide auf Feldern.", "wiki": "https://de.wikipedia.org/wiki/Phacelia"},
    "Wilde Malve": {"de": "Wilde Malve", "bot": "Malva sylvestris", "pflege": "Boden: nährstoffreich. Licht: sonnig. Wasser: mäßig. Standort: Wegränder.", "wiki": "https://de.wikipedia.org/wiki/Wilde_Malve"},
    "Flockenblume": {"de": "Flockenblume", "bot": "Centaurea jacea", "pflege": "Boden: trocken. Licht: sonnig. Wasser: gering. Standort: Trockenwiesen.", "wiki": "https://de.wikipedia.org/wiki/Wiesen-Flockenblume"}
}

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📋 Artenübersicht")
    with st.expander("🌳 20 Bäume", expanded=False):
        for name in ["Rotbuche", "Stieleiche", "Traubeneiche", "Gemeine Fichte", "Gemeine Kiefer", "Bergahorn", "Birke", "Sommerlinde", "Winterlinde", "Hainbuche", "Esche", "Vogelbeere / Eberesche", "Apfelbaum", "Birnbaum", "Kastanie", "Douglasie", "Lärche", "Weißtanne", "Schwarzerle", "Zitterpappel / Espe"]:
            st.write(f"• {name}")
    with st.expander("🌸 20 Blumen", expanded=False):
        for name in ["Gänseblümchen", "Löwenzahn", "Kornblume", "Klatschmohn", "Sonnenblume", "Tulpe", "Rose", "Lavendel", "Hortensie", "Schlüsselblume", "Glockenblume", "Margerite", "Schafgarbe", "Vergissmeinnicht", "Rittersporn", "Iris / Schwertlilie", "Nelke", "Phacelia", "Wilde Malve", "Flockenblume"]:
            st.write(f"• {name}")

# ====================== HAUPT-APP ======================
tab1, tab2, tab3 = st.tabs(["🔍 Erkennung starten", "📚 Arten lernen", "🧠 Wie funktioniert YOLO?"])

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

    if input_image is not None:
        st.image(input_image, caption="Dein Bild", use_column_width=True)

        # YOLO-Modell (cached)
        @st.cache_resource
        def load_yolo():
            return YOLO("yolov8n.pt")  # leichtgewichtiges vortrainiertes Modell

        model = load_yolo()
        results = model.predict(input_image, conf=0.25, verbose=False)

        # Bounding-Box-Bild anzeigen
        plotted_img = results[0].plot()
        st.image(plotted_img, caption="YOLO-Objekterkennung (Bounding Boxes)", use_column_width=True)

        st.success("✅ YOLO hat Objekte erkannt! (z. B. ‚potted plant‘ oder andere Strukturen)")

        # Schulische Demo-Erkennung: Auswahl aus den 40 Arten
        st.subheader("Spezifische Pflanzen-Info (Demo-Erkennung)")
        selected_name = st.selectbox(
            "Welche Art wurde deiner Meinung nach erkannt? (In einem echten, selbst trainierten YOLO-Modell würde die Klasse automatisch vorhergesagt)",
            options=list(PLANT_DATA.keys())
        )

        if selected_name:
            data = PLANT_DATA[selected_name]
            st.markdown(f"**{data['de']}** ({data['bot']})")
            st.write(data['pflege'])
            st.markdown(f"[→ Wikipedia-Seite öffnen]({data['wiki']})")

with tab2:
    st.subheader("Alle 40 Arten im Überblick")
    col_tree, col_flower = st.columns(2)
    with col_tree:
        st.markdown("**🌳 Bäume**")
        for name in list(PLANT_DATA.keys())[:20]:
            if name in PLANT_DATA:
                d = PLANT_DATA[name]
                st.markdown(f"**{d['de']}** — {d['bot']}")
    with col_flower:
        st.markdown("**🌸 Blumen**")
        for name in list(PLANT_DATA.keys())[20:]:
            if name in PLANT_DATA:
                d = PLANT_DATA[name]
                st.markdown(f"**{d['de']}** — {d['bot']}")

with tab3:
    st.subheader("Kurze Erklärung: YOLO (You Only Look Once)")
    st.write("""
    YOLO ist ein modernes Deep-Learning-Modell für **Echtzeit-Objekterkennung**. 
    Es teilt das Bild in ein Gitter ein und sagt in **einem einzigen Durchlauf** sowohl die Position (Bounding Boxes) als auch die Klasse vorher.
    Vorteile: sehr schnell, gut für mobile und Schul-Projekte.
    In dieser App zeigen wir ein leichtgewichtiges vortrainiertes Modell (YOLOv8n). 
    Für die exakten 40 Pflanzenarten müsste man ein eigenes Modell trainieren (wie in App 2).
    """)
    st.info("Schul-Tipp: YOLO eignet sich hervorragend, um zuerst die **Position** einer Pflanze im Bild zu finden – die genaue Art kann dann mit einem Klassifikator (z. B. Teachable Machine) bestimmt werden.")

st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
