# 🌳 Pflanzen-Detektor – Selbst trainiert mit Teachable Machine

**Schulprojekt 2026**  
KI-gestützte Erkennung von heimischen Bäumen und Blumen

## Projektbeschreibung
Diese Web-App wurde mit **Google Teachable Machine** trainiert und erkennt 12 heimische Pflanzenarten zuverlässig. Die App ist einfach bedienbar und liefert neben der Erkennung auch botanische Namen, Pflegetipps für das deutsche Klima sowie direkte Wikipedia-Links.

## Features
- Foto-Upload und Live-Kamera
- Echtzeit-Erkennung mit eigenem TensorFlow-Modell
- Anzeige von botanischem Namen, Pflegetipps und Wikipedia-Link
- Einheitliches grün-weißes Design (passend zur zweiten App)
- Vollständig auf Deutsch

## Trainierte Arten

**Bäume:**
- Birke
- Gemeine Fichte
- Gemeine Kiefer
- Rotbuche
- Stieleiche
- Traubeneiche

**Blumen:**
- Gänseblümchen
- Glockenblume
- Lavendel
- Rittersporn
- Sonnenblume
- Vergissmeinnicht

## Technologien
- Streamlit
- TensorFlow / Keras (exportiert aus Teachable Machine)
- Python

## Installation (lokal)

```bash
pip install -r requirements.txt
streamlit run app.py
