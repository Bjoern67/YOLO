import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Seiten-Setup
st.set_page_config(page_title="YOLO26 Objekterkennung", layout="wide")
st.title("🖼️ YOLO26 Objekt-Detektion App")
st.markdown("**Bild hochladen → Objekte (inkl. Kleidung/Accessoires) werden automatisch mit Boxen hervorgehoben!**")

# Modell einmalig laden (Caching für Speed)
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")  # n = nano (schnell), alternativ yolo26s.pt, yolo26m.pt etc.

model = load_model()

# File-Uploader
uploaded_file = st.file_uploader(
    "Bild hochladen (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Originalbild
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Originalbild", use_column_width=True)
    
    # Erkennung
    with st.spinner("🔍 YOLO26 analysiert das Bild..."):
        results = model.predict(
            original_image,
            conf=0.5,          # Konfidenz-Schwelle (kannst du später als Slider hinzufügen)
            verbose=False
        )
    
    # Annotiertes Bild (Bounding-Boxes + Labels)
    annotated_array = results[0].plot()  # numpy-Array mit Hervorhebungen
    annotated_pil = Image.fromarray(annotated_array)
    
    with col2:
        st.image(annotated_pil, caption="Erkannte Objekte hervorgehoben (YOLO26)", use_column_width=True)
    
    # Liste der erkannten Objekte
    st.subheader("✅ Erkannte Objekte:")
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detected.append(f"**{class_name}** (Konfidenz: {confidence:.1%})")
    
    if detected:
        for item in detected:
            st.markdown(f"- {item}")
    else:
        st.info("Keine Objekte mit ausreichender Konfidenz erkannt.")
    
    # Download-Button für das annotierte Bild
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 Annotiertes Bild herunterladen",
        data=buf.getvalue(),
        file_name="yolo26_detected.png",
        mime="image/png"
    )
