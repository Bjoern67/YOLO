import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(page_title="YOLO Objekterkennung", layout="wide")
st.title("🖼️ YOLO Objekterkennung")

# Modell sofort laden (am besten nano-Variante → wenig RAM)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")   # oder "yolo26n.pt" – nano ist am sichersten

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    
    with st.spinner("Analysiere..."):
        results = model.predict(image, conf=0.5, verbose=False)
    
    # Annotiertes Bild
    annotated = Image.fromarray(results[0].plot())
    with col2:
        st.image(annotated, caption="Erkannt", use_column_width=True)
    
    # Liste der Objekte
    st.subheader("Erkannte Objekte:")
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"- {cls} ({conf:.1%})")
    else:
        st.info("Nichts mit ausreichender Sicherheit erkannt.")
