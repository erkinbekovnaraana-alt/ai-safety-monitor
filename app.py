import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="AI Safety Monitor", layout="wide")
st.title("🧠 AI Safety Monitor")
st.write("Dangerous situation detection (weapons and hazardous objects)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

dangerous_objects = {
    "knife": "🔪 Knife",
    "gun": "🔫 Gun", 
    "scissors": "✂️ Scissors",
    "hammer": "🔨 Hammer",
}

def analyze_frame(frame, conf_threshold=0.25):
    results = model(frame, conf=conf_threshold, verbose=False)
    
    persons = []
    dangerous_items = []
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        if label == "person":
            persons.append((x1, y1, x2, y2, confidence))
        elif label.lower() in dangerous_objects:
            dangerous_items.append((x1, y1, x2, y2, label, confidence))
    
    alerts = []
    for obj in dangerous_items:
        ox1, oy1, ox2, oy2, obj_label, obj_conf = obj
        for person in persons:
            px1, py1, px2, py2, person_conf = person
            alerts.append({
                "object": obj_label,
                "object_name": dangerous_objects.get(obj_label.lower(), obj_label),
                "confidence": obj_conf,
                "bbox": (ox1, oy1, ox2, oy2),
            })
    
    return persons, dangerous_items, alerts

def draw_analysis(frame, persons, dangerous_items, alerts):
    for person in persons:
        x1, y1, x2, y2, conf = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    for item in dangerous_items:
        x1, y1, x2, y2, label, conf = item
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"⚠️ {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
    
    for alert in alerts:
        x1, y1, x2, y2 = alert["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "🔴 DANGER!", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    return frame

with st.sidebar:
    conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.25)

tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📹 Camera"])

with tab1:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        persons, dangerous, alerts = analyze_frame(frame, conf_threshold)
        frame = draw_analysis(frame, persons, dangerous, alerts)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if alerts:
            st.error(f"⚠️ DANGERS: {len(alerts)}")
        else:
            st.success("✅ Safe")

with tab2:
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded:
        st.video(uploaded)

with tab3:
    st.info("Camera: Use image upload from phone camera")

st.markdown("---")
st.markdown("**Note:** Standard YOLO doesn't detect weapons. For weapon detection, train custom model.")
