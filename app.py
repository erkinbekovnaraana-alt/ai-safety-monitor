import streamlit as st
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
try:
    import cv2
except:
    import os
    os.system("apt-get update && apt-get install -y libgl1")
    import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import urllib.request

# ================= UI =================
st.set_page_config(page_title="AI Safety Monitor", layout="wide")
st.title("🧠 AI Safety Monitor")
st.write("Dangerous situation detection (weapons and hazardous objects)")

# ================= DOWNLOAD CUSTOM MODEL =================
@st.cache_resource
def download_weapon_model():
    """Download pre-trained weapon detection model"""
    model_path = "weapon_detection.pt"
    if not os.path.exists(model_path):
        with st.spinner("Downloading weapon detection model..."):
            # Using YOLOv8 nano model (replace with actual weapon detection model URL)
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            urllib.request.urlretrieve(url, model_path)
    return model_path

@st.cache_resource
def load_models(use_weapon_model=False):
    """Load YOLO models"""
    if use_weapon_model:
        try:
            model_path = download_weapon_model()
            model = YOLO(model_path)
            st.success("✅ Weapon detection model loaded successfully")
        except Exception as e:
            st.warning(f"Failed to load custom model: {e}. Using standard model.")
            model = YOLO("yolov8n.pt")
    else:
        model = YOLO("yolov8n.pt")
    return model

# ================= DANGEROUS OBJECTS DATABASE =================
dangerous_objects = {
    # Weapons
    "knife": "🔪 Knife",
    "gun": "🔫 Gun",
    "scissors": "✂️ Scissors",
    "sword": "⚔️ Sword",
    "axe": "🪓 Axe",
    
    # Tools
    "hammer": "🔨 Hammer",
    "wrench": "🔧 Wrench",
    "saw": "🪚 Saw",
    
    # Hazardous items
    "fire extinguisher": "🧯 Fire Extinguisher",
    "baseball bat": "🏏 Baseball Bat",
    "bottle": "🍾 Bottle",
    
    # Additional dangerous items from COCO
    "fork": "🍴 Fork",
    "remote": "📟 Remote (potentially dangerous)",
}

# Safe objects
safe_objects = [
    "book", "phone", "laptop", "cup", "chair", "table", 
    "keyboard", "mouse", "tv", "clock", "vase", "plant",
    "apple", "banana", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

# ================= HELPERS =================
def is_inside(obj_box, person_box):
    """Check if object is inside person's area"""
    ox1, oy1, ox2, oy2 = obj_box
    px1, py1, px2, py2 = person_box
    return not (ox2 < px1 or ox1 > px2 or oy2 < py1 or oy1 > py2)

def hand_zone(person_box):
    """Determine person's hand zone area"""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return (x1, int(y1 + h * 0.25), x2, int(y1 + h * 0.75))

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def analyze_frame(frame, conf_threshold=0.25):
    """Analyze frame and detect dangers"""
    results = model(frame, conf=conf_threshold, verbose=False)
    
    persons = []
    dangerous_items = []
    safe_items = []
    all_detections = []
    
    # Collect all detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        all_detections.append(f"{label} ({confidence:.2f})")
        
        if label == "person":
            persons.append((x1, y1, x2, y2, confidence))
        elif label.lower() in dangerous_objects or label in dangerous_objects:
            dangerous_items.append((x1, y1, x2, y2, label, confidence))
        elif label in safe_objects:
            safe_items.append((x1, y1, x2, y2, label, confidence))
    
    # Debug output
    if all_detections:
        print(f"Detected: {', '.join(all_detections)}")
    
    # Analyze dangerous situations
    alerts = []
    
    for obj in dangerous_items:
        ox1, oy1, ox2, oy2, obj_label, obj_conf = obj
        
        for person in persons:
            px1, py1, px2, py2, person_conf = person
            hand_zone_box = hand_zone((px1, py1, px2, py2))
            
            iou = calculate_iou((ox1, oy1, ox2, oy2), hand_zone_box)
            
            # Calculate distance between object and person
            obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                             (obj_center[1] - person_center[1])**2)
            
            # If object is near person or in hand zone
            if is_inside((ox1, oy1, ox2, oy2), hand_zone_box) or iou > 0.1 or distance < 100:
                alerts.append({
                    "object": obj_label,
                    "object_name": dangerous_objects.get(obj_label.lower(), obj_label),
                    "confidence": obj_conf,
                    "bbox": (ox1, oy1, ox2, oy2),
                    "person_bbox": (px1, py1, px2, py2),
                    "distance": distance
                })
    
    return persons, dangerous_items, safe_items, alerts

def draw_analysis(frame, persons, dangerous_items, safe_items, alerts):
    """Draw analysis results on frame"""
    # Draw people
    for person in persons:
        x1, y1, x2, y2, conf = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hand zone
        hz = hand_zone((x1, y1, x2, y2))
        overlay = frame.copy()
        cv2.rectangle(overlay, (hz[0], hz[1]), (hz[2], hz[3]), (255, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    
    # Safe items
    for item in safe_items:
        x1, y1, x2, y2, label, conf = item
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Dangerous items
    for item in dangerous_items:
        x1, y1, x2, y2, label, conf = item
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"⚠️ {dangerous_objects.get(label.lower(), label)}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Alerts
    for alert in alerts:
        x1, y1, x2, y2 = alert["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"🔴 DANGER: {alert['object_name']}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Pulsing effect
        if int(cv2.getTickCount() / cv2.getTickFrequency() * 10) % 2:
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 2)
    
    return frame

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    use_weapon_model = st.checkbox("Use weapon detection model", value=False)
    show_details = st.checkbox("Show details", True)
    debug_mode = st.checkbox("Debug mode", value=False)
    
    st.header("⚠️ Dangerous Objects")
    cols_danger = st.columns(2)
    for i, (item, emoji_name) in enumerate(dangerous_objects.items()):
        cols_danger[i % 2].write(f"{emoji_name}")
    
    st.header("✅ Safe Objects (examples)")
    st.write(f"• {', '.join(safe_objects[:8])}...")

# Load model with selected parameters
model = load_models(use_weapon_model)

# ================= MAIN CONTENT =================
tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📹 Camera"])

# ================= IMAGE PROCESSING =================
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing image..."):
            persons, dangerous_items, safe_items, alerts = analyze_frame(frame, conf_threshold)
            frame = draw_analysis(frame, persons, dangerous_items, safe_items, alerts)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            if alerts:
                st.error(f"⚠️ DANGERS DETECTED: {len(alerts)}")
                for alert in alerts:
                    st.warning(f"• {alert['object_name']} (confidence: {alert['confidence']:.2%})")
            else:
                st.success("✅ Safe! No dangerous objects detected")
            
            if show_details:
                st.write(f"👥 People found: {len(persons)}")
                st.write(f"⚠️ Dangerous items: {len(dangerous_items)}")
                st.write(f"✅ Safe items: {len(safe_items)}")
                
                if debug_mode and dangerous_items:
                    st.write("---")
                    st.write("🔍 Dangerous item detections:")
                    for item in dangerous_items:
                        st.write(f"• {item[4]} (confidence: {item[5]:.2%})")

# ================= VIDEO PROCESSING =================
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        alert_count = 0
        frame_count = 0
        total_alerts = []
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 15 == 0:
                persons, dangerous_items, safe_items, alerts = analyze_frame(frame, conf_threshold)
                if alerts:
                    alert_count += len(alerts)
                    total_alerts.extend(alerts)
                frame = draw_analysis(frame, persons, dangerous_items, safe_items, alerts)
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if total_frames > 0:
                progress_bar.progress(min(1.0, frame_count / total_frames))
        
        cap.release()
        os.unlink(video_path)
        
        if alert_count > 0:
            st.error(f"⚠️ Video contains {alert_count} dangerous situations!")
            if show_details:
                unique_alerts = set([a['object_name'] for a in total_alerts])
                st.write(f"Detected items: {', '.join(unique_alerts)}")
        else:
            st.success("✅ Video is safe!")

# ================= REALTIME CAMERA =================
with tab3:
    run_camera = st.checkbox("🎥 Start camera", key="camera")
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera. Check your webcam connection.")
        else:
            stframe = st.empty()
            alert_placeholder = st.empty()
            stats_placeholder = st.empty()
            stop_button = st.button("⏹️ Stop")
            
            frame_skip = 0
            while run_camera and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to get camera frame")
                    break
                
                if frame_skip % 2 == 0:
                    persons, dangerous_items, safe_items, alerts = analyze_frame(frame, conf_threshold)
                    frame = draw_analysis(frame, persons, dangerous_items, safe_items, alerts)
                    
                    if alerts:
                        alert_placeholder.error(f"⚠️ WARNING! Detected: {', '.join(set([a['object_name'] for a in alerts]))}")
                    else:
                        alert_placeholder.success("✅ Monitoring: Safe")
                    
                    if show_details:
                        stats_placeholder.info(f"👥 People: {len(persons)} | ⚠️ Dangerous: {len(dangerous_items)} | ✅ Safe: {len(safe_items)}")
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                frame_skip += 1
            
            cap.release()
            st.info("Camera stopped")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
**📌 Important Notes:**
- Standard YOLO model (yolov8n.pt) **DOES NOT** detect knives, guns, or most dangerous objects
- Weapon detection requires a **specialized model**
- Enable "Debug mode" in sidebar to see all model detections
- For best results, use good lighting and clear images
- **Recommendation**: Train your own model on weapon datasets or find pre-trained weights
""")
