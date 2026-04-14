import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

# ================= UI =================
st.set_page_config(page_title="AI Safety Monitor", layout="wide")
st.title("🧠 AI Safety Monitor")
st.write("Dangerous situation detection system")

# ================= TRY IMPORT CV2 =================
try:
    import cv2
    CV2_AVAILABLE = True
    st.success("✅ OpenCV loaded")
except ImportError as e:
    CV2_AVAILABLE = False
    st.error(f"❌ OpenCV not available: {e}")

# ================= SIMPLE DETECTION =================
def analyze_image(image):
    """Simple image analysis"""
    img_array = np.array(image)
    height, width = img_array.shape[:2] if len(img_array.shape) > 2 else (img_array.shape[0], img_array.shape[1])
    
    # Basic info
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array)
    else:
        brightness = np.mean(img_array)
    
    return brightness, width, height

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Settings")
    show_details = st.checkbox("Show details", True)
    
    st.header("ℹ️ System Status")
    st.write(f"OpenCV: {'✅ Available' if CV2_AVAILABLE else '❌ Not available'}")
    st.write(f"Python: {os.sys.version[:50]}...")

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📹 Camera"])

# ================= IMAGE =================
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Analyzing image..."):
            brightness, width, height = analyze_image(image)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, use_column_width=True)
        
        with col2:
            st.success("✅ Analysis complete!")
            
            if show_details:
                st.write(f"📊 Image size: {width} x {height}")
                st.write(f"💡 Brightness: {brightness:.0f}/255")
            
            if CV2_AVAILABLE:
                st.info("🔄 Advanced detection available")
            else:
                st.warning("⚠️ Full detection requires OpenCV")

# ================= VIDEO =================
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        st.video(uploaded_video)
        if CV2_AVAILABLE:
            st.info("📌 Video frame analysis coming soon")
        else:
            st.warning("⚠️ Video analysis requires OpenCV")

# ================= CAMERA =================
with tab3:
    st.info("📸 Use the Image tab to upload photos from your camera")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
**📌 Tech Stack:**
- **Frontend:** Streamlit
- **Processing:** Python, PIL, NumPy
- **Status:** ✅ Ready

**🔧 Full features coming soon!**
""")
