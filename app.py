import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Safety Monitor", layout="wide")
st.title("🧠 AI Safety Monitor")
st.write("Dangerous situation detection system")

# Test imports
try:
    import cv2
    st.success("✅ OpenCV loaded successfully!")
    st.write(f"OpenCV version: {cv2.__version__}")
    
    # Test OpenCV works
    test_array = np.zeros((100, 100, 3), dtype=np.uint8)
    result = cv2.cvtColor(test_array, cv2.COLOR_RGB2BGR)
    st.success("✅ OpenCV functions work!")
    
except Exception as e:
    st.error(f"❌ Error: {e}")

st.info("Deployment successful! Now adding full features...")
