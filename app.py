import streamlit as st

st.title("🧠 AI Safety Monitor")
st.write("Testing deployment...")

try:
    import cv2
    st.success("✅ OpenCV imported successfully!")
    st.write(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    st.error(f"❌ OpenCV import failed: {e}")

try:
    from ultralytics import YOLO
    st.success("✅ Ultralytics imported successfully!")
except Exception as e:
    st.error(f"❌ Ultralytics import failed: {e}")

st.info("If you see green checks, deployment is working!")
