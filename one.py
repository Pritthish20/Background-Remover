import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation as selfSeg
import os
import streamlit as st

st.title("Real-time Background Removal with Streamlit")

ImgList = os.listdir("images")
img_options = [f"images/{i}" for i in ImgList]
selected_bg = st.sidebar.selectbox("Choose a background image", img_options)

segmentor = selfSeg()
cut_threshold = st.sidebar.slider("Cut Threshold", 0.0, 1.0, 0.88)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

stframe = st.empty()

bg_image = cv2.imread(selected_bg)
bg_image = cv2.resize(bg_image, (640, 480))

stop_button = st.sidebar.button("Stop", key="unique_stop_button")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from camera.")
        break
   
    img_out = segmentor.removeBG(img, bg_image, cutThreshold=cut_threshold)
  
    stack_img = cvzone.stackImages([img, img_out], 2, 1)

    stframe.image(stack_img, channels="BGR")

    if stop_button:
        break

cap.release()
st.write("Stopped.")
