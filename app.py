import streamlit as st
import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string
import keras
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# Page Configuration
st.set_page_config(page_title="ISL Detection", page_icon="🤟", layout="wide")

st.title("🤟 Indian Sign Language (ISL) Detection")
st.markdown("Use your live camera feed, take a picture, or upload an image to detect the ISL sign.")

# Load models safely into cache
@st.cache_resource
def load_model():
    model = keras.models.load_model("model.h5")
    return model

@st.cache_resource
def load_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands

model = load_model()
hands = load_mediapipe_hands()

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet =  ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def process_frame(image, detect_static=False):
    debug_image = copy.deepcopy(image)
    detected_labels = []

    # If static image, we might want to use a separate Hands instance, but reusing the global one works 
    # if we don't care about the minor tracking state mismatch between images and video.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            # Predict
            predictions = model.predict(df, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            label = alphabet[predicted_classes[0]]
            detected_labels.append(label)
            
            # Add text to image
            cv2.putText(debug_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
    return debug_image, detected_labels

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    # Flip the image horizontally for selfie view
    image = cv2.flip(image, 1)
    
    processed_img, _ = process_frame(image)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

tab1, tab2, tab3 = st.tabs(["🎥 Live Feed (WebRTC)", "📷 Camera Snapshot", "📤 Upload Image"])

with tab1:
    st.markdown("### Real-Time Indian Sign Language Detection")
    st.markdown("Ensure your webcam is enabled and perform hand signs in front of the camera.")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="isl-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    st.markdown("### Take a picture with your camera")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            # Need to ensure RGB to BGR for process_frame which expects BGR like cv2
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed_img, labels = process_frame(image_np, detect_static=True)
        # Convert back to RGB for display in Streamlit
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed_img, caption="Processed Image with Landmarks", use_container_width=True)
            
        if labels:
            st.success(f"### Detected Sign: {', '.join(labels)}")
        else:
            st.warning("No hands detected in the image. Please try again.")

with tab3:
    st.markdown("### Upload an existing image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        processed_img, labels = process_frame(image_np, detect_static=True)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed_img, caption="Processed Image with Landmarks", use_container_width=True)
            
        if labels:
            st.success(f"### Detected Sign: {', '.join(labels)}")
        else:
            st.warning("No hands detected in the image. Please try again.")
