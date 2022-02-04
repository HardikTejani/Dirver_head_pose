import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3
import pygame 
from pygame import mixer
from streamlit_webrtc import webrtc_streamer

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# pygame.init()
# pygame.mixer.init()
# voice_left = mixer.Sound('left.wav')
# voice_right = mixer.Sound('Right.wav')
# voice_down = mixer.Sound('down.wav')
# eyes_blink= mixer.Sound('eyes_blink.wav')
# yawn = mixer.Sound('Yawning.wav')


st.title("Webcam Application")

webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        #video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# cam = cv2.VideoCapture(0)

# while run:
#     ret, frame = cam.read()
#     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')
