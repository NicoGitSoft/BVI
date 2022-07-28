#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import pyttsx3 and mediapipe
import pyttsx3
import cv2
import mediapipe as mp

# create objet mediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# create funtion to speak text
def speak(text):
    # create engine
    engine = pyttsx3.init()
    # set rate and volume
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    # speak text
    engine.say(text)
    # run engine
    engine.runAndWait()
    # stop engine
    engine.stop()
    # close engine
    engine.close()
    # return to main
    return

hands = mp_hands.Hands( # create hands object
		max_num_hands=1,
		model_complexity=0,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5)


face_detection = mp_face_detection.FaceDetection( # create face detection object
		model_selection=0,
		min_detection_confidence=0.5)


# capture video
cap1 = cv2.VideoCapture(0)
success, image = cap1.read()
height, width, _ = image.shape


