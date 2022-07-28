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


# define distance minimum between hand and face
delta = .05
delta_x = int(width*delta)
delta_y = int(height*delta)

# boolean flag to check if hand is detected
PrevFingerDetect  = False

# create loop to capture video
while cap1.isOpened():
    # capture frame from video from cap1
    success, image = cap1.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    results_hands = hands.process(image)
    results_face = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results_hands.multi_hand_landmarks is not None:
        
        if not PrevFingerDetect:
            speak("Detected Hand") 
            PrevFingerDetect = True

        for hand_landmarks in results_hands.multi_hand_landmarks:   
            x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
            y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
            x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
            x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
            y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
            x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
            y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
            x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
            y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

        cv2.circle(image, (x5, y5), 3,(255,0,0), 3)
        cv2.circle(image, (x1, y1), 3,(255,0,0), 3)
        cv2.circle(image, (x2, y2), 3,(255,0,0), 3)
        cv2.circle(image, (x3, y3), 3,(255,0,0), 3)
        cv2.circle(image, (x4, y4), 3,(255,0,0), 3)

        cv2.putText(image, "5", (x5, y5), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(image, "1", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(image, "2", (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(image, "3", (x3, y3), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(image, "4", (x4, y4), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        

        if results_face.detections:
            for detection in results_face.detections:
                # Ojo derecho
                x_RE = int(detection.location_data.relative_keypoints[0].x * width)
                y_RE = int(detection.location_data.relative_keypoints[0].y * height)
                cv2.circle(image, (x_RE, y_RE), 3, (0, 0, 255), 5)
                # Ojo izquierdo
                x_LE = int(detection.location_data.relative_keypoints[1].x * width)
                y_LE = int(detection.location_data.relative_keypoints[1].y * height)
                cv2.circle(image, (x_LE, y_LE), 3, (255, 0, 255), 5)
                # Punta de la nariz
                x_NT = int(detection.location_data.relative_keypoints[2].x * width)
                y_NT = int(detection.location_data.relative_keypoints[2].y * height)
                cv2.circle(image, (x_NT, y_NT), 3, (255, 0, 0), 5)

            pos = [(x_RE,y_RE),(x_LE,y_LE),(x_NT,y_NT)]
            stringObjets = ["Left-Eye", "Rigth-Eye", "Nose"]
            sumCateros = [abs(x-x1) + abs(y-y1) for x,y in pos]
            nearObjectIndex = sumCateros.index(min(sumCateros))
            cv2.line(image, (x1,y1), pos[nearObjectIndex], (255,0,255), 4)
            cv2.putText(image, stringObjets[nearObjectIndex], pos[nearObjectIndex], cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
    else:
        if PrevFingerDetect:
            speak("Lost Hand") 
            PrevFingerDetect=False

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('manitos y cara', image) #cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap1.release()