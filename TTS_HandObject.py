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


hands = mp_hands.Hands( # create hands object
		max_num_hands=1,
		model_complexity=0,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5)


face_detection = mp_face_detection.FaceDetection( # create face detection object
		model_selection=0,
		min_detection_confidence=0.5)


# create funtion to speak text
def speak(text):
    
    engine = pyttsx3.init()             # create engine
    engine.setProperty('rate', 150)     # set rate and volume
    engine.setProperty('volume', 1)     # set volume``
    engine.say(text)                    # say text
    engine.runAndWait()                 # run engine
    engine.stop()                       # stop engine
    return

# function to return the position of the fingertips in a list
def fingertips_positions(results_hands):
    for hand_landmarks in results_hands.multi_hand_landmarks:   
        x0 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
        y0 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
        x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
        y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
        x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
        y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
        x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
        y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
        x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
        y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
    return [(x0,y0),(x1,y1),(x2,y2),(x3,y3),(x4,y4)]


# function to return the position of the eyes and nose tip in a list
def eyesnose_positions(results_face_detection):
    for detection in results_face.detections:
        # Right eye position
        x_RE = int(detection.location_data.relative_keypoints[0].x * width)
        y_RE = int(detection.location_data.relative_keypoints[0].y * height)
        # Left eye position
        x_LE = int(detection.location_data.relative_keypoints[1].x * width)
        y_LE = int(detection.location_data.relative_keypoints[1].y * height)
        # Nose tip position
        x_NT = int(detection.location_data.relative_keypoints[2].x * width)
        y_NT = int(detection.location_data.relative_keypoints[2].y * height)
    return [(x_RE,y_RE),(x_LE,y_LE),(x_NT,y_NT)]

# function to draw dots and labels at the position of the detected objects in the video frame
def draw_detected_objects(frame, objects_labels, objects_positions):
    for i, position in enumerate(objects_positions):
        cv2.circle(frame, position, 5, (0, 0, 255), 2)
        cv2.putText(frame, objects_labels[i], position, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return frame


# capture video
cap1 = cv2.VideoCapture(0)
success, frame = cap1.read()
height, width, _ = frame.shape

# define distance minimum between hand and face
delta = .05
delta_x = int(width*delta)
delta_y = int(height*delta)

# boolean flag to check if hand is detected
PrevFingerDetect  = False

# create loop to capture video
while cap1.isOpened():
    # capture frame from video from cap1
    success, frame = cap1.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame.flags.writeable = False                   # make frame read-only
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB
  
    results_hands = hands.process(frame)            # process frame with hands object
    results_face = face_detection.process(frame)    # process frame with face detection object

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # check if hand is detected
    if results_hands.multi_hand_landmarks is not None:

        # check if hand is in the frame previously
        if not PrevFingerDetect:
            speak("Detected Hand") 
            PrevFingerDetect = True
        
        fingertips = fingertips_positions(results_hands)                    # position points of fingertips detected
        fingertips_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]   # labels of fingertips detected
        index_position = fingertips[1]                                      # position of index finger
        draw_detected_objects(frame, fingertips_labels, fingertips)         # draw dots and labels at the fingertip position on the frame            

        if results_face.detections:
            object_positions = eyesnose_positions(results_face.detections)  # position points of the eyes and nose tip detected
            eyesnose_labes = ["Left-Eye", "Rigth-Eye", "Nose"]

            sumCateros = [abs(x-index_position[0]) + abs(y-index_position[1]) for x,y in object_positions]
            nearObjectIndex = sumCateros.index(min(sumCateros))
            cv2.line(frame, index_position, object_positions[nearObjectIndex], (255,0,255), 4)
            cv2.putText(frame, eyesnose_labes[nearObjectIndex], pos[nearObjectIndex], cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
    else:
        if PrevFingerDetect:
            speak("Lost Hand") 
            PrevFingerDetect=False

    # Flip the frame horizontally for a selfie-view display.
    cv2.imshow('manitos y cara', frame) #cv2.flip(frame, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap1.release()
