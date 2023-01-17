"""
yolo model: size 640x640
paml detection model: size 128×128
hand_landmarks model: size 256x256
""" 


import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import time
import sys, os, json


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
MY_YOLO_MODEL = str(SCRIPT_DIR / "models/best3_openvino_2021.4_6shave.blob")
MY_YOLO_CONFIG = str(SCRIPT_DIR / "models/best3.json")

# Extraer metadata del archivo de configuración .json
with open(MY_YOLO_CONFIG, 'r') as file:
    config = json.load(file)
metadata = config.get("nn_config").get("NN_specific_metadata")
classes = metadata.get("classes")
coordinates = metadata.get("coordinates")
anchors = metadata.get("anchors")
anchorMasks = metadata.get("anchor_masks")
iouThreshold = metadata.get("iou_threshold")
confidenceThreshold = metadata.get("confidence_threshold")
labels = config.get("mappings").get("labels")
width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

# Create pipeline
pipeline = dai.Pipeline()

# ColorCamera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(width, height)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(40)
cam_out = pipeline.create(dai.node.XLinkOut)
cam_out.setStreamName("cam_out")

# StereoDepth
left = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right = pipeline.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(left.getResolutionWidth(), right.getResolutionHeight())
stereo_out = pipeline.create(dai.node.XLinkOut)
stereo_out.setStreamName("stereo_out")

# ImageManip
PALM_DETECTION_MODEL_SIZE = 128
manip  = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(PALM_DETECTION_MODEL_SIZE, PALM_DETECTION_MODEL_SIZE)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip_out = pipeline.create(dai.node.XLinkOut)
manip_out.setStreamName("manip_out")

# YoloSpatialDetectionNetwork
yoloSpatial= pipeline.create(dai.node.YoloSpatialDetectionNetwork)
yoloSpatial.setBlobPath(MY_YOLO_MODEL)
yoloSpatial.setConfidenceThreshold(confidenceThreshold)
yoloSpatial.input.setBlocking(False)
yoloSpatial.setBoundingBoxScaleFactor(0.5)
yoloSpatial.setDepthLowerThreshold(350)
yoloSpatial.setDepthUpperThreshold(5000)

yoloSpatial.setNumClasses(classes)
yoloSpatial.setCoordinateSize(coordinates)
yoloSpatial.setAnchors(anchors)
yoloSpatial.setAnchorMasks(anchorMasks)
yoloSpatial.setIouThreshold(iouThreshold)
yoloSpatial_out = pipeline.create(dai.node.XLinkOut)
yoloSpatial_out.setStreamName("yoloSpatial_out")
yolo_out = pipeline.create(dai.node.XLinkOut)
yolo_out.setStreamName("yolo_out")

# Define palm detection model
pd_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
pd_nn.setBlobPath(PALM_DETECTION_MODEL)
pd_nn.input.setQueueSize(1)
pd_nn.input.setBlocking(False)
pd_nn.setConfidenceThreshold(0.5)
pd_nn_out = pipeline.create(dai.node.XLinkOut)
pd_nn_out.setStreamName("pd_nn_out")

# Linking nodes for the output of manip_out
cam.preview.link(manip.inputImage)      # cam.preview -> manip.inputImage
manip.out.link(manip_out.input)         # manip.out -> manip_out.input

# Linking nodes for the output of pd_nn_out
manip.out.link(pd_nn.input)             # manip.out -> pd_nn.input
pd_nn.passthrough.link(manip_out.input) # pd_nn.passthrough -> manip_out.input
pd_nn.out.link(pd_nn_out.input)         # pd_nn.out -> pd_nn_out.input

# Linking nodes for the output of stereo_out
left.out.link(stereo.left)              # left.out -> stereo.left
right.out.link(stereo.right)            # right.out -> stereo.right
stereo.depth.link(stereo_out.input)     # stereo.depth -> stereo_out.input

# Linking nodes for the YoloSpatialDetectionNetwork
cam.preview.link(yoloSpatial.input)
yoloSpatial.passthrough.link(cam_out.input)
yoloSpatial.out.link(yolo_out.input)
stereo.depth.link(yoloSpatial.inputDepth)
yoloSpatial.passthroughDepth.link(stereo_out.input)
yoloSpatial.outNetwork.link(yoloSpatial_out.input)

# Connect to device and start pipeline
device = dai.Device(pipeline)

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_cam_out = device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
q_yolo_out = device.getOutputQueue(name="yolo_out", maxSize=4, blocking=False)
q_yoloSpatial_out= device.getOutputQueue(name="yoloSpatial_out", maxSize=4, blocking=False)
q_manip_out = device.getOutputQueue(name="manip_out", maxSize=4, blocking=False)
q_pd_nn_out = device.getOutputQueue(name="pd_nn_out", maxSize=4, blocking=False)
q_stereo_out = device.getOutputQueue(name="stereo_out", maxSize=4, blocking=False)

frames_count = 0
start_time = time.time()
while True:
    camera_frame = q_cam_out.get().getCvFrame()
    manip_frame = q_manip_out.get().getCvFrame()
    yolo_detections = q_yolo_out.get().detections
    pd_detections = q_pd_nn_out.get().detections

    if len(yolo_detections) > 0:
        for detection in yolo_detections:
            x1 = int(detection.xmin * width)
            y1 = int(detection.ymin * height)
            x2 = int(detection.xmax * width)
            y2 = int(detection.ymax * height)
            label = labels[detection.label]
            cv2.rectangle(camera_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(camera_frame, label, (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
    
    if len(pd_detections) > 0:
        for detection in pd_detections:
            x1 = int(detection.xmin * PALM_DETECTION_MODEL_SIZE)
            y1 = int(detection.ymin * PALM_DETECTION_MODEL_SIZE)
            x2 = int(detection.xmax * PALM_DETECTION_MODEL_SIZE)
            y2 = int(detection.ymax * PALM_DETECTION_MODEL_SIZE)
            cv2.rectangle(manip_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show FPS
    frame_time = time.time() - start_time
    if frame_time > 1:
        fps = frames_count / frame_time
        cv2.putText(manip_frame, f"FPS: {fps:.2f}", (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
        frames_count = 0
        start_time = time.time()


    #cv2.imshow("rgb", camera_frame)
    cv2.imshow("manip", manip_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

# cerrar la conexión con el dispositivo
device.close()