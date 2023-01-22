from pathlib import Path
import cv2, json
import depthai as dai
import numpy as np
from time import monotonic

# Get argument first
parentDir = Path(__file__).parent
MODEL_PATH = str((parentDir / Path('../Raspberry/Models/MyModelYOLOv7tiny/best_openvino_2021.4_6shave.blob')).resolve().absolute())
videoPath = str((parentDir / Path('exit.mp4')).resolve().absolute())
imgPath = str((parentDir / Path('./cuadros/PXL_20221011_183430412 (2).jpg')).resolve().absolute())
CONFIG_PATH = str((parentDir / Path('../Raspberry/Models/MyModelYOLOv7tiny/best.json')).resolve().absolute())

# Extraer metadata del archivo de configuración .json
with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)
metadata = config.get("nn_config").get("NN_specific_metadata")
classes = metadata.get("classes")
coordinates = metadata.get("coordinates")
anchors = metadata.get("anchors")
anchorMasks = metadata.get("anchor_masks")
iouThreshold = metadata.get("iou_threshold")
confidenceThreshold = metadata.get("confidence_threshold")

# Extraer labels del archivo de configuración .json
labelMap = config.get("mappings").get("labels")
# Anhcho y alto de la imagen de entrada a la red neuronal
width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

xinFrame = pipeline.create(dai.node.XLinkIn)
nnOut = pipeline.create(dai.node.XLinkOut)

xinFrame.setStreamName("inFrame")
nnOut.setStreamName("detectedObjects")

# Properties
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(MODEL_PATH)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Yolo specific parameters
nn.setNumClasses(classes)
nn.setCoordinateSize(coordinates)
nn.setAnchors(anchors)
nn.setAnchorMasks(anchorMasks)
nn.setIouThreshold(0.5)
nn.setConfidenceThreshold(0.6)

# Linking
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="detectedObjects", maxSize=4, blocking=False)

    cap = cv2.VideoCapture(videoPath)
    # Mostrar
    print(f"Video size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print("Network input size: {}x{}".format(width, height))


    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        
        # Convertir a frame rectarnagular a uno cuadrado recortando los laterales
        if frame.shape[0] != frame.shape[1]:
            if frame.shape[0] > frame.shape[1]:
                frame = frame[(frame.shape[0] - frame.shape[1])//2:(frame.shape[0] - frame.shape[1])//2 + frame.shape[1], :, :]
            else:
                frame = frame[:, (frame.shape[1] - frame.shape[0])//2:(frame.shape[1] - frame.shape[0])//2 + frame.shape[0], :]
        
        # Resize frame 
        frame = cv2.resize(frame, (640, 640))

        img = dai.ImgFrame()
        img.setData(frame)
        img.setTimestamp(monotonic())
        img.setWidth(640)
        img.setHeight(640)
        qIn.send(img)

        inDet = qDet.get().detections
        print(inDet)

        cv2.imshow("frames", frame)

        # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q} 
        if cv2.waitKey(1) in [27, 32, ord('q')]:
            break