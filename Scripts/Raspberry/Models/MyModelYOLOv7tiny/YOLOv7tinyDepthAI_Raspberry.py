import cv2, os, time, json, math, csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import depthai as dai

# Importar RPi.GPIO para controlar los pines de la raspberryPi
import RPI.GPIO as GPIO

upPIN = 18
downPIN = 12
leftPIN = 13
rightPIN = 19

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(upPIN, downPIN, leftPIN, rightPIN, GPIO.OUT) 

# Ruta del modelo la configuración de la red neuronal entrenada para la deteción de objetos
MODEL_PATH = "best_openvino_2021.4_6shave.blob"
CONFIG_PATH = "best.json"

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Extraer metadata del archivo de configuración .json
metadata = config.get("nn_config").get("NN_specific_metadata")
classes = metadata.get("classes")
coordinates = metadata.get("coordinates")
anchors = metadata.get("anchors")
anchorMasks = metadata.get("anchor_masks")
iouThreshold = metadata.get("iou_threshold")
confidenceThreshold = metadata.get("confidence_threshold")

# Extraer labels del archivo de configuración .json
labels = config.get("mappings").get("labels")

# Anhcho y alto de la imagen de entrada a la red neuronal
width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

label_messages = [
    'airplane symbol', 'baggage claim', 'bathrooms', 'danger-electricity', 'down arrow', 'emergency down arrow', 'emergency exit', 
    'emergency left arrow', 'emergency right arrow', 'emergency up arrow', 'extinguisher symbol', 'fire-extinguisher', 'handicapped symbol',
    'left arrow', 'no trespassing', 'restaurants', 'right arrow', 'thin left arrow', 'thin right arrow', 'thin, up arrow', 'up arrow']

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

# Define sources and outputs for the spatial detection network
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(width, height)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Depth specific settings
spatialDetectionNetwork.setBlobPath(MODEL_PATH)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(classes)
spatialDetectionNetwork.setCoordinateSize(coordinates)
spatialDetectionNetwork.setAnchors(anchors)
spatialDetectionNetwork.setAnchorMasks(anchorMasks)
spatialDetectionNetwork.setIouThreshold(0.5)
spatialDetectionNetwork.setConfidenceThreshold(0.6)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
spatialDetectionNetwork.passthrough.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
device =dai.Device(pipeline)

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
xoutBoundingBoxDepthMappingQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

# Calcular coordenadas de los vertices un bounding box 
def Vertices(detection):        
    x1 = int(detection.xmin * width)
    x2 = int(detection.xmax * width)
    y1 = int(detection.ymin * height)
    y2 = int(detection.ymax * height)
    return x1, x2, y1, y2

# Calcular la coordenada del centro de un bounding box
def Center(x1, x2, y1, y2):
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y

# Calcular la distancia en 3D de un objeto a la camara
def Distance3D(detection):
    X = detection.spatialCoordinates.x
    Y = detection.spatialCoordinates.y
    Z = detection.spatialCoordinates.z
    return math.sqrt(X**2 + Y**2 + Z**2)/1000

# Determina las coordenasdas del centro del bounding box más cercano y el índice correspondiente
def Nearest_Coordinate(OriginPoint, Centroids):
    x0, y0 = OriginPoint
    minDist = min((x-x0)**2 + (y-y0)**2 for x, y in Centroids)
    for index, (x, y) in enumerate(Centroids):
        if (x-x0)**2 + (y-y0)**2 == minDist:
            return x, y, index

#####################################################
############## INSTANCIAS DE VARIABLES ##############
#####################################################            

# Centro de la imagen
x0 = width//2 
y0 = height//2

# Región de interés (ROI) para la estimación de la distancia de obstáculos
DELTA = 5
THRESH_LOW = 200 # 20cm
THRESH_HIGH = 30000 # 30m
xmin, ymin, xmax, ymax = x0-DELTA, y0-DELTA, x0+DELTA, y0+DELTA

# Estilos de dibujo (colores y timpografía)
BoxesColor = (0, 255, 0)
BoxesSize = 2
LineColor = (0, 0, 255)
CircleColor = (255, 0, 0)
TextColor = (0, 255, 255)
FontFace = cv2.FONT_HERSHEY_SIMPLEX
FontSize = 0.5

# Variables de tiempo
frame_time = 0
move_time = 0

# bandera booleana para evitar que se repita el nombre del objeto detectado
mentioned_object = False 

# funciones anónimas para incremento de pulsos exponencial en los vibradores
f1 = lambda x: math.sqrt(1 + x) - 1
f2 = lambda x: (x + 1)**2 - 1

# Variables para graficar
t = [0] # Lista para almacenar los tiempos de ejecución del programa desde la primera captura de frame
fps = [0] # Lista para almacenar los fps de cada frame
dROI = [0] # Lista para almacenar las distancia a los obstáculos con visión estereoscópica
dOBJ = [0] # Lista para almacenar las distancia a las deteciones con visión monocular

# Instancia de variables para el loop principal
move_time = 0
frame_time = 0
start_time = time.time()
while True:
    # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q} 
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

    # Extraer datos del dispositivo OAK-D 
    frame = previewQueue.get().getCvFrame()         # Obtener el fotograma de la cámara RGB
    depthFrame = depthQueue.get().getFrame()        # Obtener el fotograma de profundidad 
    detections = detectionNNQueue.get().detections  # Obtener las detecciones de la red neuronal

    # Deteciones de la camara RGB
    if len(detections) != 0:
        Centroids = []  # Coordenadas del centro de los objetos detectados
        for detection in detections:
            detection_label = str(labels[detection.label])
            confidence = detection.confidence*100
            # Calcular los vertices de la caja delimitadora
            x1, x2, y1, y2 = Vertices(detection)
            # Calcular el centro de la caja delimitadora y agregarlo a la lista de centroides
            x, y = Center(x1, x2, y1, y2)
            Centroids.append((x, y))
            # Calcular la distancia a la caja delimitadora
            distance = Distance3D(detection)
            # Escribir información de la detección en el frame
            cv2.putText(frame, detection_label , (x1, y1), FontFace, FontSize, TextColor, 2)
            cv2.putText(frame, "{:.0f} %".format(confidence), (x2, y), FontFace, FontSize, TextColor, 1)
            cv2.putText(frame, "{:.2f} [m]".format(distance) , (x2, y2), FontFace, FontSize, TextColor)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BoxesColor, BoxesSize)
            
        # Determina las coordenasdas del centro del bounding box más cercano y el índice correspondiente
        x, y, i = Nearest_Coordinate((x0,y0), Centroids)
        # Reasignar coordenadas (x1, x2, y1, y2) a los vértices del bounding box más cercano
        x1, x2, y1, y2 = Vertices(detections[i])
        # Calcular la distancia al bounding box más cercano
        distance = Distance3D(detections[i])
        # Traducir la etiqueta del objeto detectado y reasingarla a la variable "detection_label"
        detection_label = str(label_messages[detections[i].label])


        # Si el centro de la imágen está dentro de la caja delimitadora del objeto más cercano
        if x1 < x0 < x2 and y1 < y0 < y2:
            if not mentioned_object:
                # Decir el nombre del objeto detectado y la distancia a la cámara al cual se encuentra
                os.system("spd-say '" + detection_label + "{:.2f} [m]".format(distance) + " metros'")
                arduino_serial.write(b'0')
                mentioned_object = True
        else: 
            # Calcular la distancia horizontal y vertical al objeto más cercano
            HorizontalDistance = abs(x - x0)
            VerticalDistance = abs(y - y0)
            if HorizontalDistance > VerticalDistance:
                if f1(time.time() - move_time) > f2(HorizontalDistance/(2*width)):
                    if (x - x0) > 0: # El objeto está a la derecha del centro de la imagen
                        arduino_serial.write(b'r') # 68 ASCII
                    else: # El objeto está a la izquierda del centro de la imagen
                        arduino_serial.write(b'l') # 76 ASCII
                    move_time = time.time()
            else:
                if f1(time.time() - move_time) > f2(VerticalDistance/(2*height)):
                    if (y - y0) > 0: # El objeto está abajo del centro de la imagen
                        arduino_serial.write(b'd') # 82 ASCII
                    else: # El objeto está arriba del centro de la imagen
                        arduino_serial.write(b'u') # 85 ASCII
                    move_time = time.time()
            mentioned_object = False

        # Dibujar una flecha que indique el objeto más cercano desde centro de la imágen
        cv2.arrowedLine(frame, (x0, y0), (x, y), LineColor, 2)
        dOBJ.append(distance)
    else:
        dOBJ.append(0)

    # Esimacion de la profundidad con las camaras estereoscopicas
    depthROI = depthFrame[ymin:ymax, xmin:xmax]
    inRange = (THRESH_LOW <= depthROI) & (depthROI <= THRESH_HIGH)
    dROI.append(np.mean(depthROI[inRange])/1000) # Almacenar la distancia a los obstáculos en la lista "d"

    # Mostrar  mapa de dispariedad
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
    cv2.putText(frame, "Z: " + ("{:.2f}m".format(dROI[-1]) ), (x0, y0), FontFace, FontSize, TextColor)
    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.imshow("Disparity Map", depthFrameColor)

    t.append(time.time() - start_time) # Almacenar el tiempo de ejecución
    fps.append( 1 / (time.time() - frame_time) ) # Almacenar fps
    frame_time = time.time() # Actualizar el tiempo de ejecución del frame

    # Mostrar: tiempo de ejecución, fps, distancia promedio a los obstáculos y anotaciones de las detecciones en el frame RGB
    cv2.circle(frame, (x0, y0), 5, CircleColor, -1) 
    cv2.putText(frame, "fps: {:.2f}".format(fps[-1]), (2, frame.shape[0] - 4), FontFace, 0.4, TextColor)
    cv2.putText(frame, "t: " + ("{:.2f} s".format(t[-1])), (10, 20), FontFace, FontSize, TextColor)
    cv2.imshow("Cámara RGB", frame)

# Cerrar todas las ventanas y apagar la cámara
cv2.destroyAllWindows()
device.close()

# Guardar los datos de las listas en un archivo .csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["t", "fps", "dOBJ", "dROI"])
    writer.writerows(zip(t, fps, dOBJ, dROI))

# Leer los datos del archivo .csv
data = pd.read_csv('data.csv')

# Graficar los datos
plt.figure(figsize=(10, 5))
plt.plot(data.t, data.dOBJ, label='Distancia la deteción YOLO más cercana')
plt.plot(data.t, data.dROI, label='Distancia promedio a los obstáculos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Distancia [m]')
plt.legend()
plt.savefig('destination_path.eps', format='eps')
plt.show()