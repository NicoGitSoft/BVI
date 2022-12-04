import serial, subprocess, cv2, os, time, json, math, csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import depthai as dai
from scipy.io import savemat

# Cambiar la ruta de ejecución aquí
MainDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(MainDir)

try: # Intenta abrir el puerto serie
    #arduino_port = subprocess.getoutput('arduino-cli board list').strip().split("\n")[1].split()[0]
    arduino_serial = serial.Serial("COM8", 9600, timeout=1) #serial.Serial(arduino_port, 9600, timeout=1)
    arduino_is_connected = True

except:
    print("No se estableció comunicación serial con una placa Arduino correctamente")
    arduino_is_connected = False

# Ruta del modelo la configuración de la red neuronal entrenada para la deteción de objetos
MODEL_PATH = "yolov7tiny_openvino_2021.4_6shave.blob"
CONFIG_PATH = "yolov7tiny.json"

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

voice_messages = [
    'símbolo avión', 'recogida de equipajes', 'baños', 'peligro electricidad', 'flecha hacia abajo', 'flecha hacia abajo de emergencia', 'salida de emergencia', 
    "flecha izquierda de emergencia", "flecha derecha de emergencia", "flecha arriba de emergencia", "símbolo de extintor", "extintor", "preferencial",
    "flecha izquierda", "no pasar", "restaurantes", "flecha derecha", "flecha izquierda", "flecha derecha", "flecha arriba"]

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
THRESH_LOW = 0 # 34cm
THRESH_HIGH = 2000 # 2m
spatialDetectionNetwork.setBlobPath(MODEL_PATH)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(THRESH_LOW)
spatialDetectionNetwork.setDepthUpperThreshold(THRESH_HIGH)

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

#####################################################
######### FUNCIONES PARA EL PROCESAMIENTO ###########
#####################################################

# funciones anónimas para incremento de pulsos exponencial en los vibradores
f1 = lambda x: math.sqrt(1 + x) - 1
f2 = lambda x: (x + 1)**2 - 1

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

def Haptic_Interface(x,y):
    global move_time
    global mentioned_object
    global haptic_messages
    global DELTA
    HorizontalDistance = abs(x - x0)
    VerticalDistance = abs(y - y0)
    if HorizontalDistance + VerticalDistance < DELTA:
        if not mentioned_object:
            # Decir el nombre del objeto detectado y la distancia a la cámara al cual se encuentra
            #os.system("spd-say '" + detection_label + "{:.2f} [m]".format(distance) + " metros'")
            arduino_serial.write(b'0')
            haptic_messages.append('c')
            mentioned_object = True
    else: 
        # Calcular la distancia horizontal y vertical al objeto más cercano
        if HorizontalDistance > VerticalDistance:
            if True:#f1(time.time() - move_time) > f2(HorizontalDistance/(2*width)):
                if (x - x0) > 0: # El objeto está a la derecha del centro de la imagen
                    arduino_serial.write(b'r') # 68 ASCII
                    haptic_messages.append('r')
                else: # El objeto está a la izquierda del centro de la imagen
                    arduino_serial.write(b'l') # 76 ASCII
                    haptic_messages.append('l')
                move_time = time.time()
        else:
            if True:#f1(time.time() - move_time) > f2(VerticalDistance/(2*height)):
                if (y - y0) > 0: # El objeto está abajo del centro de la imagen
                    arduino_serial.write(b'd') # 82 ASCII
                    haptic_messages.append('d')
                else: # El objeto está arriba del centro de la imagen
                    arduino_serial.write(b'u') # 85 ASCII
                    haptic_messages.append('u')
                move_time = time.time()
        mentioned_object = False

###################################################
########## CONSTANTES Y CONFIGURACIONES ###########
###################################################

# Ancho y alto de la imagen de profundidad
depthWidth = monoLeft.getResolutionWidth()
depthHeight = monoLeft.getResolutionHeight()
# Coordenadas del centro de la imagen RGB y el mápa de dispariedad
x0, y0 = width//2, height//2
X0, Y0 = depthWidth//2, depthHeight//2
# Video para las detecciones y el mapa de profundidad
VideoRGB = cv2.VideoWriter('VideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
VideoDepth = cv2.VideoWriter('VideoDepth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
# Región de interés (ROI) para la estimación de la distancia de obstáculos z
DELTA_ACTIVATION = 30
DELTA = 30
xmin, ymin, xmax, ymax = X0-DELTA, Y0-DELTA, X0+DELTA, Y0+DELTA
# Estilos de dibujo (colores y timpografía)
BoxesColor = (0, 255, 0)
BoxesSize = 2
LineColor = (0, 0, 255)
CircleColor = (255, 0, 0)
TextColor = (0, 255, 255)
FontFace = cv2.FONT_HERSHEY_SIMPLEX
FontSize = 1

#####################################################
#################### MAIN PROGRAM ###################
#####################################################

# Muestras de tiempo y distancia para graficar
sample_times = []      # Muestras de los tiempos de ejecución del programa desde la primera captura de frame
d = []   # Muestras de la distancias a la detecion más cercana con visión monocular
z = []   # Muestras de las distancias a los obstáculos en la ROI central con visión estereoscópica
dist2UpperROI = []     # Muestras de las distancias a los obstáculos en la ROI superior con visión estereoscópica
haptic_messages  = []  # Muestras de los mensajes de activación del vibrador UP de la interfaz háptica
nearest_labels = []  # Muestras de la etiquta de la deteción más cercana

# Declaración de variables
mentioned_object = False # bandera booleana para evitar que se repita el nombre del objeto detectado
frame_start_time = 0
frames_timer = 0
frames_counter = 0
move_time = 0

loop_start_time = time.time()
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
        x, y, index = Nearest_Coordinate((x0,y0), Centroids)
        
        # Almacenar distancia e indice de la clase detectada del objeto más cercano 
        nearest_labels.append(labels[detections[index].label])
        d.append(Distance3D(detections[index]))

        # Dibujar una flecha que indique el objeto más cercano desde centro de la imágen
        cv2.arrowedLine(frame, (x0, y0), (x, y), LineColor, 2)

        if arduino_is_connected: # Efectuar interacción háptica si se estableció la comunicación serial
            Haptic_Interface(x,y)
        else:
            haptic_messages.append("nan")
    else:
        nearest_labels.append("nan")
        d.append("nan")

    # Esimacion de la profundidad con las camaras estereoscopicas
    depthROI = depthFrame[ymin:ymax, xmin:xmax]
    #inRange = (THRESH_LOW <= depthROI) & (depthROI <= THRESH_HIGH)
    z.append(np.mean(depthROI)/1000) # Almacenar la distancia a los obstáculos en la lista "d"

    ## Crear una matriz de cerros con las mismas dimensiones que el frame de profundidad
    #depth_filtered_frame = np.zeros((depthFrame.shape[0], depthFrame.shape[1]), dtype=np.uint8)
    #for i in range(0, depthHeight):
    #    for j in range(0, depthWidth):
    #        if depthFrame[i,j] > 1000 and depthFrame[i,j] < 1500: # Filtrar los valores de profundidad
    #            depth_filtered_frame[i,j] = depthFrame[i,j]
    #        else:
    #            depth_filtered_frame[i,j] = 0
#
    ## Colorear mapas de dispariedad con y sin filtrado
    #depth_filtered_frame = cv2.applyColorMap(depth_filtered_frame, cv2.COLORMAP_AUTUMN)
    #depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    #depthFrameColor = cv2.equalizeHist(depthFrameColor)
    #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_AUTUMN)
    

    # Contador de FPS 
    frames_counter += 1
    current_time = time.time()
    sample_times.append(current_time - loop_start_time) # Almacenar el tiempo de ejecución
    if (current_time - frames_timer) > 1:
        fps = ( frames_counter / (current_time - frames_timer) )
        frames_counter = 0
        frames_timer = current_time
    
    # Mostrar: tiempo de ejecución, fps, distancia promedio a los obstáculos y anotaciones de las detecciones
    #cv2.putText(depthFrameColor, ("{:.2f} [m]".format(z[-1]) ), (X0, Y0+2*DELTA), FontFace, FontSize, TextColor)
    cv2.rectangle(frame, (x0-DELTA, y0-DELTA), (x0+DELTA, y0+DELTA), (0, 0, 255), 2)
    cv2.circle(frame, (x0, y0), 5, CircleColor, -1)
    cv2.putText(frame, ("{:.2f} [m]".format(z[-1]) ), (x0+DELTA, y0), FontFace, FontSize, TextColor)
    cv2.putText(frame, "fps: {:.2f}".format(fps), (0,height-FontSize-4), FontFace, FontSize, TextColor)
    cv2.putText(frame, "t: " + ("{:.2f} s".format(sample_times[-1])), (0, 25), FontFace, FontSize, TextColor)
    
    # Mostrar ventanas de video
    cv2.imshow("RGB", frame)
    #cv2.imshow("Disparity Map", depthFrameColor)
    #cv2.imshow("Filtered disparity map", depth_filtered_frame)

    # Mostrar por consola los datos almacenados rescribiendo en la misma línea
    print(
        "sample: " + str(len(sample_times)),
        "t: {:.2f} [s]".format(sample_times[-1]),
        "fps: {:.2f}".format(fps),
        "z: {:.2f} [m]".format(z[-1]),
        "class: " + str(nearest_labels[-1]) + " d: {:.2f} [m]".format(d[-1]) + " msg: " + haptic_messages[-1] if d[-1] != "nan" else " ",
    end="\r", sep=" ")

    # Guardar fotograma 100 de profundidad, profundidad filtrada y frame RGB en una imagenenes en la carpeta images
    #if len(sample_times) == 100: 
    #    if not os.path.exists("images"):# crear una carpeta para guardar las imagenes si no existe
    #        os.makedirs("images")
    #    cv2.imwrite("images/depthFrame_"+str(len(sample_times))+".png", depthFrameColor)
    #    cv2.imwrite("images/depth_filtered_frame_"+str(len(sample_times))+".png", depth_filtered_frame)
    #    cv2.imwrite("images/rgbFrame_"+str(len(sample_times))+".png", frame)

    # Gabaciones de video
    #VideoRGB.write(frame)
    #VideoDepth.write(depthFrameColor)

# Cerrar todas las ventanas y apagar la cámara
cv2.destroyAllWindows()
device.close()
#VideoDepth.release()
#VideoRGB.release()

# Guardar los datos de las listas en un archivo .csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_times", "z", "d", "nearest_labels", "haptic_messages"])
    writer.writerows(zip(sample_times, z, d, nearest_labels, haptic_messages))

# Leer los datos del archivo .csv
data = pd.read_csv('data.csv')

# Graficar distancia
fig = plt.figure(figsize=(10, 5))

# Graficar mensajes de la interfaz háptica como lineas verticales
for i in range(len(data.sample_times)):
    if data.haptic_messages[i] == "l":
        # graficar una linea vertical azul delgada con un texto "l" en la parte superior
        plt.axvline(x=data.sample_times[i], color='b', linewidth=0.1)
        plt.text(data.sample_times[i], 0.5, "l", color='b', fontsize=8)
    elif data.haptic_messages[i] == "r":
        # graficar una linea vertical roja delgada con un texto "r" en la parte superior
        plt.axvline(x=data.sample_times[i], color='r', linewidth=0.1)
        plt.text(data.sample_times[i], 0.5, "r", color='r', fontsize=8)
    elif data.haptic_messages[i] == "f":
        # graficar una linea vertical verde delgada con un texto "f" en la parte superior
        plt.axvline(x=data.sample_times[i], color='g', linewidth=0.1)
        plt.text(data.sample_times[i], 0.5, "f", color='g', fontsize=8)
        

plt.plot(data.sample_times, data.d, label='Distancia a la deteción más cercana a la ROI')
plt.plot(data.sample_times, data.z, label='Distancia promedio al obstáculo en la ROI') 
plt.xlabel('Tiempo [s]')
plt.ylabel('Distancia [m]')
plt.legend()
plt.savefig('graph.svg', format='svg', dpi=1200)
plt.show()