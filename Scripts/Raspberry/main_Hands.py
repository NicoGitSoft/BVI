import serial, time, os, json, math, csv, cv2
import numpy as np
import depthai as dai

# detectar si es raspberry pi o no
if os.uname()[4][:3] == 'arm':
    import max6675, subprocess
    # set the pin for communicate with MAX6675
    cs = 22
    sck = 18
    so = 16
    max6675.set_pin(cs, sck, so, 1)
    Measure = True
else:
    import mediapipe as mp
    Measure = False
    # crear un objeto para de mediaPipe para la detección una sola mano
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

# Cambiar la ruta de ejecución aquí
MainDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(MainDir)

visualize = True
try:# Intentar establecer un objeto para comunicación serial a usando UART 
    serial = serial.Serial("/dev/ttyS0", 9600, timeout=1)
    serial_is_connected = True
except:
    serial_is_connected = False

# Ruta del modelo la configuración de la red neuronal entrenada para la deteción de objetos
MY_YOLO_MODEL = os.path.join(MainDir, './Models/MyModelYOLOv7tiny', "best_openvino_2021.4_6shave.blob")
CONFIG_PATH = os.path.join(MainDir, './Models/MyModelYOLOv7tiny', "best.json")

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
labels = config.get("mappings").get("labels")
# Anhcho y alto de la imagen de entrada a la red neuronal
width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

voice_messages = [
    'símbolo avión', 'recogida de equipajes', 'baños', 'peligro electricidad', 'flecha hacia abajo',
    'flecha hacia abajo de emergencia', 'salida de emergencia', "flecha izquierda de emergencia",
    "flecha derecha de emergencia", "flecha arriba de emergencia", "símbolo de extintor", "extintor", 
    "preferencial", "flecha izquierda", "no pasar", "restaurantes", "flecha derecha", "flecha izquierda",
    "flecha derecha", "flecha arriba"]

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

# SystemLogger
sysLog = pipeline.create(dai.node.SystemLogger)
sysLog.setRate(1)  # 1 Hz
temp_out = pipeline.create(dai.node.XLinkOut)
temp_out.setStreamName("temp_out")

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

# YoloDetectionNetwork
yolo = pipeline.create(dai.node.YoloDetectionNetwork)
yolo.setBlobPath(MY_YOLO_MODEL)
yolo.setConfidenceThreshold(confidenceThreshold)
yolo.input.setBlocking(False)
yolo.setNumClasses(classes)
yolo.setCoordinateSize(coordinates)
yolo.setAnchors(anchors)
yolo.setAnchorMasks(anchorMasks)
yolo.setIouThreshold(iouThreshold)
yolo_out = pipeline.create(dai.node.XLinkOut)
yolo_out.setStreamName("yolo_out")

# Linking nodes for the output of yolo_out
cam.preview.link(yolo.input)            # cam.preview -> yolo.input
yolo.passthrough.link(cam_out.input)    # yolo.passthrough -> cam_out.input
yolo.out.link(yolo_out.input)           # yolo.out -> yolo_out.input

# Linking nodes for the output of stereo_out
left.out.link(stereo.left)              # left.out -> stereo.left
right.out.link(stereo.right)            # right.out -> stereo.right
stereo.depth.link(stereo_out.input)     # stereo.depth -> stereo_out.input

# Linking nodes for the output of temp_out
sysLog.out.link(temp_out.input)         # sysLog.out -> temp_out.input

# Connect to device and start pipeline
device = dai.Device(pipeline)

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_cam_out = device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
q_yolo_out = device.getOutputQueue(name="yolo_out", maxSize=4, blocking=False)
q_stereo_out = device.getOutputQueue(name="stereo_out", maxSize=4, blocking=False)
q_temp_out = device.getOutputQueue(name="temp_out", maxSize=4, blocking=False)

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

# Determina las coordenasdas del centro del bounding box más cercano y el índice correspondiente
def Nearest_Coordinate(OriginPoint, Centroids):
    x0, y0 = OriginPoint
    minDist = min((x-x0)**2 + (y-y0)**2 for x, y in Centroids)
    for index, (x, y) in enumerate(Centroids):
        if (x-x0)**2 + (y-y0)**2 == minDist:
            return x, y, index

# Calcular la distancia en 3D de un objeto a la camara
def Distance3D(detection):
    X = detection.spatialCoordinates.x
    Y = detection.spatialCoordinates.y
    Z = detection.spatialCoordinates.z
    return math.sqrt(X**2 + Y**2 + Z**2)/1000

# Función que crea una ROI a partir de una coordenada y un radio DELTA
def ROI(COORDINATE, DELTA):
    x, y = COORDINATE
    x1 = x - DELTA
    x2 = x + DELTA
    y1 = y - DELTA
    y2 = y + DELTA
    return x1, x2, y1, y2

# Función que calcula el promedio de los valores de profundidad de un ROI en metros
def AverageDepth(ROI, depthFrame):
    x1, x2, y1, y2 = ROI
    depth = depthFrame[y1:y2, x1:x2]
    return np.mean(depth)/1000

# Función que entrega la posición de la punta del dedo índice
def IndexTipPosition(results_hands, width, height):
    for hand_landmarks in results_hands.multi_hand_landmarks:
        x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
        y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
    return x1, y1

# Función que entrega la posición de la punta de la muñeca
def WristPosition(results_hands, width, height):
    for hand_landmarks in results_hands.multi_hand_landmarks:
        x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
        y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
    return x2, y2


###################################################
########## CONSTANTES Y CONFIGURACIONES ###########
###################################################

# Ancho y alto de la imagen de profundidad 
depthWidth = left.getResolutionWidth()
depthHeight = right.getResolutionHeight()
# Coordenadas del centro del mápa de dispariedad
X0, Y0 = depthWidth//2, depthHeight//2

# Video para las detecciones y el mapa de profundidad
VideoRGB = cv2.VideoWriter('VideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
#VideoDepth = cv2.VideoWriter('VideoDepth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Región de interés (ROI) para la estimación de la distancia de obstáculos z
DELTA = 30 # pixeles
CentralROI = ROI((X0, Y0), DELTA)
#xmin, ymin, xmax, ymax = X0-DELTA, Y0-DELTA, X0+DELTA, Y0+DELTA
bps = 10 # bits por segundo (envío de datos para sonar el buzzer por segundo)
zmin, zmax = 1, 2 # Distancia mínima y máxima para sonar el buzzer

# Estilos de dibujo (colores y timpografía)
BoxesColor = (0, 255, 0)
BoxesSize = 2
LineColor = (0, 0, 255)
CircleColor = (255, 0, 0)
TextColor = (0,0,255)
FontFace = cv2.FONT_HERSHEY_SIMPLEX
FontSize = 1

#####################################################
#################### MAIN PROGRAM ###################
#####################################################

# Muestras de tiempo y distancia para graficar
times = []      # Muestras de los tiempos de ejecución del programa desde la primera captura de frame
d = []   # Muestras de la distancias a la detecion más cercana con visión monocular
z = []   # Muestras de las distancias a los obstáculos en la ROI central con visión estereoscópica
h = []   # Muestras de las distancias horizontales en piexeles de la detección más cercana al centro de la imagen
v = []   # Muestras de las distancias verticales en piexeles de la detección más cercana al centro de la imagen 
dist2UpperROI = []     # Muestras de las distancias a los obstáculos en la ROI superior con visión estereoscópica
haptic_messages  = []  # Muestras de los mensajes de activación del vibrador UP de la interfaz háptica
buzzer_messages  = []  # Muestras de los mensajes de activación del buzzer
nearest_labels = []  # Muestras de la etiquta de la deteción más cercana

# Muestras de temperatura
chipTemperature = []  # Muestras de la temperatura del chip
cpuTemperature = []  # Muestras de la temperatura del CPU
max6675Temperature = []  # Muestras de la temperatura del sensor DHT22

# Declaración de variables
mentioned_object = False # bandera booleana para evitar que se repita el nombre del objeto detectado
use_finger = False # bandera booleana para activar el uso del dedo índice y no el centro de la imagen
frame_start_time = 0
frames_timer = 0
frames_counter = 0
start_move_time = 0
start_buzze_time = 0

loop_start_time = time.time()
while True:
    # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q} 
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

    # Extraer datos del dispositivo OAK-D 
    frame = q_cam_out.get().getCvFrame()         # Obtener el fotograma de la cámara RGB
    depthFrame = q_stereo_out.get().getFrame()        # Obtener el fotograma de profundidad
    detections = q_yolo_out.get().detections  # Obtener las detecciones de la red neuronal

    # Almacenar la distancia de los obstáculos en la ROI central
    z.append(AverageDepth(CentralROI, depthFrame))
    
    # Comunicación serial para el buzzer
    """ Uso del buzzer para sonar más frecuentemente a medida que se acerca un objeto en la ROI central"""
    if z[-1] < zmax and z[-1] > zmin: # si el obstáculo está a una distancia menor a zmax y mayor a zmin
        # Calcular el tiempo de espera entre cada envío de datos para sonar el buzzer
        wait_time_buzzer = 1/(bps)*z[-1]/(zmax-zmin) # segundos
        if f1(time.time() - start_buzze_time) > f2(wait_time_buzzer):
            if serial_is_connected: serial.write(b'1') # Sonar el buzzer
            start_buzze_time = time.time()
            buzzer_messages.append('1')
        else:
            buzzer_messages.append('nan')
    else:
        buzzer_messages.append('nan')

    if not Measure:
        # Definir el punto de referencia para la detección más cercana
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            wrist_position = WristPosition(results_hands, width, height)
            wrist_distance = AverageDepth(ROI(wrist_position, DELTA), depthFrame)
            # mostrar ROI del la muñeca
            cv2.rectangle(frame, (wrist_position[0]-DELTA, wrist_position[1]-DELTA), (wrist_position[0]+DELTA, wrist_position[1]+DELTA), BoxesColor, BoxesSize)
            # mostrar distancia de la muñeca en la ROI de la muñeca
            cv2.putText(frame, f'{wrist_distance:.2f} m', (wrist_position[0]-DELTA, wrist_position[1]-DELTA), FontFace, FontSize, TextColor)
            if wrist_distance < 1:
                x0, y0 = IndexTipPosition(results_hands, width, height)
            else:
                x0, y0 = width//2, height//2
    else:
        x0, y0 = width//2, height//2

    if len(detections) != 0: # si hay deteciones de la camara RGB
        Centroids = []  # Coordenadas del centro de los objetos detectados
        for detection in detections: # Para cada detección
            # Calcular los vertices de la caja delimitadora
            x1, x2, y1, y2 = Vertices(detection)
            # Calcular el centro de la caja delimitadora y agregarlo a la lista de centroides
            x, y = Center(x1, x2, y1, y2)
            Centroids.append((x, y))
            if visualize:
                # Calcular la distancia a la caja delimitadora, la confianza y la etiqueta
                distance = Distance3D(detection)
                detection_label = str(labels[detection.label])
                confidence = detection.confidence*100
                # Escribir información de la detección en el frame
                cv2.putText(frame, detection_label , (x1, y1), FontFace, FontSize, TextColor, 2)
                cv2.putText(frame, "{:.0f} %".format(confidence), (x2, y), FontFace, FontSize, TextColor, 1)
                cv2.putText(frame, "{:.2f} [m]".format(distance) , (x2, y2), FontFace, FontSize, TextColor)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BoxesColor, BoxesSize)
        # Determina las coordenasdas del centro del bounding box más cercano al punto de referencia (x0, y0)
        x, y, index = Nearest_Coordinate((x0,y0), Centroids)
        # Almacenar distancia e indice de la clase detectada del objeto más cercano 
        nearest_labels.append(labels[detections[index].label])
        d.append(Distance3D(detections[index]))
        # Dibujar una flecha que indique el objeto más cercano desde centro de la imágen
        if visualize: cv2.arrowedLine(frame, (x0, y0), (x, y), LineColor, 2)

        # Comunicación serial para actuadores vibrotáctiles y buzzer
        """ Envío de mensajes para indicarcaciones vibrotáctiles {up,down,left,right} al usuario,
        y mencionar el nombre del objeto detectado más cercano si se entra en la ROI central
        mediante una voz sintetizada en el microprocesador. """
        h.append( abs(x - x0) )
        v.append( abs(y - y0) )
        if h[-1] < DELTA and v[-1] < DELTA:
            """ Decir mensaje traducido al español, basado en el objeto detectado más cercano y 
            la distancia a la cámara a la cual se encuentra. """
            if not mentioned_object:
                haptic_messages.append('c')
                msg = voice_messages[detections[index].label]
                os.system("spd-say '" + msg + "{:.2f} [m]".format(distance) + " metros'")
                if serial_is_connected: serial.write(b'0')
                mentioned_object = True
            else:
                haptic_messages.append('nan')
        else:
            if h[-1] > v[-1]:
                wait_time = 1/bps * h[-1]/((width-DELTA)//2)
                if f1(time.time() - start_move_time) > f2(wait_time):
                    if (x - x0) > 0: # El objeto está a la derecha del centro de la imagen
                        if serial_is_connected: serial.write(b'r') # 68 ASCII
                        haptic_messages.append('r')
                    else: # El objeto está a la izquierda del centro de la imagen
                        if serial_is_connected: serial.write(b'l') # 76 ASCII
                        haptic_messages.append('l')
                    start_move_time = time.time()
                else:
                    haptic_messages.append('nan')
            else:
                wait_time = 1/bps * v[-1]/((height-DELTA)//2)
                if f1(time.time() - start_move_time) > f2(wait_time):
                    if (y - y0) > 0: # El objeto está abajo del centro de la imagen
                        if serial_is_connected: serial.write(b'd') # 82 ASCII
                        haptic_messages.append('d')
                    else: # El objeto está arriba del centro de la imagen
                        if serial_is_connected: serial.write(b'u') # 85 ASCII
                        haptic_messages.append('u')
                    start_move_time = time.time()
                else:
                    haptic_messages.append('nan')
            mentioned_object = False
    else:
        haptic_messages.append("nan")
        nearest_labels.append("nan")
        d.append("nan")
        h.append("nan")
        v.append("nan")
    
    # Contador de FPS 
    frames_counter += 1
    current_time = time.time()
    times.append(current_time - loop_start_time) # Almacenar el tiempo de ejecución
    if (current_time - frames_timer) > 1:
        fps = ( frames_counter / (current_time - frames_timer) )
        frames_counter = 0
        frames_timer = current_time
        # Medir temperaturas
        if Measure:
            chipTemperature.append(q_temp_out.get().chipTemperature.average) # Temperaratura del chip de la OAk-D
            max6675Temperature.append(max6675.read_temp(cs)) # Temperatura del sensor DHT22
            cpuTemperature.append(float(subprocess.check_output("vcgencmd measure_temp", shell=True).decode("utf-8").replace("temp=","").replace("'C\n",""))) # Temperatura de la CPU de la Raspberry Pi
            # Mostrar por consola las temperaturas
            print(
                "Chip temperature: {:.2f} °C".format(chipTemperature[-1]),
                "CPU temperature: {:.2f} °C".format(cpuTemperature[-1]), 
                "max6675 temperature: {:.2f} °C".format(max6675Temperature[-1]),
                sep = "\t"#, end = "\r"
            )

    if visualize:
        # Colorear mapas de dispariedad con y sin filtrado
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_AUTUMN)
        # Mostrar: tiempo de ejecución, fps, distancia promedio a los obstáculos y anotaciones de las detecciones
        cv2.rectangle(frame, (x0-DELTA, y0-DELTA), (x0+DELTA, y0+DELTA), (0, 0, 255), 2)
        cv2.putText(frame, ("{:.2f} [m]".format(z[-1]) ), (width//2+DELTA, height//2), FontFace, FontSize, TextColor, 3)
        cv2.circle(frame, (x0, y0), 8, CircleColor, -1)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (0,height-FontSize-4), FontFace, FontSize, TextColor, 3)
        cv2.putText(frame, "t: " + ("{:.2f} s".format(times[-1])), (0, 25), FontFace, FontSize, TextColor, 3)
        # Mostrar ventanas de video
        cv2.imshow("Disparity Map", depthFrameColor)
        cv2.imshow("RGB", frame)
        # Guardar fotograma en el videoRGB
        VideoRGB.write(frame)

# Cerrar todas las ventanas y apagar la cámara
if visualize: cv2.destroyAllWindows()
device.close()

# Guardar los datos de las listas en un archivo .csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["times", "z", "d", "h", "v", "nearest_labels", "haptic_messages", "buzzer_messages", "chipTemperature", "max6675Temperature", "cpuTemperature"])
    writer.writerows(zip(times, z, d, h, v, nearest_labels, haptic_messages, buzzer_messages, chipTemperature, max6675Temperature, cpuTemperature))
