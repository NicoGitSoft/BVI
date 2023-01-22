import serial, time, os, json, math, csv, cv2
import numpy as np
import depthai as dai

#Cambiar la ruta de ejecución aquí
MainDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(MainDir)

visualize = True
try:#Intentar establecer un objeto para comunicación serial a usando UART
    serial = serial.Serial("/dev/ttyS1", 9600, timeout=1)
    serial_is_connected = True
    print("/dev/ttyS1 CONECTED")
except:
    serial_is_connected = False
    print("/dev/ttyS1  CONECTED")

#Ruta del modelo la configuración de la red neuronal entrenada para la deteción de objetos
MODEL_PATH = os.path.join(MainDir, './Models/MyModelYOLOv7tiny', "best_openvino_2021.4_6shave.blob")
CONFIG_PATH = os.path.join(MainDir, './Models/MyModelYOLOv7tiny', "best.json")

#Extraer metadata del archivo de configuración .json
with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)
metadata = config.get("nn_config").get("NN_specific_metadata")
classes = metadata.get("classes")
coordinates = metadata.get("coordinates")
anchors = metadata.get("anchors")
anchorMasks = metadata.get("anchor_masks")
iouThreshold = metadata.get("iou_threshold")
confidenceThreshold = metadata.get("confidence_threshold")

#Extraer labels del archivo de configuración .json
labels = config.get("mappings").get("labels")
#Ancho y alto de la imagen de entrada a la red neuronal
width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

voice_messages = [
    'símbolo avión', 'recogida de equipajes', 'baños', 'peligro electricidad', 'flecha hacia abajo',
    'flecha hacia abajo de emergencia', 'salida de emergencia', "flecha izquierda de emergencia",
    "flecha derecha de emergencia", "flecha arriba de emergencia", "símbolo de extintor", "extintor", 
    "preferencial", "flecha izquierda", "no pasar", "restaurantes", "flecha derecha", "flecha izquierda",
    "flecha derecha", "flecha arriba"]

#Crear canalización
pipeline = dai.Pipeline()

# ColorCamera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(width, height)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_out = pipeline.create(dai.node.XLinkOut)
cam_out.setStreamName("cam_out")

# MonoCameras & StereoDepth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Alinee el mapa de profundidad con la perspectiva de la cámara RGB
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_out = pipeline.create(dai.node.XLinkOut)
stereo_out.setStreamName("stereo_out")

# YoloSpatialDetectionNetwork
yoloSpatial = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
yoloSpatial.setBlobPath(MODEL_PATH)
yoloSpatial.setConfidenceThreshold(0.5)
yoloSpatial.input.setBlocking(False)
yoloSpatial.setBoundingBoxScaleFactor(0.5)
yoloSpatial.setDepthLowerThreshold(350)
yoloSpatial.setDepthUpperThreshold(5000)
yoloSpatial.setNumClasses(classes)
yoloSpatial.setCoordinateSize(coordinates)
yoloSpatial.setAnchors(anchors)
yoloSpatial.setAnchorMasks(anchorMasks)
yoloSpatial.setIouThreshold(0.5)
yoloSpatial.setConfidenceThreshold(0.6)
# Outputs of YoloSpatialDetectionNetwork
yolo_out = pipeline.create(dai.node.XLinkOut)
yoloSpatial_out = pipeline.create(dai.node.XLinkOut)
yolo_out.setStreamName("yolo_out")
yoloSpatial_out.setStreamName("yoloSpatial_out")

# links
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# YoloSpatialDetectionNetwork node links
cam.preview.link(yoloSpatial.input)
stereo.depth.link(yoloSpatial.inputDepth)
yoloSpatial.passthrough.link(cam_out.input)
yoloSpatial.passthroughDepth.link(stereo_out.input)
yoloSpatial.out.link(yolo_out.input)
yoloSpatial.outNetwork.link(yoloSpatial_out.input)



# Conéctese al dispositivo e inicie la canalización
device =dai.Device(pipeline)

#Las colas de salida se utilizarán para obtener los marcos rgb y los datos nn de las salidas definidas anteriormente.
previewQueue = device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
detectionNNQueue = device.getOutputQueue(name="yolo_out", maxSize=4, blocking=False)
depthQueue = device.getOutputQueue(name="stereo_out", maxSize=4, blocking=False)
networkQueue = device.getOutputQueue(name="yoloSpatial_out", maxSize=4, blocking=False)

#####################################################
######### FUNCIONES PARA EL PROCESAMIENTO ###########
#####################################################
#funciones anónimas para incremento de pulsos exponencial en los vibradores
f1 = lambda x: math.sqrt(1 + x) - 1
f2 = lambda x: (x + 1)**2 - 1

#Calcular coordenadas de los vertices un bounding box
def Vertices(detection):        
    x1 = int(detection.xmin * width)
    x2 = int(detection.xmax * width)
    y1 = int(detection.ymin * height)
    y2 = int(detection.ymax * height)
    return x1, x2, y1, y2

#Calcular la coordenada del centro de un bounding box
def Center(x1, x2, y1, y2):
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y

# Determina las coordenadas del centro del bounding box más cercano y el índice correspondiente
def Nearest_Coordinate(OriginPoint, Centroids):
    x0, y0 = OriginPoint
    minDist = min((x-x0)**2 + (y-y0)**2 for x, y in Centroids)
    for index, (x, y) in enumerate(Centroids):
        if (x-x0)**2 + (y-y0)**2 == minDist:
            return x, y, index

#Calcular la distancia en 3D de un objeto a la camara
def Distance3D(detection):
    X = detection.spatialCoordinates.x
    Y = detection.spatialCoordinates.y
    Z = detection.spatialCoordinates.z
    return math.sqrt(X**2 + Y**2 + Z**2)/1000

def FilterDepth(depthFrame, dmin, dmax):
    #Crear una matriz de cerros con las mismas dimensiones que el frame de profundidad
    depth_filtered_frame = np.zeros((depthFrame.shape[0], depthFrame.shape[1]), dtype=np.uint8)
    for i in range(0, depthHeight):
        for j in range(0, depthWidth):
            if depthFrame[i,j] > dmin and depthFrame[i,j] < dmax: #Filtrar los valores de profundidad
                depth_filtered_frame[i,j] = depthFrame[i,j]
            else:
                depth_filtered_frame[i,j] = 0
    return depth_filtered_frame

###################################################
########## CONSTANTES Y CONFIGURACIONES ###########
###################################################
#Ancho y alto de la imagen de profundidad
depthWidth = monoLeft.getResolutionWidth()
depthHeight = monoLeft.getResolutionHeight()
#Coordenadas del centro de la imagen RGB y el mápa de dispariedad
x0, y0 = width//2, height//2
X0, Y0 = depthWidth//2, depthHeight//2

#Video para las detecciones y el mapa de profundidad
VideoRGB = cv2.VideoWriter('VideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
#VideoDepth = cv2.VideoWriter('VideoDepth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
#Región de interés (ROI) para la estimación de la distancia de obstáculos z
DELTA = 30 #Pixeles
xmin, ymin, xmax, ymax = X0-DELTA, Y0-DELTA, X0+DELTA, Y0+DELTA
bps = 10 #bits por segundo (envío de datos para sonar el buzzer por segundo)
zmin, zmax = 1, 2 #Distancia mínima y máxima para sonar el buzzer
#Estilos de dibujo (colores y timpografía)
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
#Muestras de tiempo y distancia para graficar
times = []      #Muestras de los tiempos de ejecución del programa desde la primera captura de frame
d = []   #Muestras de la distancias a la detención más cercana con visión monocular
z = []   #Muestras de las distancias a los obstáculos en la ROI central con visión estereoscópica
h = []   #Muestras de las distancias horizontales en piexeles de la detección más cercana al centro de la imagen
v = []   #Muestras de las distancias verticales en piexeles de la detección más cercana al centro de la imagen
dist2UpperROI = []     #Muestras de las distancias a los obstáculos en la ROI superior con visión estereoscópica
haptic_messages  = []  #Muestras de los mensajes de activación del vibrador UP de la interfaz háptica
buzzer_messages  = []  #Muestras de los mensajes de activación del buzzer
nearest_labels = []  #Muestras de la etiqueta de la detención más cercana
#Declaración de variables
mentioned_object = False #bandera booleana para evitar que se repita el nombre del objeto detectado
frame_start_time = 0
frames_timer = 0
frames_counter = 0
start_move_time = 0
start_buzze_time = 0

loop_start_time = time.time()
while True:
    #Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q}
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

    #Extraer datos del dispositivo OAK-D
    frame = previewQueue.get().getCvFrame()         #Obtener el fotograma de la cámara RGB
    depthFrame = depthQueue.get().getFrame()        #Obtener el fotograma de profundidad
    detections = detectionNNQueue.get().detections  #Obtener las detecciones de la red neuronal
#Estimación de la profundidad con las camaras estereoscopicas
    depthROI = depthFrame[ymin:ymax, xmin:xmax]
    #Almacenar la distancia a los obstáculos en la lista "d"
    z.append(np.mean(depthROI)/1000)

    #Comunicación serial para el buzzer
    if serial_is_connected:
        """ Uso del buzzer para sonar más frecuentemente a medida que se acerca un objeto en la ROI central"""
        if z[-1] < zmax and z[-1] > zmin: #si el obstáculo está a una distancia menor a zmax y mayor a zmin
#Calcular el tiempo de espera entre cada envío de datos para sonar el buzzer
            wait_time_buzzer = 1/(bps)*z[-1]/(zmax-zmin) #Segundos
            if f1(time.time() - start_buzze_time) > f2(wait_time_buzzer):
                #Sonar el buzzer
                serial.write(b'0')
                start_buzze_time = time.time()
                buzzer_messages.append('1')
            else:
                buzzer_messages.append('nan')
        else:
            buzzer_messages.append('nan')
    
    if len(detections) != 0: #si hay deteciones de la camara RGB
        Centroids = []  #Coordenadas del centro de los objetos detectados
        for detection in detections: #Para cada detección
#Calcular los vertices de la caja delimitadora
            x1, x2, y1, y2 = Vertices(detection)
            #Calcular el centro de la caja delimitadora y agregarlo a la lista de centroides
            x, y = Center(x1, x2, y1, y2)
            Centroids.append((x, y))
            if visualize:
                #Calcular la distancia a la caja delimitadora, la confianza y la etiqueta
                distance = Distance3D(detection)
                detection_label = str(labels[detection.label])
                confidence = detection.confidence*100
                #Escribir información de la detección en el frame
                cv2.putText(frame, detection_label , (x1, y1), FontFace, FontSize, TextColor, 2)
                cv2.putText(frame, "{:.0f} %".format(confidence), (x2, y), FontFace, FontSize, TextColor, 1)
                cv2.putText(frame, "{:.2f} [m]".format(distance) , (x2, y2), FontFace, FontSize, TextColor)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BoxesColor, BoxesSize)
        # Determina las coordenadas del centro del bounding box más cercano y el índice correspondiente
        x, y, index = Nearest_Coordinate((x0,y0), Centroids)
        # Almacenar distancia e indice de la clase detectada del objeto más cercano
        nearest_labels.append(labels[detections[index].label])
        d.append(Distance3D(detections[index]))
        # Dibujar una flecha que indique el objeto más cercano desde centro de la imágen
        if visualize: cv2.arrowedLine(frame, (x0, y0), (x, y), LineColor, 2)

        #Comunicación serial para actuadores vibrotáctiles y buzzer
        if serial_is_connected:
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
                    serial.write(b'0')
                    mentioned_object = True
                else:
                    haptic_messages.append('nan')
            else:
                if h[-1] > v[-1]:
                    wait_time = 1/bps * h[-1]/((width-DELTA)//2)
                    if f1(time.time() - start_move_time) > f2(wait_time):
                        if (x - x0) > 0: #El objeto está a la derecha del centro de la imagen
                            serial.write(b'r') #68 ASCII
                            haptic_messages.append('r')
                        else: #El objeto está a la izquierda del centro de la imagen
                            serial.write(b'l') #76 ASCII
                            haptic_messages.append('l')
                        start_move_time = time.time()
                    else:
                        haptic_messages.append('nan')
                else:
                    wait_time = 1/bps * v[-1]/((height-DELTA)//2)
                    if f1(time.time() - start_move_time) > f2(wait_time):
                        if (y - y0) > 0: #El objeto está abajo del centro de la imagen
                            serial.write(b'd') #82 ASCII
                            haptic_messages.append('d')
                        else: #El objeto está arriba del centro de la imagen
                            serial.write(b'u') #85 ASCII
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
    
    #Contador de FPS
    frames_counter += 1
    current_time = time.time()
    times.append(current_time - loop_start_time) #Almacenar el tiempo de ejecución
    if (current_time - frames_timer) > 1:
        fps = ( frames_counter / (current_time - frames_timer) )
        frames_counter = 0
        frames_timer = current_time

    if visualize:
        #Colorear mapas de dispariedad con y sin filtrado
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_AUTUMN)
        #Mostrar: tiempo de ejecución, fps, distancia promedio a los obstáculos y anotaciones de las detecciones
        cv2.rectangle(frame, (x0-DELTA, y0-DELTA), (x0+DELTA, y0+DELTA), (0, 0, 255), 2)
        cv2.circle(frame, (x0, y0), 5, CircleColor, -1)
        cv2.putText(frame, ("{:.2f} [m]".format(z[-1]) ), (x0+DELTA, y0), FontFace, FontSize, TextColor, 3)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (0,height-FontSize-4), FontFace, FontSize, TextColor, 3)
        cv2.putText(frame, "t: " + ("{:.2f} s".format(times[-1])), (0, 25), FontFace, FontSize, TextColor, 3)
        #Mostrar ventanas de video
        cv2.imshow("Disparity Map", depthFrameColor)
        cv2.imshow("RGB", frame)
        #Guardar fotograma en el videoRGB
        VideoRGB.write(frame)

#Cerrar todas las ventanas y apagar la cámara
if visualize: cv2.destroyAllWindows()
device.close()

#Guardar los datos de las listas en un archivo .csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["times", "z", "d", "h", "v", "nearest_labels", "haptic_messages", "buzzer_messages"])
    writer.writerows(zip(times, z, d, h, v, nearest_labels, haptic_messages, buzzer_messages))
