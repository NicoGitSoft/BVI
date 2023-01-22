from HandTracker4 import HandTracker
import numpy as np
import scipy.io as sio
import cv2, serial, time, math, os

# Coordenadas de los vertices un bounding box
def Vertices(detection):        
    x1 = int(detection.xmin * width)
    x2 = int(detection.xmax * width)
    y1 = int(detection.ymin * height)
    y2 = int(detection.ymax * height)
    return x1, x2, y1, y2

# Coordenada del centro de un bounding box
def Center(x1, x2, y1, y2):
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y

# Coordenadas del centro del bounding box más cercano a un punto de referencia
def Nearest_Coordinate(refPoint, centroids):
    x_ref, y_ref = refPoint
    minDist = min((x-x_ref)**2 + (y-y_ref)**2 for x, y in centroids)
    for index, (x, y) in enumerate(centroids):
        if (x-x_ref)**2 + (y-y_ref)**2 == minDist:
            return x, y, index

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
    return np.nanmean(depth)/1000

# funciones anónimas para incremento de la frecuencia de pulsos en los vibradores
f1 = lambda x: math.sqrt(1 + x) - 1
f2 = lambda x: (x + 1)**2 - 1

##################### CONSTANTES Y CONFIGURACIONES #####################
width = height = 640 # Resolución de entrada de la red neuronal
VideoRGB = cv2.VideoWriter('VideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Región de interés (ROI) para la estimación de la distancia de obstáculos z
DELTA = 30 # Radio de la ROI Cental en pixeles
DELTA_DOLL = 5 # Umbral de distancia en pixeles entre la referencia y el centroide de la detección más cercana (centroid of the nearest detection CND)
x_center, y_center = width//2, height//2 # Coordenadas del centro de la imagen
CentralROI = ROI((x_center, y_center), DELTA)

# Configuración del buzzer
bps = 10 # bits por segundo (envío de datos para sonar el buzzer por segundo)
zmin, zmax = 1, 2 # Umbral de distancia mínima y máxima para sonar el buzzer

# Estilos de dibujo (colores y timpografía)
BoxesColor = (0, 255, 0)
BoxesSize = 2
LineColor = (0, 0, 255)
CircleColor = (255, 0, 0)
TextColor = (0,0,255)
FontFace = cv2.FONT_HERSHEY_SIMPLEX
FontSize = 1

voice_messages = [
    'símbolo avión', 'recogida de equipajes', 'baños', 'peligro electricidad', 'flecha hacia abajo',
    'flecha hacia abajo de emergencia', 'salida de emergencia', "flecha izquierda de emergencia",
    "flecha derecha de emergencia", "flecha arriba de emergencia", "símbolo de extintor", "extintor", 
    "preferencial", "flecha izquierda", "no pasar", "restaurantes", "flecha derecha", "flecha izquierda",
    "flecha derecha", "flecha arriba"]

##################### inicialización de variables #####################
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
mentioned_hand_use = False # bandera booleana para evitar que se repita el la mencion TTS del uso de la mano
frame_start_time = 0
frames_timer = 0
frames_counter = 0
start_move_time = 0
start_buzze_time = 0

##################### Inicialización de objetos #####################
tracker = HandTracker(xyz=True, solo=True, crop=True)

visualize = True
try:# Intentar establecer un objeto para comunicación serial a usando UART 
    serial = serial.Serial("/dev/ttyS0", 9600, timeout=1)
    serial_is_connected = True
except:
    serial_is_connected = False

##################### Bucle principal #####################
while True:
    frame, hands, yoloDetections, labels, width, height, depthFrame = tracker.next_frame()

    # Almacenar la distancia de los obstáculos en la ROI central
    z.append(AverageDepth(CentralROI, depthFrame))

    # Dedección de obstáculos en la ROI Central
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

    # Detección de la mano del usuario
    """ Si se detecta una mano, se determina si la distandia de la muñeca a la cámera es menor a 1 metro,
    si es así, se le informa al usuario que que su mano está siendo untilizada como referencia para la 
    detección del objeto más cercano, de lo contrario se usa centro de la imagen como referencia """

    x_ref, y_ref = (x_center, y_center) # Usar coordenadas del centro de la imagen como referencia en primera instancia
    if len(hands) > 0: # Si se detecta la mano del usuario, cambiar la referencia a la punta del dedo índice
        x_doll, y_doll = hands[0].landmarks[0,:2] # Coordenadas de la muñeca
        dollROI = ROI((x_doll, y_doll), DELTA_DOLL) # ROI de la muñeca
        dollDistance = AverageDepth(dollROI, depthFrame) # Distancia de la muñeca a la cámara
        if dollDistance < 1: # Si la distancia de la muñeca a la cámara es menor a 1 metro
            x_ref, y_ref = hands[0].landmarks[8,:2] # Usar coordenadas de la punta del dedo índice como referencia
            if not mentioned_hand_use:
                os.system("spd-say 'Dedo indice como referencia'") # Informar al usuario que su mano está siendo utilizada como referencia
                mentioned_hand_use = True # Activar la bandera para evitar que se repita el mensaje de uso de la mano
            if visualize: cv2.circle(frame, (x_ref, y_ref), 5, (0, 255, 0), -1) # Dibujar un círculo en la muñeca
        else:
            mentioned_hand_use = False # Desactivar la bandera para que se vuelva a mencionar el uso de la mano
            if visualize: cv2.circle(frame, (x_doll, y_doll), 5, (0, 0, 255), -1) # Dibujar un círculo en la muñeca

    # Detección del centro del bounding box más cercano al punto de referencia (CNBB)
    if len(yoloDetections) > 0:
        Centroids = []  # Coordenadas del centro de los objetos detectados
        for detection in yoloDetections:
            x1, x2, y1, y2 = Vertices(detection)
            x, y = Center(x1, x2, y1, y2)
            Centroids.append((x, y))
            label = labels[detection.label]
            if visualize:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))  
        x, y, index = Nearest_Coordinate((x_ref, y_ref), Centroids) # Coordenadas del CNBB
        nearest_labels.append(labels[yoloDetections[index].label]) # Almacenar la etiqueta del CNBB
        d.append(AverageDepth(Vertices(yoloDetections[index]), depthFrame)) # Almacenar la profundidad del CNBB
        if visualize: cv2.arrowedLine(frame, (x_ref, y_ref), (x, y), LineColor, 2)# Dibujar una flecha desde el punto de referencia al CNBB

        # Comunicación serial para actuadores vibrotáctiles y buzzer
        """ Envío de mensajes para indicarcaciones vibrotáctiles {up,down,left,right} al usuario,
        y mencionar el nombre del objeto detectado más cercano si se entra en la ROI central
        mediante una voz sintetizada en el microprocesador. """
        h.append( abs(x - x_ref) )
        v.append( abs(y - y_ref) )
        if h[-1] < DELTA and v[-1] < DELTA:
            """ Decir mensaje traducido al español, basado en el objeto detectado más cercano y 
            la distancia a la cámara a la cual se encuentra. """
            if not mentioned_object:
                haptic_messages.append('c')
                msg = voice_messages[yoloDetections[index].label]
                os.system("spd-say '" + msg + "{:.2f} [m]".format(d[-1]) + " metros'")
                if serial_is_connected: serial.write(b'0')
                mentioned_object = True
            else:
                haptic_messages.append('nan')
        else:
            if h[-1] > v[-1]:
                wait_time = 1/bps * h[-1]/((width-DELTA)//2)
                if f1(time.time() - start_move_time) > f2(wait_time):
                    if (x - x_ref) > 0: # El objeto está a la derecha del centro de la imagen
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
                    if (y - y_ref) > 0: # El objeto está abajo del centro de la imagen
                        if serial_is_connected: serial.write(b'd') # 82 ASCII
                        haptic_messages.append('d')
                    else: # El objeto está arriba del centro de la imagen
                        if serial_is_connected: serial.write(b'u') # 85 ASCII
                        haptic_messages.append('u')
                    start_move_time = time.time()
                else:
                    haptic_messages.append('nan')
            mentioned_object = False

    if visualize: 
        # Visualizar la mapa de profundidad
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_AUTUMN)
        cv2.imshow("depth", depthFrameColor)

        # Mostrar el frame CentralROI y la profundidad de los obstáculos en la ROI central
        cv2.rectangle(frame, (CentralROI[0], CentralROI[2]), (CentralROI[1], CentralROI[3]), (0, 255, 0), 2)
        cv2.putText(frame, "z = {:.2f} m".format(z[-1]), (CentralROI[0], CentralROI[2]), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
        cv2.imshow("frame", frame)
    
    #frame = renderer.draw(frame, hands)
    #key = renderer.waitKey(delay=1)

    # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q}
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

tracker.exit()

# Guardar los datos en un archivo .mat
sio.savemat('data.mat', {'x': x, 'y': y, 'z': z, 'd': d, 'h': h, 'v': v, 'haptic_messages': haptic_messages, 'nearest_labels': nearest_labels})