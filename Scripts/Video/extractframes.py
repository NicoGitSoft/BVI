# este es un script para extraer n frames de un video de forma aleatoria, y guardarlos en una carpeta.

import cv2, os, random

# cambiar directorio de trabajo a la carpeta actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# abrir el video y contar el numero de frames
namevideo = "VideoRGB_Hand"
cap = cv2.VideoCapture(namevideo+".mp4")
FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("numero de frames: ", FRAME_COUNT)

# crear una carpeta para guardar los frames si es que no existe
if not os.path.exists(namevideo):
    os.makedirs(namevideo)

# Lista de frames a extraer (aleatorios)
numFames = FRAME_COUNT//2
ramdomFramesList = random.sample(range(1, FRAME_COUNT), numFames)

# leer el video frame por frame
for currentFrame in range(FRAME_COUNT):
    # leer el frame actual
    _ , frame = cap.read()

    if currentFrame in ramdomFramesList:
        # guardar el frame en la carpeta frames
        name = str(namevideo) + "_" + str(currentFrame) + '.png'
        # ruta absoluta de la imagen
        path = os.path.abspath("./" + namevideo + "/" + name) 
        # guardar el frame en la carpeta namevideo
        cv2.imwrite(os.path.join(namevideo, name), frame)
        print (path + ' guardado!')
    else:
        currentFrame += 1

