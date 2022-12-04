import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos del archivo .csv
data = pd.read_csv("./Scripts/YoloDepthAI_for_BVI/data.csv")
N = len(data) # Número de muestras en el archivo .csv
n = [i for i in range(N)] # lista de números enteros de 0 a N-1
bps = 10 # bits por segundo (envío de datos para sonar el buzzer por segundo)
zmax, zmin = 2, 1 # Distancia máxima y mínima para sonar el buzzer

# mostrar el largo de todas las variables de data
print(
    "times: ", len(data['times']),
    "z: ", len(data['z']),
    "d: ", len(data['d']),
    "h: ", len(data['h']),
    "v: ", len(data['v']),
    "haptic_messages: ", len(data['haptic_messages']),
    "buzzer_messages: ", len(data['buzzer_messages']),
    "nearest_labels: ", len(data['nearest_labels'])
    )

# Graficar buzzer_messages y distancia z
fig = plt.figure(figsize=(15, 8))
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15)
panel = fig.add_subplot(2, 1, 1)
plt.grid(color='gray', linestyle=':', alpha=.2)# Agregar grid minor con lineas, punteadas y color gris claro
# Graficar mensajes del buzzer como lineas verticales
for i in range(N):
    if data.buzzer_messages[i] == 1:
        plt.axvline(x=n[i], color='k', linewidth=1, alpha=0.5)
        plt.text(n[i], 1.5, "♪", color='k', fontsize=8)
# asintotas horizontales para zmax y zmin
plt.axhline(y=zmax, color='k', linestyle="--" , linewidth=1, alpha=0.5)
plt.axhline(y=zmin, color='k', linestyle="--" , linewidth=1, alpha=0.5)
# Graficar distancia z como una linea continua
plt.plot(n, data.z, color='r', label='Distancia promedio al obstáculo en la ROI')
plt.plot(n, data.d, color='b', label='Distancia a la deteción más cercana a la ROI')
plt.xlabel("Tiempo (s)")
plt.ylabel("Distancia (m)")
plt.title("Distancia vs Tiempo")
plt.legend()
## Guardar la gráfica en un archivo .svg en alta resolución
#plt.savefig('exp1.svg', format='svg', dpi=1200)
## Guardar la gráfica en un archivo .png en alta resolución
#plt.savefig('exp1.png', format='png', dpi=1200)#
#plt.show()


# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(15, 5))
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.20, hspace=0.40) #
panel = fig.add_subplot(2, 1, 2)
plt.grid(color='gray', linestyle=':', alpha=.2) # Agregar grid minor con lineas, punteadas y color gris claro
# Graficar mensajes de la interfaz háptica como lineas verticales
for i in range(len(n)):
    if data.haptic_messages[i] == "l":
        # graficar una linea vertical azul delgada con un text" en la parte superior
        plt.axvline(x=n[i], color='b', linewidth=0.5)
        #plt.text(n[i], 0.5, "l", color='b', fontsize=8)
    if data.haptic_messages[i] == "r":
        # graficar una linea vertical roja delgada con un texto "r" en la parte superior
        plt.axvline(x=n[i], color='r', linewidth=0.5)
        #plt.text(n[i], 0.5, "r", color='r', fontsize=8)
    if data.haptic_messages[i] == "d":
        # graficar una linea vertical verde delgada con un texto "d" en la parte superior
        plt.axvline(x=n[i], color='g', linewidth=0.5)
        #plt.text(n[i], 0.5, "d", color='g', fontsize=8)
    if data.haptic_messages[i] == "u":
        # graficar una linea vertical amarilla delgada con un texto "u" en la parte superior
        plt.axvline(x=n[i], color='y', linewidth=0.5)
        #plt.text(n[i], 0.5, "u", color='y', fontsize=8)
    if data.haptic_messages[i] == "c":
        # graficar una linea vertical negra delgada con un texto "c" en la parte superior
        plt.axvline(x=n[i], color='k', linewidth=1)
        plt.text(n[i], 300, "♫", color='k', fontsize=8, verticalalignment='top')
plt.plot(n, data.h, label='Distancia horizontal del centro de la imagen a la detección más cercana')
plt.plot(n, data.v, label='Distancia vertical del centro de la imagen a la detección más cercana')
#plt.plot(n, data.d, color='b', label='Etiqueta de la detección más cercana')
#plt.plot(n, data.z, color='r', label='Distancia promedio al obstáculo en la ROI')
plt.xlabel('Tiempo [s]')
plt.ylabel('Distancia en píxeles normalizados [px/px]')
plt.legend()
#plt.savefig('graph2.svg', format='svg', dpi=1200)
plt.show()