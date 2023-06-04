import pandas as pd
import os
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np

# Set the font to Computer Modern 12pt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos del archivo .csv
data = pd.read_csv("../Activity logs.csv")

# Recortar las primeras crop1 muestras de los datos
crop1 = 1300
times = list(data.times[crop1:])
haptic_messages =  list(data.haptic_messages[crop1:])
buzzer_messages = list(data.buzzer_messages[crop1:])
d = list(data.d[crop1:])
v = list(data.v[crop1:])
h = list(data.h[crop1:])
z = list(data.z[crop1:])

N = len(times)
n = [i for i in range(N)] # lista de números enteros de 0 a N-1
bps = 10 # bits por segundo (envío de datos para sonar el buzzer por segundo)
zmax, zmin = 2, 1 # Distancia máxima y mínima para sonar el buzzer
DELTA = 30 # pixeles
width = height = 640 # Tamaño de la imagen
# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(16, 8))
# Ajustar el tamaño de la grafica al tamaño de la figura fig
deltaX, deltaY = 0.065, .09
offsetX, offsetY = 0.05, 0.03
ax = fig.add_axes([deltaX, deltaY + offsetY, 1 - 2 * deltaX, 1 - 2 * deltaY])
# asintuta horizontal en el valor DELTA
plt.grid(color='gray', linestyle=':', alpha=.2) # Agregar grid minor con lineas, punteadas y color gris claro
# Graficar mensajes de la interfaz háptica como lineas verticales
for i in range(N):
    #if haptic_messages[i] == "l":
    #    # graficar una linea vertical azul delgada con un text" en la parte superior
    #    plt.axvline(x=times[i], color='b', linewidth=0.5)
    #if haptic_messages[i] == "r":
    #    # graficar una linea vertical roja delgada con un texto "r" en la parte superior
    #    plt.axvline(x=times[i], color='r', linewidth=0.5)
    #if haptic_messages[i] == "d":
    #    # graficar una linea vertical verde delgada con un texto "d" en la parte superior
    #    plt.axvline(x=times[i], color='g', linewidth=0.5)
    #if haptic_messages[i] == "u":
    #    # graficar una linea vertical amarilla delgada con un texto "u" en la parte superior
    #    plt.axvline(x=times[i], color='y', linewidth=0.5)
    #if haptic_messages[i] == "c":
    #    # graficar una linea vertical negra delgada con un texto "c" en la parte superior
    #    plt.axvline(x=times[i], color='k', linewidth=2)
    #    plt.text(times[i], 1.5, r"$\twonotes$", color='k', fontsize=16, label=r'Informar detección $\twonotes$')
    if buzzer_messages[i] == 1:
        # graficar una linea vertical negra delgada con un texto "b" en la parte superior
        plt.axvline(x=times[i], color='k', linewidth=1.5)
        plt.text(times[i], 1.5, r"$\eighthnote$", color='k', fontsize=16, label=r'$\eighthnote$ Report detection')
#plt.plot(times, h, label=r'Distancia horizontal')
#plt.plot(times, v, label=r'Distancia vertical')

plt.plot(times, z, color='r', label=r'Average distance to the OAK-D sensor in the ROI')
plt.axhline(y=1, color='k', linestyle="--" , linewidth=1, alpha=0.5)
plt.axhline(y=2, color='k', linestyle="--" , linewidth=1, alpha=0.5, label=r'Obstacle proximity threshold')

#plt.plot(times, d, label=r'detected')
#plt.axhline(y=DELTA, color='k', linestyle="--" , linewidth=1, alpha=0.5, label=r'Umbral de detección')
plt.plot([], [], ' ',color='k', label=r'$\eighthnote$ ROI obstacle alert')
plt.title(r"Obstacle proximity alerts")
plt.xlabel(r'Time [s]')
plt.ylabel(r'Distance [m]')
plt.legend(loc='upper right')
plt.xlim(times[0], times[-1])
plt.savefig('Obstacle proximity alerts.svg', format='svg', dpi=1200)
plt.savefig('Obstacle proximity alerts.png', format='png', dpi=1200)
plt.savefig('Obstacle proximity alerts.pdf', format='pdf', dpi=1200)
plt.show()