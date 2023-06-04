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
    if buzzer_messages[i] == 1:
        # graficar una linea vertical negra delgada con un texto "b" en la parte superior
        plt.axvline(x=times[i], color='k', linewidth=1.5)
        plt.text(times[i], 1.5, r"$\eighthnote$", color='k', fontsize=16, label=r'Detección de informes $\eighthnote$')

plt.plot(times, z, color='r', label=r'Distancia medida al obstáculo de prueba')
plt.axhline(y=1, color='k', linestyle="--" , linewidth=1, alpha=0.5)
plt.axhline(y=2, color='k', linestyle="--" , linewidth=1, alpha=0.5, label=r'Umbral superior e inferior para alertas')

#plt.plot(times, d, label=r'detected')
#plt.axhline(y=DELTA, color='k', linestyle="--" , linewidth=1, alpha=0.5, label=r'Umbral de detección')
plt.plot([], [], ' ',color='k', label=r'Alerta de un obstáculo detectado $\eighthnote$')
#plt.title(r"Obstacle proximity alerts")
plt.xlabel(r'Tiempo [s]')
plt.ylabel(r'Distancia [m]')
plt.legend(loc='upper right')
plt.xlim(times[0], times[-1])
plt.savefig('Alertas de proximidad de obstáculos.svg', format='svg', dpi=1200, bbox_inches='tight')
plt.savefig('Alertas de proximidad de obstáculos.png', format='png', dpi=1200, bbox_inches='tight')
plt.savefig('Alertas de proximidad de obstáculos.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()