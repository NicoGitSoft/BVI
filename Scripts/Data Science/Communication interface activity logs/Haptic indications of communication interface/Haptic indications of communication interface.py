import pandas as pd
import os
from matplotlib import rc
import matplotlib.pylab as plt

# Set the font to Computer Modern 12pt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos del archivo .csv
data = pd.read_csv("../Activity logs.csv")


N = len(data) # Número de muestras en el archivo .csv
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
for i in range(len(n)):
    if data.haptic_messages[i] == "l":
        # graficar una linea vertical amarilla delgada
        plt.axvline(x=data.times[i], color='y', linewidth=0.5)
    if data.haptic_messages[i] == "r":
        # graficar una linea vertical roja delgada con un texto "r" en la parte superior
        plt.axvline(x=data.times[i], color='r', linewidth=0.5)
    if data.haptic_messages[i] == "d":
        # graficar una linea vertical verde delgada
        plt.axvline(x=data.times[i], color='g', linewidth=0.5)
    if data.haptic_messages[i] == "u":
        # graficar una linea vertical azul delgada
        plt.axvline(x=data.times[i], color='b', linewidth=0.5)
    if data.haptic_messages[i] == "c":
        # graficar una linea vertical negra delgada con un texto "c" en la parte superior
        plt.axvline(x=data.times[i], color='k', linewidth=2)
        plt.text(data.times[i], 150, r"$\twonotes$", color='k', fontsize=16, label=r'Report detection $\twonotes$')
plt.plot(data.times, data.h, label=r'Horizontal pixels')
plt.plot(data.times, data.v, label=r'Vertical pixels')
plt.axhline(y=DELTA, color='k', linestyle="--" , linewidth=1, alpha=0.5, label=r'Detection threshold $\delta$')
plt.text(data.times[100], DELTA , r"$\delta$", color='k', fontsize=26, label=r'Detection threshold $\delta$', horizontalalignment='left', verticalalignment='center')

plt.plot([], [], ' ',color='k', label=r'Report detection $\twonotes$')
#plt.title(r"Haptic indications of communication interface", size=28)
plt.xlabel(r'Time [s]')
plt.ylabel(r'Number of pixels [px]')
plt.xlim(data.times[100], data.times[1240])
plt.legend(loc='upper right')

# Ajustar los subplots a los bordes de la figura
plt.tight_layout()
#plt.subplots_adjust(top=0.947, bottom=0.109, left=0.04, right=0.995, hspace=0.2, wspace=0.103)

# Guardar la imagen en formato .svg, .png y .pdf
plt.savefig('Haptic indications of communication interface.svg', format='svg', dpi=1200, bbox_inches='tight')
plt.savefig('Haptic indications of communication interface.png', format='png', dpi=1200, bbox_inches='tight')
plt.savefig('Haptic indications of communication interface.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()