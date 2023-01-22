"""
    Extraer la temperatura del Chip del sensor OAK-D, la temperatura del procesador y la temperatura de la CPU, del sensor DHT22 y del sensor DHT11
    del archivo de texto Temperaturas.txt

    Estructura:
012345678901234567890123456789012345678901234567890123456789012345678901234567890
Chip temperature: 68.99 °C	CPU temperature: 78.40 °C	DHT22 temperature: 41.50 °C
Chip temperature: 68.99 °C	CPU temperature: 78.40 °C	DHT22 temperature: 41.50 °C
Chip temperature: 68.99 °C	CPU temperature: 78.40 °C	DHT22 temperature: 41.50 °C
Chip temperature: 68.99 °C	CPU temperature: 78.40 °C	DHT22 temperature: 41.50 °C

"""

import pandas as pd
import os
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np

# Set the font to Computer Modern 12pt
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
#rc('text', usetex=True)
#rc('legend', fontsize=20)
#plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Abrir el archivo de texto
archivo = open("Temperaturas.txt", "r")

# Leer el archivo de texto
linea = archivo.readline()

# Listas para almacenar las temperaturas
OAKD_ChipTemperture = []
CPU_Temperature = []
DHT22_Temperature = []

# Extraer las temperaturas
while linea != "":
    # Extraer la temperatura del Chip del sensor OAK-D
    OAKD_ChipTemperture.append(float(linea[18:22]))

    # Extraer la temperatura del procesador
    CPU_Temperature.append(float(linea[44:49]))
    
    # Extraer la temperatura del sensor DHT22
    DHT22_Temperature.append(float(linea[73:77]))


    # Leer la siguiente linea
    linea = archivo.readline()

    #print(
    #    "OAKD_ChipTemperture: ", OAKD_ChipTemperture,
    #    "CPU_Temperature: ", CPU_Temperature,
    #    "DHT22_Temperature: ", DHT22_Temperature,
    #    sep = "\t"#, end = "\r"
    #    )

# Leer los datos del archivo .csv
data = pd.read_csv("Temperature.csv")
# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(16, 8))
# Ajustar el tamaño de la grafica al tamaño de la figura fig
deltaX, deltaY = 0.065, .09
offsetX, offsetY = 0.05, 0.03
ax = fig.add_axes([deltaX, deltaY + offsetY, 1 - 2 * deltaX, 1 - 2 * deltaY])
plt.grid(color='gray', linestyle=':', alpha=.2) # Agregar grid minor con lineas, punteadas y color gris claro

# Graficar la temperatura del Chip del sensor OAK-D
plt.plot(OAKD_ChipTemperture, label="OAKD_ChipTemperture", color="red", linewidth=2)
# Graficar la temperatura del procesador
plt.plot(CPU_Temperature, label="CPU_Temperature", color="blue", linewidth=2)
# Graficar la temperatura del sensor DHT22
plt.plot(DHT22_Temperature, label="DHT22_Temperature", color="green", linewidth=2)
# Etiquetas del eje x
plt.xlabel("Tiempo (s)")
# Etiquetas del eje y
plt.ylabel("Temperatura (°C)")
# Título de la grafica
plt.title("Temperaturas")
# Mostrar leyenda
plt.legend()
# Mostrar la grafica
plt.show()
