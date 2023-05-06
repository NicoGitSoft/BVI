"""
Script para graficar los datos dos archivos llamados:

file1: "system temperatures without distributed processing.csv"
file2: "system temperatures with distributed processing.csv"

los cuales contienen los datos de temperatura de la VPU del sensor OAK-D, 
la temperatura de la CPU de la Raspberry Pi 4 y la temperatura del sensor MAX6675.
Los datos se encuentran en archivos .csv con el siguiente formato:

TIMES,VPU,CPU,MAX6675
t1,vpu1,cpu1,max6675_1
t2,vpu2,cpu2,max6675_2
...
tn,vpun,cpun,max6675_n

Las gráficas usan formato LaTeX para los ejes y leyendas, por otro lado,
se grafican dos subplots para comparar los datos de las temperaturas de lo los dos archivos
"""

import pandas as pd
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Ejecutar el script en la ruta actual donde se encuentran los datos
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos
file1 = pd.read_csv('system temperatures without distributed processing.csv')
file2 = pd.read_csv('system temperatures with distributed processing.csv')

# Obtener los datos de las temperaturas
start_sample_file_1 = 0
start_sample_file_2 = 100

# vector de nuestras
n1 = np.arange(0, len(file1['TIMES'][start_sample_file_1:-start_sample_file_2]))
n2 = np.arange(0, len(file2['TIMES'][start_sample_file_2:]))

DELTA_VPU_MAX6675 = 12.028342016016863 # calibración de la termocupla

# datos del archivo 1 
VPU_TEMPERATURE_1 = file1['VPU'][start_sample_file_1:-start_sample_file_2]
CPU_TEMPERATURE_1 = file1['CPU'][start_sample_file_1:-start_sample_file_2]
MAX6675_TEMPERATURE_1 = file1['THERMOCOUPLE'][start_sample_file_1:-start_sample_file_2] - DELTA_VPU_MAX6675

# datos del archivo 2
VPU_TEMPERATURE_2 = file2['VPU'][start_sample_file_2:] 
CPU_TEMPERATURE_2 = file2['CPU'][start_sample_file_2:]
MAX6675_TEMPERATURE_2 = file2['THERMOCOUPLE'][start_sample_file_2:] 

######################### AJUSTE DE CURVAS #########################

# Función para ajustes de curva exponencial
def func(x, T_0, T_inf, tau):
    return (T_inf - T_0) * (1 - np.exp(-x/tau)) + T_0

# Puntos iniciales para ajuste de curva exponencial
p0_VPU_TEMPERATURE = [37, 57, 500]
p0_CPU_TEMPERATURE = [59.5, 66.6604982, 450]
p0_MAX6675_TEMPERATURE = [20, 27, 500]


# Ajuste de curva exponencial para los datos del archivo 1
popt_VPU_TEMPERATURE_1, pcov_VPU_TEMPERATURE_1 = curve_fit(func, n1, VPU_TEMPERATURE_1, p0=p0_VPU_TEMPERATURE)
popt_CPU_TEMPERATURE_1, pcov_CPU_TEMPERATURE_1 = curve_fit(func, n1, CPU_TEMPERATURE_1, p0=p0_CPU_TEMPERATURE)
popt_MAX6675_TEMPERATURE_1, pcov_MAX6675_TEMPERATURE_1 = curve_fit(func, n1, MAX6675_TEMPERATURE_1, p0=p0_MAX6675_TEMPERATURE)

# Ajuste de curva exponencial para los datos del archivo 2
popt_VPU_TEMPERATURE_2, pcov_VPU_TEMPERATURE_2 = curve_fit(func, n2, VPU_TEMPERATURE_2, p0=p0_VPU_TEMPERATURE)
popt_CPU_TEMPERATURE_2, pcov_CPU_TEMPERATURE_2 = curve_fit(func, n2, CPU_TEMPERATURE_2, p0=p0_CPU_TEMPERATURE)
popt_MAX6675_TEMPERATURE_2, pcov_MAX6675_TEMPERATURE_2 = curve_fit(func, n2, MAX6675_TEMPERATURE_2, p0=p0_MAX6675_TEMPERATURE)

########################## ANÁLISIS ##########################

# Mostrar por consola los parámetros de ajuste pcov
print('Archivo 1\n VPU: ', popt_VPU_TEMPERATURE_1, '\n CPU: ', popt_CPU_TEMPERATURE_1, '\n MAX6675: ', popt_MAX6675_TEMPERATURE_1)
print('Archivo 2\n VPU: ', popt_VPU_TEMPERATURE_2, '\n CPU: ', popt_CPU_TEMPERATURE_2, '\n MAX6675: ', popt_MAX6675_TEMPERATURE_2)

# Calcular desviación estándar de los parámetros de la termocupla MAX6675
s_MAX6675_TEMPERATURE_1 = np.sqrt( sum((MAX6675_TEMPERATURE_1 - func(n1, *popt_MAX6675_TEMPERATURE_1))**2) / (len(n1) - 3) )
s_MAX6675_TEMPERATURE_2 = np.sqrt( sum((MAX6675_TEMPERATURE_2 - func(n2, *popt_MAX6675_TEMPERATURE_2))**2) / (len(n2) - 3) )

# Mostrar por consola la desviación estándar de los parámetros de la termocupla MAX6675
print('Desviación estándar de los parámetros de la termocupla MAX6675\nArchivo 1\n', s_MAX6675_TEMPERATURE_1, '\nArchivo 2\n', s_MAX6675_TEMPERATURE_2)

# Coeficiente de determinación
r2_VPU_TEMPERATURE_1 = 1 - sum((VPU_TEMPERATURE_1 - func(n1, *popt_VPU_TEMPERATURE_1))**2) / sum((VPU_TEMPERATURE_1 - np.mean(VPU_TEMPERATURE_1))**2)
r2_CPU_TEMPERATURE_1 = 1 - sum((CPU_TEMPERATURE_1 - func(n1, *popt_CPU_TEMPERATURE_1))**2) / sum((CPU_TEMPERATURE_1 - np.mean(CPU_TEMPERATURE_1))**2)
r2_MAX6675_TEMPERATURE_1 = 1 - sum((MAX6675_TEMPERATURE_1 - func(n1, *popt_MAX6675_TEMPERATURE_1))**2) / sum((MAX6675_TEMPERATURE_1 - np.mean(MAX6675_TEMPERATURE_1))**2)

r2_VPU_TEMPERATURE_2 = 1 - sum((VPU_TEMPERATURE_2 - func(n2, *popt_VPU_TEMPERATURE_2))**2) / sum((VPU_TEMPERATURE_2 - np.mean(VPU_TEMPERATURE_2))**2)
r2_CPU_TEMPERATURE_2 = 1 - sum((CPU_TEMPERATURE_2 - func(n2, *popt_CPU_TEMPERATURE_2))**2) / sum((CPU_TEMPERATURE_2 - np.mean(CPU_TEMPERATURE_2))**2)
r2_MAX6675_TEMPERATURE_2 = 1 - sum((MAX6675_TEMPERATURE_2 - func(n2, *popt_MAX6675_TEMPERATURE_2))**2) / sum((MAX6675_TEMPERATURE_2 - np.mean(MAX6675_TEMPERATURE_2))**2)

# Mostrar por consola el coeficiente de determinación
print('Coeficiente de determinación\nArchivo 1\n VPU: ', r2_VPU_TEMPERATURE_1, '\n CPU: ', r2_CPU_TEMPERATURE_1, '\n MAX6675: ', r2_MAX6675_TEMPERATURE_1)
print('Archivo 2\n VPU: ', r2_VPU_TEMPERATURE_2, '\n CPU: ', r2_CPU_TEMPERATURE_2, '\n MAX6675: ', r2_MAX6675_TEMPERATURE_2)

# Error absoluto medio
MAE_VPU_TEMPERATURE_1 = sum(abs(VPU_TEMPERATURE_1 - func(n1, *popt_VPU_TEMPERATURE_1))) / len(n1)
MAE_CPU_TEMPERATURE_1 = sum(abs(CPU_TEMPERATURE_1 - func(n1, *popt_CPU_TEMPERATURE_1))) / len(n1)
MAE_MAX6675_TEMPERATURE_1 = sum(abs(MAX6675_TEMPERATURE_1 - func(n1, *popt_MAX6675_TEMPERATURE_1))) / len(n1)

MAE_VPU_TEMPERATURE_2 = sum(abs(VPU_TEMPERATURE_2 - func(n2, *popt_VPU_TEMPERATURE_2))) / len(n2)
MAE_CPU_TEMPERATURE_2 = sum(abs(CPU_TEMPERATURE_2 - func(n2, *popt_CPU_TEMPERATURE_2))) / len(n2)
MAE_MAX6675_TEMPERATURE_2 = sum(abs(MAX6675_TEMPERATURE_2 - func(n2, *popt_MAX6675_TEMPERATURE_2))) / len(n2)

# Mostrar por consola el error absoluto medio
print('Error absoluto medio\nArchivo 1\n VPU: ', MAE_VPU_TEMPERATURE_1, '\n CPU: ', MAE_CPU_TEMPERATURE_1, '\n MAX6675: ', MAE_MAX6675_TEMPERATURE_1)
print('Archivo 2\n VPU: ', MAE_VPU_TEMPERATURE_2, '\n CPU: ', MAE_CPU_TEMPERATURE_2, '\n MAX6675: ', MAE_MAX6675_TEMPERATURE_2)

# valor de tau
tau_VPU_TEMPERATURE_1 = popt_VPU_TEMPERATURE_1[2]
tau_CPU_TEMPERATURE_1 = popt_CPU_TEMPERATURE_1[2]
tau_MAX6675_TEMPERATURE_1 = popt_MAX6675_TEMPERATURE_1[2]

tau_VPU_TEMPERATURE_2 = popt_VPU_TEMPERATURE_2[2]
tau_CPU_TEMPERATURE_2 = popt_CPU_TEMPERATURE_2[2]
tau_MAX6675_TEMPERATURE_2 = popt_MAX6675_TEMPERATURE_2[2]

# Mostrar por consola el valor de tau
print('Valor de tau\nArchivo 1\n VPU: ', tau_VPU_TEMPERATURE_1, '\n CPU: ', tau_CPU_TEMPERATURE_1, '\n MAX6675: ', tau_MAX6675_TEMPERATURE_1)
print('Archivo 2\n VPU: ', tau_VPU_TEMPERATURE_2, '\n CPU: ', tau_CPU_TEMPERATURE_2, '\n MAX6675: ', tau_MAX6675_TEMPERATURE_2)

########################## GRAFICAS ##########################
# margenes de los ejes
y_delta = 2
y_offset = 1
y_offset_extra = 0.25
y_min = min(popt_MAX6675_TEMPERATURE_1[0], popt_MAX6675_TEMPERATURE_2[0]) + y_offset
y_max = max(popt_CPU_TEMPERATURE_1[1], popt_CPU_TEMPERATURE_2[1], popt_VPU_TEMPERATURE_1[1], popt_VPU_TEMPERATURE_2[1]) + y_offset

# Configuración de las graficas
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
rc('text', usetex=True)
rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Crear una figura para los subplots
figure_1 = plt.figure(figsize=(14, 6))


# Crear el primer subplot
ax1 = figure_1.add_subplot(121)
# Agrergar grid principal con lineas, punteadas y color gris claro (color='gray', linestyle=':', alpha=.2)
ax1.grid(color='gray', linestyle=':', alpha=.2)
# Agregar datos del archivo 1
ax1.plot(n1, CPU_TEMPERATURE_1, 'b.', markersize=2, label=r'CPU')
ax1.plot(n1, VPU_TEMPERATURE_1, 'r.', markersize=2, label=r'VPU')
ax1.plot(n1, MAX6675_TEMPERATURE_1, 'g.', markersize=2, label=r'Disipador de calor')
# Graficar los ajustes de curva exponencial para los datos del archivo 1
ax1.plot(n1, func(n1, *popt_VPU_TEMPERATURE_1), 'k--', label="_nolegend_")
ax1.plot(n1, func(n1, *popt_CPU_TEMPERATURE_1), 'k--', label="_nolegend_")
ax1.plot(n1, func(n1, *popt_MAX6675_TEMPERATURE_1), 'k--', label="_nolegend_")
# Asintotas horizontales en el valor de T_inf para cada curva
ax1.axhline(y=popt_VPU_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax1.axhline(y=popt_MAX6675_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax1.axhline(y=popt_CPU_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
# texto de los valores de T_inf
yText_CPU_TEMPERATURE_1 = (popt_CPU_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta) - 0.008
yText_VPU_TEMPERATURE_1 = (popt_VPU_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta) - 0.008
yText_MAX6675_TEMPERATURE_1 = (popt_MAX6675_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta) - 0.008
ax1.text(1-0.008, yText_CPU_TEMPERATURE_1, str(round(popt_CPU_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14)
ax1.text(1-0.008, yText_VPU_TEMPERATURE_1, str(round(popt_VPU_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14)	
ax1.text(1-0.008, yText_MAX6675_TEMPERATURE_1, str(round(popt_MAX6675_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14)
# limites de los ejes
ax1.set_ylim(y_min-y_delta+y_offset_extra, y_max+y_delta+y_offset_extra)
ax1.set_xlim(0, 3999)
# Etiquetas 
ax1.set_ylabel(r'Temperatura ($^{\circ}$C)')
ax1.set_xlabel(r'Tiempo (s)')
ax1.set_title(r'Temperaturas sin procesamiento distribuido')
# Leyenda desplazada un poco hacia arriva
ax1.legend(loc='center right', bbox_to_anchor=(1, 0.6))
# Crear el segundo subplot
ax2 = figure_1.add_subplot(122)
# Agrergar grid principal con lineas, punteadas y color gris claro (color='gray', linestyle=':', alpha=.2)
ax2.grid(color='gray', linestyle=':', alpha=.2)
# Agregar datos del archivo 2
ax2.plot(n2, VPU_TEMPERATURE_2, 'r.', markersize=2, label=r'VPU')
ax2.plot(n2, CPU_TEMPERATURE_2, 'b.', markersize=2, label=r'CPU')
ax2.plot(n2, MAX6675_TEMPERATURE_2, 'g.', markersize=2, label=r'Disipador de calor')
# Graficar los ajustes de curva exponencial para los datos del archivo 2
ax2.plot(n2, func(n2, *popt_VPU_TEMPERATURE_2), 'k--', label="_nolegend_")
ax2.plot(n2, func(n2, *popt_CPU_TEMPERATURE_2), 'k--', label="_nolegend_")
ax2.plot(n2, func(n2, *popt_MAX6675_TEMPERATURE_2), 'k--', label="_nolegend_")
# Asintotas horizontales en el valor de T_inf para cada curva
ax2.axhline(y=popt_VPU_TEMPERATURE_2[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax2.axhline(y=popt_MAX6675_TEMPERATURE_2[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax2.axhline(y=popt_CPU_TEMPERATURE_2[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
# texto de los valores de T_inf
yText_CPU_TEMPERATURE_2 = (popt_CPU_TEMPERATURE_2[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)-0.008
yText_VPU_TEMPERATURE_2 = (popt_VPU_TEMPERATURE_2[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)-0.008
yText_MAX6675_TEMPERATURE_2 = (popt_MAX6675_TEMPERATURE_2[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)-0.008
ax2.text(1-0.008, yText_CPU_TEMPERATURE_2, str(round(popt_CPU_TEMPERATURE_2[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, fontsize=14)
ax2.text(1-0.008, yText_VPU_TEMPERATURE_2, str(round(popt_VPU_TEMPERATURE_2[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, fontsize=14)
ax2.text(1-0.008, yText_MAX6675_TEMPERATURE_2, str(round(popt_MAX6675_TEMPERATURE_2[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, fontsize=14)
# limites de los ejes
ax2.set_ylim(y_min-y_delta+y_offset_extra, y_max+y_delta+y_offset_extra)
ax2.set_xlim(0, 3900)
# Etiquetas
ax2.set_ylabel(r'Temperatura ($^{\circ}$C)')
ax2.set_xlabel(r'Tiempo (s)')
ax2.set_title(r'Temperaturas con procesamiento distribuido')
# Leyenda desplazada un poco hacia arriva
ax2.legend(loc="center right", bbox_to_anchor=(1, 0.6))

# Ajustar los subplots a los bordes de la figura
plt.tight_layout()
plt.subplots_adjust(top=0.945, bottom=0.11, left=0.05, right=0.995, hspace=0.2, wspace=0.15)

# Guardar la figura en pdf
plt.savefig('Temperaturas del sistema.pdf', format='pdf', dpi=1200)

# Mostrar las graficas
plt.show()