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
data = pd.read_csv("Temperature.csv")
# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(16, 8))
# Ajustar el tamaño de la grafica al tamaño de la figura fig
deltaX, deltaY = 0.065, .09
offsetX, offsetY = 0.05, 0.03
ax = fig.add_axes([deltaX, deltaY + offsetY, 1 - 2 * deltaX, 1 - 2 * deltaY])
plt.grid(color='gray', linestyle=':', alpha=.2) # Agregar grid minor con lineas, punteadas y color gris claro

# define my model fitting function
def func(x, A, tau, T0): # función
    return T0 * (1 - A * np.exp(-tau * x))

# define my data
xdata = data.times
ydata = data.chipTemperature
# define my initial guess for the parameters
p0 = [-1, 0.03, 65]
# fit the data
from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, xdata, ydata, p0)
# plot the data and the fitted curve
ax.plot(xdata[20:], ydata[20:], '.', label='Temperature data', markersize=1)
ax.plot(xdata[20:], func(xdata[20:], *popt), 'g-', label=r'$T_0(1-Ae^{-\tau t})$')
plt.title(r"OAK-D chip temperature while the program is running")
plt.xlabel(r'Time [s]')
plt.ylabel(r'Temperature [°C]')
plt.legend(loc='upper left')
plt.savefig('Tempetature.svg', format='svg', dpi=1200)
plt.savefig('Tempetature.png', format='png', dpi=1200)
plt.savefig('Tempetature.pdf', format='pdf', dpi=1200)
plt.show()