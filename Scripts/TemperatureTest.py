from Utilities import *
import max6675, subprocess, time, os, csv

# Modelo a utilizar
SingsYOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t_openvino_2021.4_6shave.blob")
SingsYOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t.json")

# Inicialización del dispositivo OAK-D
Device = DepthYoloHandTracker(use_depth=True, use_hand=True, use_mediapipe=True, temperature_sensing=True,
                              yolo_model=SingsYOLOv7t_MODEL, yolo_configurations=SingsYOLOv7t_CONFIG)

# Objetos y variables globales
CS, SCK, SO, UNIT = 22, 18, 16, 1
max6675.set_pin(CS, SCK, SO, UNIT)
vpu_Temperatures = []           # Muestras de la temperatura del chip
cpu_Temperatures = []           # Muestras de la temperatura del CPU
thermocouple_temperatures = []  # Muestras de la temperatura del sensor MAX6675
times = []                      # Muestras de tiempo
aux_time = 0                    # Tiempo auxiliar para almacenar cada segundo
start_time = time.time()        # Tiempo de inicio del programa

# Bucle principal
for i in range(3466):
    
    # Obtener la temperatura instantanea del chip de la OAK-D
    vpu_Temperature = Device.next_frame()[7]

    # Almacenar cada segundo las temperaturas medidas desde el chip de la OAK-D, la CPU de la Raspberry Pi y el sensor MAX6675
    stopwatch = time.time() - aux_time
    if stopwatch >= 1:
        # Almacenar tiempo
        times.append(time.time() - start_time)
        aux_time = time.time()

        # Almacenar temperaturas
        vpu_Temperatures.append(vpu_Temperature)
        cpu_Temperatures.append(float(subprocess.check_output("vcgencmd measure_temp", shell=True).decode("utf-8").replace("temp=","").replace("'C\n","")))
        thermocouple_temperatures.append(max6675.read_temp(SCK))

        # Mostrar teperaturas por consola las temperaturas con 2 decimales
        print("VPU: " + 
              str(round(vpu_Temperatures[-1], 2)) + "ºC, CPU: " +
              str(round(cpu_Temperatures[-1], 2)) + "ºC, MAX6675: " + 
              str(round(thermocouple_temperatures[-1], 2)) + "ºC") 

# Guardar las muestras en un archivo .csv usando writerows
with open('Temperatures1CPU.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Tiempo (s)", "VPU (ºC)", "CPU (ºC)", "MAX6675 (ºC)"])
    writer.writerows(zip(times, vpu_Temperatures, cpu_Temperatures, thermocouple_temperatures))