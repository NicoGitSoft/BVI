import os, subprocess, serial

SketchPath = './Atmega328P'
MainDir = os.path.dirname(os.path.abspath(__file__))
ArduinoSketchDir = os.path.join(MainDir, '.', SketchPath)
os.chdir(ArduinoSketchDir)

try:
    InfoBoard = subprocess.getoutput('arduino-cli board list').split()
    PuertoArduino  = InfoBoard[9] # Obtener el puerto de la placa Arduino
    FQBN  = InfoBoard[-2] # Obtener el FQBN de la placa Arduino
    print(InfoBoard, PuertoArduino, FQBN)
    os.system("arduino-cli compile --fqbn " + FQBN)
    os.system("arduino-cli upload -p " + PuertoArduino +  " --fqbn " + FQBN)
    ArduinoSerial = serial.Serial(PuertoArduino, 9600, timeout=1) # open the serial port
    print("El sketch se compiló y cargó correctamente", ArduinoSketchDir)
except Exception as e:
    print("No se estableció comunicación serial con una placa Arduino correctamente")