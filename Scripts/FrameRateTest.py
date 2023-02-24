from Utilities import *
from prettytable import PrettyTable
import time, os

def FrameRateTest(Device):

    # Inicialización de variables
    frames_counter = 0
    initial_timed_time = 0
    successful_detections = 0
    failed_detections = 0
    no_detections = 0
    fps = []

    # Inicialización del loop
    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = Device.next_frame()

        # Contador de detecciones
        detections_number = len(yolo_detections)
        if detections_number > 0:
            failed_detections += sum([1 for i in range(detections_number) if yolo_detections[i].label != 5])
            successful_detections += 1 if detections_number != failed_detections else 0
        else:
            no_detections += 1

        # Contador de FPS
        frames_counter += 1
        current_time = time.time()
        timed_time = time.time() - initial_timed_time
        if timed_time > 1:
            fps.append( frames_counter / timed_time )
            frames_counter = 0
            initial_timed_time = current_time
        
        # Mostrar resultados por pantalla
        print(f"FPS: {fps[-1]:.2f} - Detecciones: {successful_detections} - Falsos positivos: {failed_detections} - No detecciones: {no_detections}", end="\r")

    # Cerrar conexión con la cámara y mostrar resultados
    Device.exit()
    average_fps = sum(fps) / len(fps)

    return average_fps, successful_detections, failed_detections, no_detections

# RUTAS DE LOS MODELOS
SCRIPT_DIR = Path(__file__).resolve().parent
SingsYOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n_openvino_2021.4_6shave.blob")
SingsYOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s_openvino_2021.4_6shave.blob")
SingsYOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t_openvino_2021.4_6shave.blob")
SingsYOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n_openvino_2021.4_6shave.blob")
SingsYOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n.json")
SingsYOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t.json")
SingsYOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s.json")
SingsYOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n.json")
YOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8n/YOLOv8n_openvino_2021.4_6shave.blob")
YOLOv8s_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8s/YOLOv8s_openvino_2021.4_6shave.blob")
YOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7s/YOLOv7s_openvino_2021.4_6shave.blob")
YOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7t/YOLOv7t_openvino_2021.4_6shave.blob")
YOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv5n/YOLOv5n_openvino_2021.4_6shave.blob")
YOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8n/YOLOv8n.json")
YOLOv8s_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8s/YOLOv8s.json")
YOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7s/YOLOv7s.json")
YOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7t/YOLOv7t.json")
YOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv5n/YOLOv5n.json")

# Listas de los modelos y sus respectivas configuraciones
SingsYOLO_MODELS = [ SingsYOLOv8n_MODEL, SingsYOLOv7s_MODEL, SingsYOLOv7t_MODEL, SingsYOLOv5n_MODEL ]
SingsYOLO_CONFIGS = [SingsYOLOv8n_CONFIG, SingsYOLOv7s_CONFIG, SingsYOLOv7t_CONFIG, SingsYOLOv5n_CONFIG]

# Tabla de resultados
table = PrettyTable()
table.field_names = ["Yolo Model", "YoloDepth", "YoloDepth + HandTrackerVPU", "YoloDepth + HandTrackerCPU", "Susccessful detections", "Failed detections", "No detections"]

visualize = True
max_frames = 1000
##################### Ejecución de los modelos #####################
for i, model_path in enumerate(SingsYOLO_MODELS):

    config_path = SingsYOLO_CONFIGS[i]
    model_name = os.path.basename(model_path).split(".")[0].split("_")[0]
    
    # TEST 1: YOLO with spatial detection in the VPU.  
    DepthYolo = DepthYoloHandTracker(use_depth = True, use_hand = False, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo = FrameRateTest(DepthYolo)

    # TEST 2: YOLO with spatial detection and tracking of the finger of the user on the VPU.
    DepthYolo_HandTrackerVPU = DepthYoloHandTracker(use_depth = True, use_hand = True, use_mediapipe=False, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo_HandTrackerVPU = FrameRateTest(DepthYolo_HandTrackerVPU)

    # TEST 3: YOLO with spatial detection in the VPU, and tracking of the finger of the user in the CPU.
    DepthYolo_HandTrackerCPU = DepthYoloHandTracker(use_depth = True, use_hand = True, use_mediapipe=True, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo_HandTrackerCPU = FrameRateTest(DepthYolo_HandTrackerCPU)

    # Agregar fila de datos a la tabla
    table.add_row([
        model_name, "{:.2f} fps".format(data_DepthYolo[0]),
        "{:.2f} fps".format(data_DepthYolo_HandTrackerVPU[0]),
        "{:.2f} fps".format(data_DepthYolo_HandTrackerCPU[0]),
        "{:.2f} %".format( (data_DepthYolo[1]+data_DepthYolo_HandTrackerVPU[1]+data_DepthYolo_HandTrackerCPU[1])/(3*max_frames)*100 ),
        "{:.2f} %".format( (data_DepthYolo[2]+data_DepthYolo_HandTrackerVPU[2]+data_DepthYolo_HandTrackerCPU[2])/(3*max_frames)*100 ),
        "{:.2f} %".format( (data_DepthYolo[3]+data_DepthYolo_HandTrackerVPU[3]+data_DepthYolo_HandTrackerCPU[3])/(3*max_frames)*100 )
    ])

    print(table)


# RASPBERRY PI 4B 4GB

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 13.90 fps |          4.82 fps          |          7.77 fps          |        98.17 %         |       0.00 %      |     1.83 %    |
# |              | 13.84 fps |          4.59 fps          |          6.90 fps          |        97.87 %         |       0.00 %      |     2.13 %    |
# |              | 13.68 fps |          4.68 fps          |          7.12 fps          |        96.80 %         |       0.00 %      |     3.20 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7s |  1.19 fps |          0.72 fps          |          1.19 fps          |        98.10 %         |       0.00 %      |     1.90 %    |
# |              |  1.19 fps |          0.73 fps          |          1.19 fps          |        92.33 %         |       0.47 %      |     3.40 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7t | 13.89 fps |          5.50 fps          |          7.10 fps          |        18.67 %         |       0.00 %      |    81.33 %    |
# |              | 13.90 fps |          3.85 fps          |          7.06 fps          |        35.07 %         |       0.00 %      |    64.93 %    |
# |              | 13.89 fps |          4.48 fps          |          7.07 fps          |        74.20 %         |       0.00 %      |    25.80 %    |
# |              | 13.71 fps |          5.43 fps          |          7.99 fps          |        95.47 %         |       0.00 %      |     4.53 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv5n | 17.32 fps |          5.19 fps          |          7.02 fps          |        33.00 %         |       0.07 %      |     1.30 %    |
# |              | 16.99 fps |          5.50 fps          |          7.81 fps          |        98.60 %         |      40.60 %      |     1.20 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
