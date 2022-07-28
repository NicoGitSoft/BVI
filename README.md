# Symbol-recognition-for-blind-people
This is a repository for spatial recognition of the environment surrounding a blind person through artificial intelligence.


rm -rf ~/BVI_env && python3 -m venv ~/BVI_env && source ~/BVI_env/bin/activate
pip install -r  requirements.txt


# Metodología para el desarrollo del proyecto (borrador)
Autor: *nicolas.ibanez.r@usach.cl*

## Contenidos
[ToC]

## Check list :memo:

- [x] Crear esta plataforma
- [x] Definir una nuevos elementos para Check list y/o listado de ideas
- [ ] NewCheck1
- [ ] NewCheck2

## Listado de ideas: 
1. Crear un script Python para extraer los fotogramas de un video de alta definición.
2. Crear un dataset para señalética del metro relevante para la orientación de las personas ciegas.
3. Incorporar tecnologías como un generador de voz de IA realista, reconocimiento y seguimieto de la mano en un script echo en python.
4. Incorporar el script creado en un dispositivo móvil ya sea un *smartphone* o un micro-controlador.
5. 

| idea | Links de documentación asociada    |
|:----:|:----------------------------------:|
|  1   |[Extracting and Saving Frames in Python][link1.1] |
|  2   |[Train Custom Data YOLOv5][link2.1], [Custom Dataset][Dataset-Drive]|
|  3   |[TTS Python][link3.1], [Box Tracking][link3.2], [EasyOCR][link3.3]  |
|  4   |[DeepAi Documentation][link4.1]                 |
|  5   |[Installing Mediapipe for Android][link5.1]|


[link1.1]:https://www.youtube.com/watch?v=SWGd2hX5p3U
[link2.1]:https://docs.ultralytics.com/tutorials/train-custom-datasets/
[link3.1]:https://pypi.org/project/TTS/
[link3.2]:https://docs.ultralytics.com/tutorials/train-custom-datasets/
[link3.3]:https://github.com/JaidedAI/EasyOCR
[link4.1]:https://docs.luxonis.com/en/latest/
[link5.1]:https://www.youtube.com/watch?v=R4HaRdEmoFU
[Dataset-Drive]:https://drive.google.com/drive/folders/1dSrF76v1vT8GWcZ8R5cJmPUMBZocqw7c

## Multimedia
![](https://i.imgur.com/ZmsaK1L.png)
![](https://i.imgur.com/LmQ0Pqr.jpg)
![](https://i.imgur.com/cxLTul3.jpg)


## Scripts & Google Colab

1. [Entrenamiento personalizado con YOLOv5](https://colab.research.google.com/drive/1JbUzKKAi8jrFSR6EB5u_6JhaWc-btLfa)
2. [Hand tracking & YOLOv6.ipynb](https://colab.research.google.com/drive/1J0qIBlP3KLpN8HH8-ohGbdEmjaVqeXM8)

<font style="color:purple" size="4">Requerimientos</font>
```
!pip install easyocr
!pip install mediapipe
# Clone and install MT-YOLOv6
!git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
!pip install -r requirements.txt
```


```
sudo apt install python3-pip
PATH=/home/cimtt4/.local/bin:$PATH


cd && rm -rf mediapipe/
git clone https://github.com/google/mediapipe.git && cd mediapipe

sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev
```