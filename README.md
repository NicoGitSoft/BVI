# Assisted vision project for blind people (borrador)
Autor: *nicolas.ibanez.r@usach.cl*

This is a repository for spatial recognition of the environment surrounding a blind person through artificial intelligence.

<img src="https://s4.gifyu.com/images/BVI.gif" width="720" />

## Task: 
1. Crear un script Python para extraer los fotogramas de un video de alta definición.
2. Crear un dataset para señalética del metro relevante para la orientación de las personas ciegas.
3. Incorporar tecnologías como un generador de voz de IA realista, reconocimiento y seguimieto de la mano en un script echo en python.
4. Incorporar el script creado en un dispositivo móvil ya sea un *smartphone* o un micro-controlador.

| idea | Links to associated documentation  |
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

## Multi-media
<img src="https://i.imgur.com/ZmsaK1L.png" width="720" />
<img src="https://i.imgur.com/LmQ0Pqr.jpg" width="720" />
<img src="https://i.imgur.com/cxLTul3.jpg" width="720" />

## Scripts & Google Colab

1. [Entrenamiento personalizado con YOLOv5](https://colab.research.google.com/drive/1JbUzKKAi8jrFSR6EB5u_6JhaWc-btLfa)
2. [Hand tracking & YOLOv6.ipynb](https://colab.research.google.com/drive/1J0qIBlP3KLpN8HH8-ohGbdEmjaVqeXM8)

# <font style="color:purple" size="6">Installation requirements</font>

First installation

(Unix/macOS)
```
python3 -m pip install --user virtualenv
rm -rf ~/BVI_env
python3 -m venv ~/BVI_env && source ~/BVI_env/bin/activate
```

(Windows)
```
python.exe -m pip install --upgrade pip
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine CurrentUser
py -m pip install --user virtualenv
mkdir ~\Desktop\Proyect
cd ~\Desktop\Proyect
py -m venv env
.\\env\Scripts\activate
pip freeze
```

Select path interpreter ```~/BVI_env/bin/python``` (Unix/macOS) or ```~\env\Scripts\python.exe``` (Windows)

![](https://i.imgur.com/EiaIJVB.png)

<font style="color:blue" size="6">requirements.txt</font>

```
pip install mediapipe
pip install pyttsx3
pip install easyocr
```
or

```
cd ~/
pip install -r  requirements.txt
```

list of packages 
```
absl-py==1.1.0
attrs==21.4.0
backcall==0.2.0
certifi==2022.6.15
charset-normalizer==2.1.0
colorama==0.4.5
comtypes==1.1.13
cycler==0.11.0
debugpy==1.6.0
decorator==5.1.1
distlib==0.3.5
easyocr==1.5.0
entrypoints==0.4
filelock==3.7.1
fonttools==4.33.3
idna==3.3
imageio==2.21.1
importlib-metadata==4.12.0
ipykernel==6.15.0
ipython==7.34.0
jedi==0.18.1
jupyter-client==7.3.4
jupyter-core==4.10.0
kiwisolver==1.4.3
matplotlib==3.5.2
matplotlib-inline==0.1.3
mediapipe==0.8.10.1
nest-asyncio==1.5.5
networkx==2.6.3
numpy==1.21.6
opencv-contrib-python==4.6.0.66
opencv-python-headless==4.5.4.60
packaging==21.3
parso==0.8.3
pickleshare==0.7.5
Pillow==9.2.0
platformdirs==2.5.2
prompt-toolkit==3.0.30
protobuf==3.20.1
psutil==5.9.1
Pygments==2.12.0
pyparsing==3.0.9
pypiwin32==223
python-bidi==0.4.2
python-dateutil==2.8.2
pyttsx3==2.90
PyWavelets==1.3.0
pywin32==304
PyYAML==6.0
pyzmq==23.2.0
requests==2.28.1
ruptures==1.1.6
scikit-image==0.19.3
scipy==1.7.3
six==1.16.0
tifffile==2021.11.2
torch==1.12.1
torchvision==0.13.1
tornado==6.2
traitlets==5.3.0
typing_extensions==4.3.0
urllib3==1.26.11
virtualenv==20.16.2
wcwidth==0.2.5
zipp==3.8.1
```
