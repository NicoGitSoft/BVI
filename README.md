# BVI System 
![https://github.com/WongKinYiu/yolov7](https://img.shields.io/badge/YOLO-v7-green.svg) ![https://docs.luxonis.com/en/latest/](https://img.shields.io/badge/DepthAI-latest-blue.svg) ![zenodo 7479656](https://user-images.githubusercontent.com/65929186/209434816-94709752-ba98-4813-95d7-fbfb8a8cb6a6.svg)
##### Complementary repository to the article: "Development Of An Assisted Vision System For Blind People Based On Hand Movement And Artificial Intelligence"

## Description
This repocitorio aims to present the files that were used for the development of the thesis "Development of a vision system for the blind based on Artificial Intelligence" of the University of Santiago, Chile (Usach).

###### Author: Nicolás Ibáñez Rojas
###### E-mail: nicolas.ibanez.r@usach.cl

## Illustrations
![Resolution](https://user-images.githubusercontent.com/65929186/209304016-66afaf8e-b362-4e75-98e6-e8511540d4c3.svg)

<img src="Media/VideoRGB_Hand.gif" alt="drawing" width="200"/> <img src="Media/6.jpg" alt="drawing" width="200"/> <img src="Media/11.jpg" alt="drawing" width="200"/> <img src="Media/7.jpg" alt="drawing" width="200"/>

<img src="Media/12.jpg" alt="drawing" width="200"/> <img src="Media/2.jpg" alt="drawing" width="200"/> <img src="Media/3.jpg" alt="drawing" width="200"/> <img src="Media/4.jpg" alt="drawing" width="200"/>

<img src="Media/1.jpg" alt="drawing" width="200"/> <img src="Media/5.jpg" alt="drawing" width="200"/> <img src="Media/8.jpg" alt="drawing" width="200"/> <img src="Media/10.jpg" alt="drawing" width="200"/>

#### System diagram
![Generic-system-diagram](https://user-images.githubusercontent.com/65929186/206758452-ac6fd6a2-e0e3-484a-bc02-a80635da9536.svg)

#### System circuit
![BVIcircuit](https://user-images.githubusercontent.com/65929186/214612848-312c50b7-6fa4-4afb-aaaa-f5df1303d0f3.svg)


#### BVI-DATASET description

BVI-DATASET contains some of the most frequent signage in airports, subways and shopping malls. Available at ROBOFLOW (link [here](https://app.roboflow.com/generic-signage/airports-ans-subways/7))


#### Requirements
```
sudo apt install speech-dispatcher pyserial scipy curl
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai
```
