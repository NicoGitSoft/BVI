"""
Script python para renombrar un dataset de imagenes con la siguiente estructura:

test
├── images
│   ├── test_1.jpg
│   ├── test_2.jpg
│   ├── test_3.jpg
...     ...
├── labels
│   ├── test_1.txt
│   ├── test_2.txt
│   ├── test_3.txt
...     ...
train
├── images
│   ├── train_1.jpg
│   ├── train_2.jpg
│   ├── train_3.jpg
...     ...
├── labels
│   ├── train_1.txt
│   ├── train_2.txt
│   ├── train_3.txt
...     ...
valid
├── images
│   ├── valid_1.jpg
│   ├── valid_2.jpg
│   ├── valid_3.jpg
...     ...
├── labels
│   ├── valid_1.txt
│   ├── valid_2.txt
│   ├── valid_3.txt
...     ...


"""

# Importar librerias
import os

# Cambiar el directorio de trabajo en la ruta del script (donde se encuentra el dataset)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Lista de los directorios del dataset
folders = ['test', 'train', 'valid']

# Recorrer cada directorio del dataset
for folder in folders:
    # Ruta del directorio
    path = './' + folder + '/'
    # Ruta de la carpeta donde se encuentran las imagenes
    images_path = path + 'images/'
    # Ruta de la carpeta donde se encuentran las etiquetas
    labels_path = path + 'labels/'
    # Lista de rutas de los archivos de imagenes
    images_list = os.listdir(images_path)
    # Lista de rutas de los archivos de etiquetas
    labels_list = os.listdir(labels_path)
    # Recorrer cada imagen del dataset
    for i, name in enumerate(images_list):
        # Ruta absoluta de la imagen y de la etiqueta
        image_path = os.path.abspath(images_path + name)
        label_path = image_path.replace('images', 'labels').rsplit('.jpg', 1)[0] + '.txt'
        # Nuevo nombre de la imagen y de la etiqueta
        new_image_path = os.path.abspath(images_path + folder + '_' + str(i) + '.jpg')
        new_label_path = os.path.abspath(images_path + folder + '_' + str(i) + '.txt')
        # Renombrar la imagen y la etiqueta
        os.rename(image_path, new_image_path)
        os.rename(label_path, new_label_path)
        # Imprimir el nuevo nombre de la imagen y la etiqueta
        print(image_path, label_path)




        