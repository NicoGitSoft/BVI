import easyocr
reader = easyocr.Reader(['es', 'en'])

result = reader.readtext('./Datasets/Best_images/PXL_20220726_134600828.jpg')