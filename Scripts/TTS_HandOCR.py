import easyocr, cv2
import PIL
from PIL import ImageDraw
im = PIL.Image.open("thai.png")
im

reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)


img = cv2.imread('chinese_tra.jpg')
result = reader.readtext(img)