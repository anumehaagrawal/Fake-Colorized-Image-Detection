import cv2
import numpy as np 
import os

folder = 'n02113978'
def load_images_from_folder():
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            names.append(filename)
    return images,names

org_images,org_names = load_images_from_folder()
count = 0 
for img in org_images:
    rgb_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_rgb = cv2.cvtColor(rgb_lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite('fake_images/'+org_names[count],lab_rgb)
    count = count + 1


