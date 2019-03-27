import cv2
import numpy as np 
import os
from skimage.color import lab2rgb
folder = 'sun6'

# Loading images from a folder into an array
def load_images_from_folder():
        images = []
        names = []
        for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder,fname))
                if img is not None:
                        images.append(img)
                        names.append(fname)
        return images,names

#Converting rgb image to lab and then back to rgb
def lab_rgb_method():
        org_images,org_names = load_images_from_folder()
        count = 0 
        for img in org_images:
                rgb_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab_rgb = cv2.cvtColor(rgb_lab, cv2.COLOR_LAB2RGB)
                cv2.imwrite('true_fake_images/'+org_names[count],lab_rgb)
                count = count + 1



lab_rgb_method()
