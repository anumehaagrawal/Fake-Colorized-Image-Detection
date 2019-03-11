import cv2
from matplotlib import pyplot as plt

def hue_equalized(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[200],[0,256])
    print(hist)
    plt.plot(hist)
    plt.show()

def saturated_equalized(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[1],None,[200],[0,256])
    plt.plot(hist)
    plt.show()
    

def load_images_from_folder(folder):
    images = []
    names = []
    for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fname))
            if img is not None:
                    images.append(img)
                    names.append(fname)
    return images,names

folder_fake = 'sun6-gthist'
fake_images = load_images_from_folder(folder_fake)
folder_real = 'sun6'
real_images = load_images_from_folder(fol)