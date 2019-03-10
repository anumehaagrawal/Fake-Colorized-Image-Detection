import cv2
from matplotlib import pyplot as plt

def hue_equalized(img):
    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist= cv2.calcHist([hsv],[0],None,[200],[0,256])
    plt.plot(hist)
    plt.show()

def saturated_equalized(img):
    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist= cv2.calcHist([hsv],[1],None,[200],[0,256])
    plt.plot(hist)
    plt.show()
    

folder = 'n02113978'
image = cv2.imread('fake_images/n02113978_118.JPEG')
hue_equalized(image)