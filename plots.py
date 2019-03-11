import cv2
from matplotlib import pyplot as plt
import os

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

def hue_distribution(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[200],[0,256])
    return hist

def saturated_distribution(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[1],None,[200],[0,256])
    return hist

def load_images_from_folder(folder):
    images = []
    names = []
    for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fname))
            if img is not None:
                    images.append(img)
                    names.append(fname)
    return images,names


def fcid_hist():
    #No. of bins
    Kh = 200
    folder_fake = 'sun6-gthist'
    fake_images,fake_names = load_images_from_folder(folder_fake)
    folder_real = 'sun6'
    real_images ,real_names = load_images_from_folder(folder_real)
    hue_fake = []
    hue_true = []
    sat_fake = []
    sat_true = []
    for img in range(len(fake_images)):
        hue_true.append(hue_distribution(real_images[img]))
        hue_fake.append(hue_distribution(fake_images[img]))
        sat_true.append(saturated_distribution(real_images[img]))
        sat_fake.append(saturated_distribution(fake_images[img]))

    training_vec = []
    for img in range(len(fake_images)):
        vh_h = -1
        vh_s = -1
        f1_h = -99999
        f1_s = -99999
        per_image_vec = []
        for val in range(len(hue_fake[img])):
            if(abs(hue_fake[img][val]-hue_true[img][val]) > f1_h):
                f1_h = abs(hue_fake[img][val]-hue_true[img][val])
                vh_h = val

            if(abs(sat_fake[img][val]-sat_true[img][val]) > f1_s):
                f1_s = abs(sat_fake[img][val]-sat_true[img][val])
                vh_s = val

        per_image_vec.append(vh_s)
        per_image_vec.append(vh_h)
        training_vec.append(per_image_vec)

    print(training_vec)


    

fcid_hist()

    

