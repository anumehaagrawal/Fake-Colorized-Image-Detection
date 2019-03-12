import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

def hue_sat_calculation():
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

    return hue_fake,hue_true,sat_fake,sat_true

def calculating_v_values():
    
    hue_fake,hue_true,sat_fake,sat_true = hue_sat_calculation()
    vh_harray = []
    vh_sarray = []
    max_hue = -99999
    max_sat = -99999
    
    for img in range(len(hue_fake)):
        
        f1_h = -99999
        f1_s = -99999
        f2_h = 0
        f2_s = 0
        per_image_vec = []
        vh_h = -1
        vh_s = -1
        
        for val in range(len(hue_fake[img])):
            max_hue = max(max_hue,hue_fake[img][val])
            max_hue = max(max_hue,hue_true[img][val])
            max_sat = max(max_sat,sat_fake[img][val])
            max_sat = max(max_sat,sat_true[img][val])
            if(val + 1 != len(hue_fake[img])):
                f2_h = f2_h + abs(abs(hue_fake[img][val+1]-hue_true[img][val+1]) - abs(hue_fake[img][val]-hue_true[img][val]))
                f2_s = f2_s + abs(abs(sat_fake[img][val+1]-sat_true[img][val+1]) - abs(sat_fake[img][val]-sat_true[img][val]))

            if(abs(hue_fake[img][val]-hue_true[img][val]) > f1_h):
                f1_h = abs(hue_fake[img][val]-hue_true[img][val])
                vh_h = val

            if(abs(sat_fake[img][val]-sat_true[img][val]) > f1_s):
                f1_s = abs(sat_fake[img][val]-sat_true[img][val])
                vh_s = val
        vh_sarray.append(vh_s)
        vh_harray.append(vh_h)

    vh_map = max(map(lambda val: (vh_harray.count(val), val), set(vh_harray)))
    vh_h = vh_map[1]
    vhs_map = max(map(lambda val: (vh_sarray.count(val), val), set(vh_sarray)))
    vh_s = vhs_map[1]

    return vh_h,vh_s,max_hue,max_sat

def train_fcid_hist():
    vh_h , vh_s,max_hue,max_sat = calculating_v_values()
    hue_fake,hue_true,sat_fake,sat_true = hue_sat_calculation()
    training_set = []
    for img in range(len(hue_fake)):
        true_image = []
        fake_image = []
        #First for true images
        f1_h = hue_true[img][vh_h]
        f1_s = sat_true[img][vh_s]
        f2_h = 0
        f2_s = 0 
        for i in range(len(hue_true[img])):
            if(i+1 != len(hue_true[img])):
                f2_h = f2_h + abs(hue_true[img][i+1]-hue_true[img][i])
                f2_s = f2_s + abs(sat_true[img][i+1]-sat_true[img][i])
        true_image.append(f1_h[0]/max_hue)
        true_image.append(f1_s[0]/max_sat)
        true_image.append(f2_h[0]/max_hue)
        true_image.append(f2_s[0]/max_sat)
        true_image.append(1)

        #Second for fake images
        f1_h = hue_fake[img][vh_h]
        f1_s = sat_fake[img][vh_s]
        f2_h = 0
        f2_s = 0 
        for i in range(len(hue_fake[img])):
            if(i+1 != len(hue_fake[img])):
                f2_h = f2_h + abs(hue_fake[img][i+1]-hue_fake[img][i])
                f2_s = f2_s + abs(sat_fake[img][i+1]-sat_fake[img][i])
        fake_image.append(f1_h[0]/max_hue)
        fake_image.append(f1_s[0]/max_sat)
        fake_image.append(f2_h[0]/max_hue)
        fake_image.append(f2_s[0]/max_sat)
        fake_image.append(0)
        training_set.append(true_image)
        training_set.append(fake_image)
    
    return training_set

def training():
    train_array = train_fcid_hist()
    tf = pd.DataFrame(train_array)
    X = tf.iloc[0:400,0:4]
    Y = tf.iloc[0:400,4]
    X_test = tf.iloc[400:,0:4]
    Y_test = tf.iloc[400:,4]
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = SVC(gamma="auto")
    clf = GridSearchCV(svc, parameters, cv=5) 
    clf.fit(X,Y)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(clf.score(X_test,Y_test))
    

    

training()