import cv2
from matplotlib import pyplot as plt
import os
import pickle
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def parse_gmm_feature_vec(img_type):
    phi_df=[]
    pickle_off = open("phi_"+img_type+".pickle","rb")
    phi_arr = pickle.load(pickle_off)
    print(len(phi_arr),len(phi_arr[0]),len(phi_arr[0][0]))
    for feature_vec in phi_arr:
        df=pd.DataFrame(feature_vec)
        phi_df.append(df)
    return phi_df

#Histogram plot for hue channel
def hue_equalized(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[200],[0,256])
    plt.plot(hist)
    plt.show()

#Historgram plot for saturated channel
def saturated_equalized(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[1],None,[200],[0,256])
    plt.plot(hist)
    plt.show()

#Histogram values for hue channel
def hue_distribution(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[200],[0,256])
    return hist

#Saturated histogram values
def saturated_distribution(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[1],None,[200],[0,256])
    return hist

#Dark channel values for an image
def dark_channel(image):
    hist_r = cv2.calcHist([image],[0],None,[200],[0,256])
    hist_g = cv2.calcHist([image],[1],None,[200],[0,256])
    hist_b = cv2.calcHist([image],[2],None,[200],[0,256])
    hist_final = hist_r
    for i in range(200):
        hist_final[i] = min(min(hist_r[i],hist_g[i]),hist_b[i])
    return hist_final

#Light channel values for an image
def bright_channel(image):
    hist_r = cv2.calcHist([image],[0],None,[200],[0,256])
    hist_g = cv2.calcHist([image],[1],None,[200],[0,256])
    hist_b = cv2.calcHist([image],[2],None,[200],[0,256])
    hist_final = hist_r
    for i in range(200):
        hist_final[i] = max(max(hist_r[i],hist_g[i]),hist_b[i])
    
    return hist_final

#Loading images and their names into an array
def load_images_from_folder(folder):
    images = []
    names = []
    for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fname))
            if img is not None:
                    images.append(img)
                    names.append(fname)
    return images,names

def load_og_images():
    folder_false = 'ctest10k'
    folder_true = 'ctest110ktrue'
    images_true = []
    images_false = []
    names_true = []
    names_false = []
    
    fname_false = os.listdir(folder_false)
    fname_true = os.listdir(folder_true)
    count = 0 
    for fname in fname_false:
        if(count > 2500):
            break
        else:
            new_name = fname.split(".")[0]+".JPEG"
            if new_name in fname_true:
                count = count + 1
                imgt = cv2.imread(os.path.join(folder_true,new_name))
                imgf = cv2.imread(os.path.join(folder_false,fname))
                if imgt is not None:
                    images_true.append(imgt)
                    names_true.append(new_name)
                if imgf is not None:
                    images_false.append(imgf)
                    names_false.append(fname)

    return images_true,images_false,names_true,names_false

#Calculation of feature values for an image
def hue_sat_calculation():
    #No. of bins
    Kh = 200
    
    folder_fake = 'sun6-gthist'
    fake_images,fake_names = load_images_from_folder(folder_fake)
    folder_real = 'sun6'
    real_images ,real_names = load_images_from_folder(folder_real)
    
    # real_images,fake_images,real_names,fake_names = load_og_images()
    # real_images = real_images + real_images1
    # fake_images = fake_images + fake_images1
    # real_names = real_names + real_names1
    # fake_names = fake_names + fake_names1

    hue_fake = []
    hue_true = []
    sat_fake = []
    sat_true = []
    dc_fake = []
    dc_true = []
    bc_fake = []
    bc_true = []
    for img in range(len(fake_images)):
        hue_true.append(hue_distribution(real_images[img]))
        hue_fake.append(hue_distribution(fake_images[img]))
        sat_true.append(saturated_distribution(real_images[img]))
        sat_fake.append(saturated_distribution(fake_images[img]))
        dc_fake.append(dark_channel(fake_images[img]))
        dc_true.append(dark_channel(real_images[img]))
        bc_fake.append(bright_channel(fake_images[img]))
        bc_true.append(bright_channel(real_images[img]))

    return hue_fake,hue_true,sat_fake,sat_true,dc_fake,dc_true, bc_fake, bc_true

#calculation vh index values where divergence is the greatest
def calculating_v_values():
    
    hue_fake,hue_true,sat_fake,sat_true,dc_fake,dc_true,bc_fake, bc_true = hue_sat_calculation()
    vh_harray = []
    vh_sarray = []
    vdc_array = []
    vbc_array = []
    max_hue = -99999
    max_sat = -99999
    max_dc =  -99999
    max_bc = -99999
    
    for img in range(len(hue_fake)):
        
        f1_h = -99999
        f1_s = -99999
        f1_dc = -99999
        f1_bc = -99999
        per_image_vec = []
        vh_h = -1
        vh_s = -1
        vdc = -1
        vbc = -1
        
        for val in range(len(hue_fake[img])):
            max_hue = max(max_hue,hue_fake[img][val])
            max_hue = max(max_hue,hue_true[img][val])
            max_sat = max(max_sat,sat_fake[img][val])
            max_sat = max(max_sat,sat_true[img][val])
            max_dc = max(max_dc,dc_fake[img][val])
            max_dc = max(max_dc,dc_true[img][val])
            max_bc = max(max_bc,bc_fake[img][val])
            max_bc = max(max_bc,bc_true[img][val])

            if(abs(hue_fake[img][val]-hue_true[img][val]) > f1_h):
                f1_h = abs(hue_fake[img][val]-hue_true[img][val])
                vh_h = val

            if(abs(sat_fake[img][val]-sat_true[img][val]) > f1_s):
                f1_s = abs(sat_fake[img][val]-sat_true[img][val])
                vh_s = val

            if(abs(dc_fake[img][val]-dc_true[img][val]) > f1_dc):
                f1_dc = abs(dc_fake[img][val]-dc_true[img][val])
                vdc = val

            if(abs(bc_fake[img][val]-bc_true[img][val]) > f1_bc):
                f1_bc = abs(bc_fake[img][val]-bc_true[img][val])
                vbc = val
        vh_sarray.append(vh_s)
        vh_harray.append(vh_h)
        vdc_array.append(vdc)
        vbc_array.append(vbc)

    vh_map = max(map(lambda val: (vh_harray.count(val), val), set(vh_harray)))
    vh_h = vh_map[1]
    vhs_map = max(map(lambda val: (vh_sarray.count(val), val), set(vh_sarray)))
    vh_s = vhs_map[1]
    vdc_map = max(map(lambda val: (vdc_array.count(val), val), set(vdc_array)))
    vdc = vdc_map[1]
    vbc_map = max(map(lambda val: (vbc_array.count(val), val), set(vbc_array)))
    vbc = vbc_map[1]


    return vh_h,vh_s,vdc, vbc,max_hue,max_sat,max_dc,max_bc

#Implement FCID-HIST algorithm
def train_fcid_hist():
    vh_h , vh_s , vdc ,vbc, max_hue , max_sat, max_dc,max_bc = calculating_v_values()
    hue_fake,hue_true,sat_fake,sat_true, dc_fake, dc_true , bc_fake, bc_true = hue_sat_calculation()
    training_set = []
    for img in range(len(hue_fake)):
        true_image = []
        fake_image = []
        #First for true images
        f1_h = hue_true[img][vh_h]
        f1_s = sat_true[img][vh_s]
        f1_dc = dc_true[img][vdc]
        f1_bc = bc_true[img][vbc]
        f2_h = 0
        f2_s = 0 
        f2_dc = 0
        f2_bc = 0 
        for i in range(len(hue_true[img])):
            if(i+1 != len(hue_true[img])):
                f2_h = f2_h + abs(hue_true[img][i+1]-hue_true[img][i])
                f2_s = f2_s + abs(sat_true[img][i+1]-sat_true[img][i])
                f2_dc = f2_dc + abs(dc_true[img][i+1]-dc_true[img][i])
                f2_bc = f2_bc + abs(bc_true[img][i+1]-bc_true[img][i])
        true_image.append(f1_h[0]/max_hue)
        true_image.append(f1_s[0]/max_sat)
        true_image.append(f2_h[0]/max_hue)
        true_image.append(f2_s[0]/max_sat)
        true_image.append(f2_dc[0]/max_dc)
        true_image.append(f2_bc[0]/max_bc)
        true_image.append(1)

        #Second for fake images
        f1_h = hue_fake[img][vh_h]
        f1_s = sat_fake[img][vh_s]
        f1_dc = dc_fake[img][vdc]
        f1_bc = bc_fake[img][vbc]
        f2_h = 0
        f2_s = 0 
        f2_dc = 0
        f2_bc = 0
        for i in range(len(hue_fake[img])):
            if(i+1 != len(hue_fake[img])):
                f2_h = f2_h + abs(hue_fake[img][i+1]-hue_fake[img][i])
                f2_s = f2_s + abs(sat_fake[img][i+1]-sat_fake[img][i])
                f2_dc = f2_dc + abs(dc_fake[img][i+1]-dc_fake[img][i])
                f2_bc = f2_bc + abs(bc_fake[img][i+1]-bc_fake[img][i])
        fake_image.append(f1_h[0]/max_hue)
        fake_image.append(f1_s[0]/max_sat)
        fake_image.append(f2_h[0]/max_hue)
        fake_image.append(f2_s[0]/max_sat)
        fake_image.append(f2_dc[0]/max_dc)
        fake_image.append(f2_bc[0]/max_bc)
        fake_image.append(0)
        training_set.append(true_image)
        training_set.append(fake_image)
    
    return training_set

#Training of SVM model
def training():

    train_array = train_fcid_hist()
    tf = pd.DataFrame(train_array)
    print("Df dimension",tf.shape)

    tf.sample(frac=1)
    X = tf.iloc[0:380,0:6]
    Y = tf.iloc[0:380,6]
    X_test = tf.iloc[380:,0:6]
    Y_test = tf.iloc[380:,6]
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = SVC(gamma="auto")
    clf = GridSearchCV(svc, parameters, cv=5) 
    clf.fit(X,Y)
    y_pred = clf.predict(X_test)

    print("Predicted output is ",y_pred)
    print("ROC",roc_auc_score(y_pred, Y_test))
    print("Accuracy is ",clf.score(X_test,Y_test))
    

# Calling the training function
training()


# folder_fake = 'sun6-gthist'
# fake_images,fake_names = load_images_from_folder(folder_fake)
# folder_real = 'sun6'
# real_images ,real_names = load_images_from_folder(folder_real)
# training()
# hue_equalized(fake_images[2])
# padImages("sun6_padded",real_images)
# parse_gmm_feature_vec("true")