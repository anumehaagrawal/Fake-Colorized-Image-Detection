import pandas as pd
from sklearn.mixture import GaussianMixture
import cv2
from matplotlib import pyplot as plt
import os
import time
import pickle
import pandas as pd
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import random
# from array import data_array
import numpy as np

def get_gmm_dataFrame(img_type):
    phi_df=[]
    pickle_off = open("phi_"+img_type+".pickle","rb")
    phi_arr = pickle.load(pickle_off)
    print(len(phi_arr),len(phi_arr[0]),len(phi_arr[0][0]))
    for feature_vec in phi_arr:
        df=pd.DataFrame(feature_vec)
        phi_df.append(df)
    return phi_df

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

def gmm():
    fishers = []
    images, _ = load_images_from_folder("true_padded")
    true_arr = get_gmm_dataFrame("true")
    images, _ = load_images_from_folder("fake_padded")
    false_arr = get_gmm_dataFrame("fake")

    for index in range(len(images)):
        # df_i = pd.DataFrame(true_arr[index])
        df_i= true_arr[index]
        df_i = df_i.fillna(df_i.mean())
        df_i.sample(frac=1)
        df_tr, df_te = df_i[: -200], df_i[-200: ]

        GMM = GaussianMixture(n_components=4, covariance_type='diag').fit(df_tr) 
        
        N = df_te.shape[0]
        Q = GMM.predict_proba(df_te)
        # print(Q)
        gmm = GMM
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_te = np.dot(Q.T, df_te) / N
        Q_te_2 = np.dot(Q.T, df_te ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - gmm.weights_
        d_mu = Q_te - Q_sum * gmm.means_
        d_sigma = (
            - Q_te_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_te * gmm.means_)
        
        fisher_vector = np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
        fisher_vector = np.insert(fisher_vector,fisher_vector.shape[0],1,axis=0)
        # fisher_vector.append(1)
        fishers.append(fisher_vector)

    # df1 = pd.DataFrame(fishers)
    # df.insert(36,'Label',[1 for _ in range(240)],True)
    # df.to_pickle('true_fisher.pkl')

    f_fishers = []
    for index in range(len(images)):
        df_i = false_arr[index] #pd.DataFrame(false_arr[index])
        df_i = df_i.fillna(df_i.mean())
   
        df_i = df_i.sample(frac=1)
        df_tr, df_te = df_i[: -200], df_i[-200: ]
        
        GMM = GaussianMixture(n_components=4, covariance_type='diag').fit(df_tr) 
        
        N = df_te.shape[0]
        Q = GMM.predict_proba(df_te)
        # print(Q)
        gmm = GMM
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_te = np.dot(Q.T, df_te) / N
        Q_te_2 = np.dot(Q.T, df_te ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - gmm.weights_
        d_mu = Q_te - Q_sum * gmm.means_
        d_sigma = (
            - Q_te_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_te * gmm.means_)
        
        fisher_vector = np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
        fisher_vector = np.insert(fisher_vector,fisher_vector.shape[0],0,axis=0)

        # print(fisher_vector)
        fishers.append(fisher_vector)

    random.shuffle(fishers)
    df = pd.DataFrame(fishers)
    # df.insert(36,'Label',[0 for _ in range(240)],True)
    df.to_pickle('fisher_vec.pkl')

def get_svm_frames():
    # true_fisher_p = open("true_fisher.pkl","rb")
    fisher_p=open("fisher_vec.pkl","rb")
    df = pickle.load(fisher_p)
    # fake_df = pickle.load(fake_fisher_p)
    # comb_df= pd.concat([true_df,fake_df])
    # print(comb_df.iloc[:,36:])
    return df
    

def training():

    tf = get_svm_frames()
    tf.sample(frac=1)
    print("Df dimension",tf.shape)
    print(tf.iloc[:,36:])
    X = tf.iloc[0:400,0:36]
    X=preprocessing.scale(X)
    Y = tf.iloc[0:400,36]
    X_test = tf.iloc[400:,0:36]
    X_test=preprocessing.scale(X_test)
    Y_test = tf.iloc[400:,36]
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = SVC(gamma="auto",verbose=False)
    clf = GridSearchCV(svc, parameters, cv=5) 
    clf.fit(X,Y)
    y_pred = clf.predict(X_test)

    print("Predicted output is ",y_pred)
    print("ROC",roc_auc_score(y_pred, Y_test))
    print("Accuracy is ",clf.score(X_test,Y_test))

# gmm()
training()

#run1 term1 180 30 
# ROC 0.7954545454545454
# Accuracy is  0.7

#run2 220 30 86% 0.798 ROC

#run3 400 50 


