import cv2
import numpy as np 
import os
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb
folder = 'n02113978'

def load_images_from_folder():
    images = []
    names = []
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,fname))
        if img is not None:
            images.append(img)
            names.append(fname)
    return images,names

def lab_rgb_method():
    org_images,org_names = load_images_from_folder()
    count = 0 
    for img in org_images:
        rgb_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_rgb = cv2.cvtColor(rgb_lab, cv2.COLOR_LAB2RGB)
        cv2.imwrite('fake_images/'+org_names[count],lab_rgb)
        count = count + 1

def create_dataset(images):
    X = []
    Y = []
    for img in images:
        img = cv2.resize(img,(200,200))
        rgb_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        #Fixing the ranges of L channel and a,b channels
        rgb_lab_norm  = (rgb_lab + [0, 128, 128]) / [100, 255, 255]
        # The input X is BW channel
        x = rgb_lab_norm[:,:,0]
        # The outpt y is  ab channels
        y = rgb_lab_norm[:,:,1:]
        x_n = x.reshape( x.shape[0], x.shape[1], 1)
        y_n = y.reshape( y.shape[0], y.shape[1], 2)
        X.append(x_n)
        Y.append(y_n)
    X = np.array(X)
    Y = np.array(Y) 
    return X,Y


def cnn_approach(images):
    X,Y = create_dataset(images)
    #Keras sequential model with conv layers
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

    adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adm)
    model.fit(x=X, y=Y, batch_size=1, epochs=1000, verbose=0)
    model.evaluate(X, Y, batch_size=1)
    model.save_weights('model_wieghts.h5')
    model.save('model_keras.h5')

        
    '''
    output = model.predict(X)
    cur = np.zeros((200, 200, 3))
    cur[:,:,0] = X[0][:,:,0]
    cur[:,:,1:] = output[0]
    cur = (cur * [100, 255, 255]) - [0, 128, 128]
    print(cur)
    rgb_image = lab2rgb(cur)
    cv2.imwrite('try.png',rgb_image)
    '''



images,_  = load_images_from_folder()
cnn_approach(images)