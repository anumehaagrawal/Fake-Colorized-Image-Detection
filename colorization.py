import cv2
import numpy as np 
import os
from keras.models import model_from_json
from keras.models import load_model
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
                cv2.imwrite('true_fake_images/'+org_names[count],lab_rgb)
                count = count + 1

def create_model():
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
        model.save_weights('model_wieghts.h5')
        model.save('model_keras.h5')
        return model


def test():
        images,_  = load_images_from_folder()
        X,Y = create_dataset(images)
        model = create_model()
        model.load_weights('model_wieghts.h5')
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


lab_rgb_method()
