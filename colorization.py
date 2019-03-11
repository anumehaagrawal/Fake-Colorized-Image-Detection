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
import autocolorize


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
        '''
        output = model.predict(X)
        for i in range(len(X)):
        cur = np.zeros((200, 200, 3))
        cur[:,:,0] = X[i][0][:,:,0]
        cur[:,:,1:] = output[i][0]
        cur = (cur * [100, 255, 255]) - [0, 128, 128]
        rgb_image = lab2rgb(cur)
        cv2.imwrite('try'+i+'.png',rgb_image)

        '''


def test():
        images,_  = load_images_from_folder()
        '''
        cnn_approach(images)
        '''
        X,Y = create_dataset(images)
        model = create_model()
        model.load_weights('model_wieghts.h5')
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


def autocolorize():
        images,_  = load_images_from_folder()
        img = images[0]
        rgb_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        classifier = autocolorize.load_default_classifier()
        gr_rgb = autocolorize.colorize(rgb_gray, classifier=classifier)
        cv2.imwrite('try.png',gr_rgb)

autocolorize()
