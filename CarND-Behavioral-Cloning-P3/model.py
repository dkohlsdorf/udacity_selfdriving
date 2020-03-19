import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import sklearn
import sys
from workspace_utils import active_session
 
from keras.applications.vgg16 import VGG16

from collections import namedtuple
from sklearn.model_selection import train_test_split
from random import *

from keras.layers import * 
from keras.models import *
from keras.optimizers import * 


def relative(path, img):
    '''
    Convert local path to cloud path
    
    :param path: the local path from the simulator
    :param img: folder containing the image folder
    :returns: the fixed path
    '''
    return "{}/IMG/{}".format(img, path.split('/')[-1])


class DrivingInstance(namedtuple("DrivingInstance", "center left right angle")):
    '''
    Instance holding the three images from the simulator and the steering angle
    '''
    CORRECTION = 0.2

    @property
    def flipped_angle(self):
        return -self.angle
    
    @property
    def left_angle(self):
        return self.angle + DrivingInstance.CORRECTION

    @property
    def right_angle(self):
        return self.angle - DrivingInstance.CORRECTION

    @property
    def img_center(self):
        return mpimg.imread(self.center)
    
    @property
    def img_flipped(self):
        return np.fliplr(mpimg.imread(self.center))
        
    @property
    def img_left(self):
        return mpimg.imread(self.right)

    @property
    def img_right(self):
        return mpimg.imread(self.left)

    @classmethod
    def from_log(cls, paths): 
        '''
        build an iterator over several driving logs
        
        :param paths: sequence of csv files from the simulator
        :returns: iterator over driving instances
        '''
        for path in paths:
            df = pd.read_csv(path)
            p  = "/".join(path.split("/")[:-1])
            print(p)
            df["center"] = df["center"].apply(lambda x: relative(x, p))
            df["left"]   = df["left"].apply(lambda x: relative(x, p)) 
            df["right"]  = df["right"].apply(lambda x: relative(x, p))
            for _, row in df.iterrows():
                yield cls(
                    row['center'], 
                    row['left'], 
                    row['right'],
                    row['steering']
                )

                
def transfer_model(input, base_model):
    '''
    Build a model using transfer lerning by
    reusing convolutional layers.
    
    :param input: input dimensions (w, h, c)
    :param base_model: keras pretrained model
    :returns: model that outputs steering angle for an input image
    '''
    i    = Input(input)
    x    = Cropping2D(cropping=((50,20), (0,0)))(i)
    x    = Lambda(lambda x: (x / 255.0) - 0.5)(x) 
    base  = base_model(input_tensor=x, include_top=False)
    x    = base.output 
    x    = Flatten()(x) 
    x    = BatchNormalization()(x)
    x    = Dense(1024, activation='relu')(x) 
    x    = Dense(256,  activation='relu')(x) 
    x    = Dropout(0.5)(x)
    x    = Dense(128,  activation='relu')(x) 
    x    = Dense(1,    activation='linear')(x) 
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    model.summary()
    return model


def nvidia(input):
    '''
    Build the nvidia self driving model.
    
    :param input: input dimensions (w, h, c)
    :returns: model that outputs steering angle for an input image
    '''
    i     = Input(input)
    x     = Cropping2D(cropping=((50,20), (0,0)))(i)
    x     = Lambda(lambda x: (x/255.0) - 0.5)(x)
    x     = Conv2D(32, (5,5), padding='same', strides=(2,2), activation='relu')(x)
    x     = Conv2D(32, (5,5), padding='same', strides=(2,2), activation='relu')(x)
    x     = Conv2D(64, (5,5), padding='same', strides=(2,2), activation='relu')(x)
    x     = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x     = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x     = Flatten()(x) 
    x     = Dense(100, activation='relu')(x) 
    x     = Dense(50, activation='relu')(x) 
    x     = Dropout(0.5)(x)
    x     = Dense(10, activation='relu')(x) 
    x     = Dense(1, activation='linear')(x) 
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='mse',
                  optimizer='adam')
    model.summary()
    return model
            
            
def inception(input):
    '''
    Build an inception model.
   
    :param input: input dimensions (w, h, c)
    :returns: model that outputs steering angle for an input image
    '''

    i     = Input(input)
    x     = Cropping2D(cropping=((50,20), (0,0)))(i)
    x     = Lambda(lambda x: (x/255.0) - 0.5)(x)
    
    # incption module 1
    a     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    a     = Conv2D(16, (3,3), padding='same', activation='relu')(a)
    b     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    b     = Conv2D(16, (5,5), padding='same', activation='relu')(b)
    c     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    x     = Concatenate(axis=3)([a,b,c])
    x     = MaxPooling2D((2,2))(x)
    
    # residual connection
    l1    = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    l1    = MaxPooling2D((2,2))(l1)
    
    # incption module 2
    a     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    a     = Conv2D(16, (3,3), padding='same', activation='relu')(a)
    b     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    b     = Conv2D(16, (5,5), padding='same', activation='relu')(b)
    c     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    x     = Concatenate(axis=3)([a,b,c])
    x     = MaxPooling2D((2,2))(x)
        
    # incption module 3
    a     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    a     = Conv2D(16, (3,3), padding='same', activation='relu')(a)
    b     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    b     = Conv2D(16, (5,5), padding='same', activation='relu')(b)
    c     = Conv2D(16, (1,1), padding='same', activation='relu')(x)
    x     = Concatenate(axis=3)([a,b,c,l1])
    x     = MaxPooling2D((2,2))(x)
    
    # classification network
    x     = Flatten()(x) 
    x     = Dense(256,  activation='relu')(x) 
    x     = Dropout(0.5)(x)
    x     = Dense(128,  activation='relu')(x) 
    x     = Dropout(0.5)(x)
    x     = Dense(1,    activation='linear')(x) 
    
    # assemble model 
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='mse',
                  optimizer='adam')
    model.summary()
    return model


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                angles.append(batch_sample.left_angle)
                images.append(batch_sample.img_left)
                angles.append(batch_sample.angle)
                images.append(batch_sample.img_center)
                angles.append(batch_sample.right_angle)
                images.append(batch_sample.img_right)
                angles.append(batch_sample.flipped_angle)
                images.append(batch_sample.img_flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            data    = sklearn.utils.shuffle(X_train, y_train)
            yield data
            
            
if __name__ == "__main__":
    with active_session():
        print("Training: {}".format(sys.argv[1:]))
        batch_size = 32
        train_samples, validation_samples = train_test_split(
            [instance for instance in DrivingInstance.from_log(sys.argv[1:])], test_size=0.2)
        train_generator      = generator(train_samples,      batch_size=batch_size)
        validation_generator = generator(validation_samples, batch_size=batch_size)
        
        m = nvidia((160, 320, 3))
        history_object = m.fit_generator(
            train_generator, 
            steps_per_epoch  = math.ceil(len(train_samples)/batch_size), 
            validation_data  = validation_generator, 
            validation_steps = math.ceil(len(validation_samples)/batch_size), 
            epochs=3, 
            verbose=1)
        m.save('model.h5')

        print("HistoryKeys: {}".format(history_object.history.keys()))
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('loss.png')
        plt.close()