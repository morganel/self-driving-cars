import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

######################################################
# Load driving log
######################################################

driving_log = pd.read_csv('data/driving_log.csv')

######################################################
# data cleansing: remove points where speed is too slow
######################################################

ind = driving_log['speed']>20 
driving_log= driving_log[ind].reset_index()

######################################################
### Create validation set
######################################################

driving_log = driving_log.sample(frac=1).reset_index(drop=True)
validation_perc = .1
train_set, valid_set = train_test_split(driving_log, test_size = validation_perc)

######################################################
# Model definition
######################################################

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64, 200, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='valid',subsample=(2,2),init="he_normal"))
model.add(ELU())

model.add(Convolution2D(36, 5, 5, border_mode='valid',subsample=(2,2),init="he_normal"))
model.add(ELU())

model.add(Convolution2D(48, 5, 5, border_mode='valid',subsample=(2,2),init="he_normal"))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1),init="he_normal"))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1),init="he_normal"))
model.add(ELU())

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(100,init="he_normal"))
model.add(ELU())

model.add(Dense(50,init="he_normal"))
model.add(ELU())

model.add(Dense(10,init="he_normal"))
model.add(ELU())

model.add(Dense(1,init="he_normal"))

optimizer = Adam(lr=0.0001)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])

# include checkpoints to retrieve the weights after each epoch
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

######################################################
# Data Augmentation
######################################################

# Load data from file. For left and right cameras, we adjust the steering angle by _ANGLE_CORRECTION_
_ANGLE_CORRECTION_ = .3

def get_data_from_file(driving_log_record, camera_side):
    img_path = 'data/' + driving_log_record[camera_side][0].strip()
    steering_angle_correction = 0
    if (camera_side == 'left'):
        steering_angle_correction = _ANGLE_CORRECTION_ # move more towards the center i.e. right
    if (camera_side == 'right'):
        steering_angle_correction = -_ANGLE_CORRECTION_ # move more towards the center i.e. left
            
    y = driving_log_record['steering'][0] + steering_angle_correction 
    
    img = mpimg.imread(img_path)
    x = img_to_array(img)
    
    return x, y

# Resize image 
def resize_image(img):
    crop_img = img[50:140, :, :] #140/160 so that we don't see the car. 50 so that we don't see the sky (issue if car goes on hill?)
    resized_image = cv2.resize(crop_img, dsize = (200, 64), interpolation =  cv2.INTER_AREA) 
    return resized_image
   
# Generate random translation. we adjust the steering angle by _ANGLE_CORRECTION_FOR_TRANSLATION_ per translated pixel.
_ANGLE_CORRECTION_FOR_TRANSLATION_ = .008

def translate_rdm(img, steering_angle, translate_range_x = 50, translate_range_y = 0):
    rows,cols,_ = img.shape
    translate_pixels_x = random.uniform(-translate_range_x, translate_range_x)
    translate_pixels_y = random.uniform(-translate_range_y, translate_range_y)
    M = np.float32([[1,0,translate_pixels_x],[0,1,translate_pixels_y]])
    trans_img = cv2.warpAffine(img, M, (cols,rows))
    trans_steering_angle = steering_angle + _ANGLE_CORRECTION_FOR_TRANSLATION_ * translate_pixels_x
    
    return trans_img, trans_steering_angle

######################################################
# Bath generators for training and validation
######################################################
def valid_batch_generator(data, batch_size):
    
    examples_ct = len(data.index)
    batch_ct = int(examples_ct/batch_size)
    
    batch_x = np.zeros((batch_size, 64, 200, 3))
    batch_y = np.zeros(batch_size)
    
    data = data.sample(frac=1).reset_index(drop=True)
    sample_id = 0
    
    while 1:
        for i_batch in range(batch_size):
            
            if sample_id == examples_ct:
                data = data.sample(frac=1).reset_index(drop=True)
                sample_id = 0

            driving_log_record = data.iloc[[sample_id]].reset_index()                
            
            img_side_rdm = np.random.randint(3)
            camera_side = 'center'
            if img_side_rdm == 0:
                camera_side = 'left'
            if img_side_rdm == 2:
                camera_side = 'right'
                
            x,y = get_data_from_file(driving_log_record, camera_side)  

            x = resize_image(x)

            batch_x[i_batch] = x
            batch_y[i_batch] = y
            
            sample_id = sample_id + 1

        yield batch_x, batch_y 

def train_batch_generator(data, batch_size):
    
    examples_ct = len(data.index)
    batch_ct = int(examples_ct/batch_size)
    
    batch_x = np.zeros((batch_size, 64, 200, 3))
    batch_y = np.zeros(batch_size)
    
    data = data.sample(frac=1).reset_index(drop=True)
    sample_id = 0
    
    while 1:
        for i_batch in range(batch_size):
            
            if sample_id == examples_ct:
                # Reshuffling once all the examples have been seen
                data = data.sample(frac=1).reset_index(drop=True)
                sample_id = 0

            driving_log_record = data.iloc[[sample_id]].reset_index()
            
            # pick which camera to load data from
            img_side_rdm = np.random.randint(3)
            camera_side = 'center'
            if img_side_rdm == 0:
                camera_side = 'left'
            if img_side_rdm == 2:
                camera_side = 'right'
            
            # Load data
            x,y = get_data_from_file(driving_log_record, camera_side)
            
            # Random Translation
            x, y = translate_rdm(x, y)

            # Horizontal flip
            horiz_flip = np.random.randint(2)
            if horiz_flip == 0:
                x = cv2.flip(x,1)
                y = -y
            
            # Resize image
            x = resize_image(x)
            
            #######################

            batch_x[i_batch] = x
            batch_y[i_batch] = y
            
            sample_id = sample_id + 1

        yield batch_x, batch_y 
                              
batch_size = 256
train_generator = train_batch_generator(
    train_set,
    batch_size=batch_size)

valid_generator = valid_batch_generator(
    valid_set,
    batch_size=batch_size
)

######################################################
# Train model
######################################################
_EPOCHS_ = 2
hist = model.fit_generator(train_generator, validation_data = valid_generator,
                    samples_per_epoch= 28160,
                    nb_val_samples = 1024,
                    nb_epoch=_EPOCHS_, callbacks=callbacks_list)

######################################################
# Save training and validation loss by epoch
######################################################

hist_history = hist.history
pickle.dump( hist_history, open( "hist_history.p", "wb" ) )

######################################################
# Save weights of last epoch's model
######################################################
model.save_weights('model.h5')  