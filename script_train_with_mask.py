'''
Script to train a CNN classifier in two stages;
- first cnn_1 is trained to predict masks
- then cnn_2 is trained to predict labels


the depth of the feature maps is progressively increasing in the network
(from 32 to 128), while the size of the feature maps is decreasing (from 148x148 to 7x7).
This is a pattern that you will see in almost all convnets.


Class ImageDataGenerator is defined in
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py

Need to implement this method of ImageDataGenerator

 def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            interpolation='nearest'):

or class Iterator(Sequence):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.


    L =  glob.glob(os.sep.join([sample_source_directory, '*.ppm']))
    Ls = random.sample(L, sample_size)

See dugong detectvor v4/ dugong detector

'''

#from datetime import datetime
import os
import os.path
#import time
import glob # glob.glob
import configparser

import numpy as np

from tensorflow import keras

#from keras.callbacks import ModelCheckpoint
#from keras import regularizers


from skimage.io import imread
from skimage import img_as_float

import matplotlib.pyplot as plt

import data_utils

# =====================     CONFIG.INI  parameters    =========================

config_full_name='config.ini' 
config = configparser.ConfigParser()        
# load the script parameters
config.read(config_full_name)

batch_size = config.getint('constants','batch_size')
cnn_input_side = config.getint('constants','cnn_input_side')

epochs = config.getint('constants','epochs')

train_image_dir = config.get('directories','train_image_dir')
test_image_dir = config.get('directories','test_image_dir')

save_dir = config.get('directories','save_dir')
log_dir = config.get('directories','log_dir')


np.random.seed(7) # fix random seed for reproducibility

# =============================================================================



#=============================================================================

def make_cnn_functional_model_1():
    '''
    
    SeparableConv2D(filters, kernel_size, ...
                    filters: Integer, the dimensionality of the output space 
                         (i.e. the number output of filters in the convolution).
                    kernel_size: An integer or tuple/list of 2 integers, 
                                        specifying the width and height of 
                                        the 2D convolution window. 
                                        Can be a single integer to specify the 
                                        same value for all spatial dimensions.

    # Compile model        
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

    '''
    
    # input_color_image 
    # InputLayer (None, 125, 125, 3) 
    input_tensor = keras.layers.Input(
            shape=(cnn_input_side,cnn_input_side,3) ,
            name='input_color_image')
    
    # first conv layer
    out_tensor = keras.layers.SeparableConv2D(32, 3, 
                                              padding='same',
                                              activation='relu')(input_tensor)
    out_tensor = keras.layers.MaxPooling2D((2, 2)) (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)
    # (None, 61, 61, 32)
    
    # second conv layer
    out_tensor = keras.layers.SeparableConv2D(32, 3, 
                                              padding='same',
                                              activation='relu')(out_tensor)
    out_tensor = keras.layers.MaxPooling2D((2, 2)) (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)
    #    (None, 29, 29, 32)
    
    # third conv layer
    out_tensor = keras.layers.SeparableConv2D(64, 3, 
                                              padding='same',
                                              activation='relu')(out_tensor)
    out_tensor = keras.layers.MaxPooling2D((2, 2)) (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)
    #    (None, 13, 13, 64)

    # fourth conv layer
    out_tensor = keras.layers.SeparableConv2D(64, 3, padding='same',
                                              activation='relu')(out_tensor)
    out_tensor = keras.layers.MaxPooling2D((2, 2)) (out_tensor)
    out_tensor = keras.layers.BatchNormalization(name = 'latent') (out_tensor)
    # (None, 5, 5, 64)


    #    keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), 
    #        padding='valid', data_format=None, activation=None, 
    #        use_bias=True, kernel_initializer='glorot_uniform', 
    #        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
    #        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    # conv trans 1
    out_tensor = keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size= 3,
                                              strides= 2,
                                              activation='relu') (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)

    # conv trans 2
    out_tensor = keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size= 3,
                                              strides= 2,
                                              activation='relu') (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)

    # conv trans 3
    out_tensor = keras.layers.Conv2DTranspose(filters=32,
                                              kernel_size= 3,
                                              strides= 2,
                                              activation='relu') (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)

    # conv trans 4
    out_tensor = keras.layers.Conv2DTranspose(filters=1,
                                              kernel_size= 3,
                                              strides= 2,
                                              name = 'last_conv_trans',
                                              activation='relu') (out_tensor)
    out_tensor = keras.layers.BatchNormalization() (out_tensor)

    #    trick to get from 127x127 to 125x125
    out_tensor = keras.layers.AveragePooling2D(pool_size=3, 
                                               strides=1, 
                                               padding='valid') (out_tensor)
    
    out_tensor = keras.layers.Activation('sigmoid')(out_tensor)


    # ...........




    model = keras.models.Model(inputs=input_tensor, outputs=out_tensor)   
        
    model.summary()
    
    return model

#=============================================================================
# ============================================================================== 
 

def train_with_mask_dugong_detector(model):
    '''
    Train the cnn model

    preprocessing_function=None,
    
    Launching the TensorBoard server from the command line
    $ tensorboard --logdir=my_log_dir
       
    You can then browse to localhost:6006 and look at your model training
    
    '''
    mdbg_train = data_utils.MaskedDugongBatchGenerator(
                 negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/negative', 
                 positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/positive',
                 mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/mask') 


    mdbg_valid = data_utils.MaskedDugongBatchGenerator(
                 negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/negative', 
                 positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/positive',
                 mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/mask') 
        
    x_test, y_test = next(mdbg_valid)

    # Compile model        
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

    
    callbacks_list = [    
        # This callback will interrupt training when we have stopped improving
#        keras.callbacks.EarlyStopping(
#            # This callback will monitor the validation accuracy of the model
#            monitor='val_binary_accuracy',
#            # Training will be interrupted when the accuracy
#            # has stopped improving for *more* than 1 epochs (i.e. 2 epochs)
#            patience=50,
#        ),
        # This callback will save the current weights after every epoch
        keras.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filepath=save_dir+'/best_checkpoint_model.hdf5', 
            verbose=1, 
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
                log_dir=log_dir, 
                histogram_freq=0,  
                write_graph=True, 
                write_images=False),
        keras.callbacks.ReduceLROnPlateau(
                # This callback will monitor the validation loss of the model
                monitor='val_loss',
                # It will divide the learning by 10 when it gets triggered
                factor=0.1,
                # It will get triggered after the validation loss has stopped improving
                # for at least 20 epochs
                patience=20,
            )
        
    ] # callbacks_list  
            
    
    # Fit the model on the batches generated by datagen.flow().
    model_hist = model.fit_generator(mdbg_train,
                        steps_per_epoch = epochs//batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
#                        validation_data=mdbg_valid,       
#                        validation_steps= 1 ,  # validation_steps=nb_validation_samples // batch_size
                        workers=1,  # might be a source of bug
                        callbacks=callbacks_list)
    
#                         callbacks= [keras.callbacks.TensorBoard(
#                log_dir=log_dir, 
#                histogram_freq=0,  
#                write_graph=True, 
#                write_images=False)])


    model_val_loss = model_hist.history['val_loss']


    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, 'last_cnn')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    plt.plot(model_val_loss, 'bo', label='Dropout-regularized model')
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')
    plt.legend()
    
    plt.show()
                        
# ============================================================================== 
    
    
if 0: # testing code
    pass
    mdbg_train = data_utils.MaskedDugongBatchGenerator(
             negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/negative', 
             positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/positive',
             mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/mask') 
    mdbg_valid = data_utils.MaskedDugongBatchGenerator(
                 negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/negative', 
                 positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/positive',
                 mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/test/mask') 
    for i in range(500):
        X,M = next(mdbg_train)
        print('train -> ', X.shape, M.shape)
#        assert X.shape==(64, 125, 125, 3) and  M.shape==(64, 125, 125, 1)
        X,M = next(mdbg_valid)
        print('valid -> ', X.shape, M.shape)
#        assert X.shape==(64, 125, 125, 3) and  M.shape==(64, 125, 125, 1)
    
if __name__ == '__main__':
    pass
    model = make_cnn_functional_model_1()
    train_with_mask_dugong_detector(model)

    # print documentation
#    print(__doc__)
    