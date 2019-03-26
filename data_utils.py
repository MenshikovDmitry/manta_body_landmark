'''

Define a batch generator class for the masked dugong dataset
 

From Keras doc

        # we create two instances with the same arguments
        data_gen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        image_datagen.fit(images, augment=True, seed=seed)
        mask_datagen.fit(masks, augment=True, seed=seed)
        
        image_generator = image_datagen.flow_from_directory(
            'data/images',
            class_mode=None,
            seed=seed)
        
        mask_generator = mask_datagen.flow_from_directory(
            'data/masks',
            class_mode=None,
            seed=seed)
        
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)
        
        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50)

    
'''


#from datetime import datetime
import os
import os.path
#import time

import glob # glob.glob
import configparser

import numpy as np

import pprint

from skimage.io import imread
from skimage.transform import resize

from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte

import cv2 as cv


from tensorflow import keras

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


#np.random.seed(7) # fix random seed for reproducibility


# =============================================================================

    
# =============================================================================
    

def makeXpositiveXmask(
        positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/positive', 
        mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/mask',
        target_size=(cnn_input_side, cnn_input_side)):
    '''
    Load in an array X_positive the images located in positive_dir
    Load in an array X_mask the masks located in positive_dir
    
    X_mask[i] is the mask of the image X_positive[i]
    
    The returned arrays  X_positive, X_mask are float32 with entries in [0,1]
    Both X_positive, X_mask are 4D
    
    @return X_positive, X_mask
            
    '''
    positive_list = sorted( glob.glob(os.sep.join([positive_dir, '*.png'])) )
    mask_list = sorted( glob.glob(os.sep.join([mask_dir, '*.npy'])) )
    
    num_example = len(positive_list)
    assert num_example == len(mask_list)
    
    #debug
#    num_example = 6
#    positive_list = positive_list[:num_example]
#    mask_list = mask_list[:num_example]
    
    X_positive = np.array([
            resize(imread(file_name), target_size)
               for file_name in positive_list],dtype=np.float32)
    
    X_mask = np.array([
            resize(np.load(file_name), target_size)
               for file_name in mask_list], dtype=np.float32)
    
    X_mask =  np.expand_dims(X_mask,-1)

    
#    print(X_mask.min(), X_mask.max(), X_positive.min(), X_positive.max())
#    print(X_positive.shape, X_mask.shape)
    return X_positive, X_mask

# =============================================================================

def debug_view_positive(x_batch,m_batch):
    '''
    Visualize a batch (input image, mask)    
    '''
    num_example = x_batch.shape[0]    
    assert (num_example>0)  and (m_batch.shape[0]==num_example)
    nrows, ncols = x_batch.shape[1:3]

    display_array = np.empty((nrows, 2*ncols,3),np.uint8)
    
    for i in range(num_example):
        display_array[:,:ncols,:] = img_as_ubyte(x_batch[i])
#        display_array[:, ncols:,:] = np.expand_dims(img_as_ubyte(m_batch[i]),-1)    
        
        display_array[:, ncols:,:] = img_as_ubyte(
                        mark_boundaries(img_as_ubyte(x_batch[i]), 
                     img_as_ubyte(m_batch[i][:,:,0]),color=(1,0,0)))    
        cv.imshow('marked', cv.cvtColor(display_array, cv.COLOR_RGB2BGR))
#        cv.imshow('marked', display_array)    
        k = cv.waitKey(0)
        if k == 27:         # esc to exit
            break

# =============================================================================

def debug_view_all(x_batch,m_batch):
    '''
    Visualize a batch (input image, mask)    
    '''
    num_example = x_batch.shape[0]    
    assert (num_example>0)  and (m_batch.shape[0]==num_example)
    nrows, ncols = x_batch.shape[1:3]

    display_array = np.empty((nrows, 2*ncols,3),np.uint8)
    
#    if np.any(x_batch<0) or np.any(x_batch>1):
#        pass
    
#    print('<debug_view_all> debug : ' ,x_batch.dtype, x_batch.shape, x_batch.min(),x_batch.max())
    for i in range(num_example):
#        print('<debug_view_all> debug : i ' ,i, x_batch[i].min(),x_batch[i].max())
        display_array[:,:ncols,:] = img_as_ubyte(x_batch[i])
#        display_array[:, ncols:,:] = np.expand_dims(img_as_ubyte(m_batch[i]),-1)    
        
        display_array[:, ncols:,:] =  cv.cvtColor( 
                    img_as_ubyte(m_batch[i]), cv.COLOR_GRAY2RGB )

        cv.imshow('marked', cv.cvtColor(display_array, cv.COLOR_RGB2BGR))
#        cv.imshow('marked', display_array)    
        k = cv.waitKey(0)
        if k == 27:         # esc to exit
            break
    
# =============================================================================

class MaskedDugongBatchGenerator(object):
    '''
    @pre
        there are three directories
            positive  : contains images of dugongs
            negative  : contains images of background
            mask : contains masks of the images of directory 'positive'
    
    '''
    
    def __init__(self, 
             negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/negative', 
             positive_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/positive', 
             mask_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/mask',
             batch_size = batch_size # config value of the module
             ):
        '''
        Build an array iterator for the positive examples
        and a directory iterator the the negative examples
        
        '''
        #        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        assert batch_size % 2 == 0        
        # we return   batch_size/2 positive examples and 
        #             batch_size/2 negative examples for each call to
        #               self._get_batches_of_transformed_samples()
        
        self.batch_size = batch_size
        self.batch_size_2 = int(self.batch_size/2)

        X_positive, X_mask = makeXpositiveXmask(positive_dir,                                                 
                                                    mask_dir, 
                                                    target_size=(cnn_input_side, cnn_input_side))
        self.X_positive = X_positive
        self.X_mask = X_mask

        data_gen_args = dict(
#                featurewise_center=False,
#                samplewise_center=False,
#                featurewise_std_normalization=False,
#                samplewise_std_normalization=False,
#                zca_whitening=False,
                rotation_range=180.0,    # degree range  
                width_shift_range=0.05,  # Float (fraction of total width). Range for random horizontal shifts.
                height_shift_range=0.05, # Float (fraction of total height). Range for random vertical shifts.
                zoom_range=0.1,         # Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
                fill_mode='reflect', #'nearest', 
                horizontal_flip=True, 
                vertical_flip=True,
                rescale=1./255)
        
        # Trick to get synchronized positive images and their masks:
        #   we pass the same seed to positive_datagen.flow() and 
        #     mask_datagen.flow()
        common_seed = 1
        positive_datagen  = keras.preprocessing.image.ImageDataGenerator(data_gen_args)
        positive_datagen.featurewise_center = None  # to prevent warning message
        mask_datagen = keras.preprocessing.image.ImageDataGenerator(data_gen_args) 
        mask_datagen.featurewise_center = None # to prevent warning message
                        
        #    save_to_dir: None or str (default: None). 
        #    This allows you to optimally specify a directory to which to save the 
        #    augmented pictures being generated 
                
        positive_generator = positive_datagen.flow(
                X_positive, # y is None
                batch_size = self.batch_size_2,
                #save_to_dir=save_dir,
                #save_prefix = 'positive_',
                seed=common_seed)
        #    
        mask_generator = mask_datagen.flow(
                X_mask, # y is None
                batch_size = self.batch_size_2,
                #save_to_dir=save_dir,
                #save_prefix = 'mask_',
                seed=common_seed)
        #    
        self.positive_mask_generator = zip(positive_generator, mask_generator)         
        #    x_batch, m_batch = next(self.positive_mask_generator)    
        
        # create a generator for negative examples
        negative_datagen  = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        negative_datagen.featurewise_center = None # to prevent warning message

        negative_parent_dir, negative_name  = os.path.split(negative_dir)
        print('** flow_from_directory')
        self.negative_generator = negative_datagen.flow_from_directory(
            directory = negative_parent_dir,  # validation images root directory
            classes = [negative_name],
            target_size = [cnn_input_side,cnn_input_side],
            class_mode = None,
            batch_size = self.batch_size_2,
            #save_to_dir='/home/frederic/Documents/data/dugong/keras_dugong_dir',
            seed=1)
                     
        
    def _get_batches_of_transformed_samples(self):
        '''
        Return a batch with 50% positive and 50% negative
        The generators for the positive examples and negative examples
        are created in the constructor.
        The positive examples are subjected to geometric transformation, but not
        the negative examples (because they are taken from a large directory).
        
        '''
        
        
        X_positive, M_positive = next(self.positive_mask_generator)
        X_negative = next(self.negative_generator) 
   
#        print('_get_batches_of_transformed_samples pos, mask, neg -> ', X_positive.shape, M_positive.shape, X_negative.shape )

        assert X_positive.shape[0]==M_positive.shape[0]
        
        batch_x = np.zeros((X_positive.shape[0]+X_negative.shape[0],cnn_input_side,cnn_input_side,3), 
                           dtype=keras.backend.floatx())        
        # the y target is a mask
        batch_y = np.zeros((X_positive.shape[0]+X_negative.shape[0],cnn_input_side,cnn_input_side,1), 
                           dtype=keras.backend.floatx())

        
        try:
            batch_x[:X_positive.shape[0]]  = X_positive
            batch_y[:X_positive.shape[0]]  = M_positive
            batch_x[X_positive.shape[0]:]  = X_negative
        # note that batch_y[X_positive.shape[0]:]  was set to zero when batch_y was created
        except ValueError:
            print('_get_batches_of_transformed_samples  X_positive.shape, M_positive.shape, X_negative.shape -> ', X_positive.shape, M_positive.shape, X_negative.shape )
            print('_get_batches_of_transformed_samples batch_x.shape, batch_y.shape -> ', batch_x.shape, batch_y.shape)
            raise

        return batch_x, batch_y

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples()

    
# =============================================================================
# debugging code
        
if  0:
    batch_size = 6
        
    X_positive, X_mask = makeXpositiveXmask()
        
    # should have rank 4. In case of grayscale data, the channels axis should have value 1, and in case of RGB 
    #X_positive =  None  # fill will all the positive image
    
    #X_mask = None
    
    data_gen_args = dict(
            rotation_range=180.0,    # degree range  
            width_shift_range=0.05,  # Float (fraction of total width). Range for random horizontal shifts.
            height_shift_range=0.05, # Float (fraction of total height). Range for random vertical shifts.
            zoom_range=0.1,         # Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
            fill_mode='reflect', #'nearest', 
            horizontal_flip=True, 
            vertical_flip=True)
    
            
    
    common_seed = 1
    positive_datagen  = keras.preprocessing.image.ImageDataGenerator(data_gen_args)        
    mask_datagen = keras.preprocessing.image.ImageDataGenerator(data_gen_args) 
        
        
    #    save_to_dir: None or str (default: None). 
    #    This allows you to optimally specify a directory to which to save the 
    #    augmented pictures being generated 
            
    positive_generator = positive_datagen.flow(
            X_positive, # y is None
            batch_size = int(batch_size/2),
            #save_to_dir=save_dir,
            save_prefix = 'positive_',
            seed=common_seed)
    #    
    mask_generator = mask_datagen.flow(
            X_mask, # y is None
            batch_size = int(batch_size/2),
            #save_to_dir=save_dir,
            save_prefix = 'mask_',
            seed=common_seed)
    #    
    train_generator = zip(positive_generator, mask_generator)
         
    #for x_batch, m_batch in train_generator:
    for k in range(4):
        x_batch, m_batch = next(train_generator)    
        print('view batch ', k)
        debug_view_all(x_batch,m_batch)

if 0: 
        # create a generator for negative examples
        negative_datagen  = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        
        negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/negative'
        negative_dir = '/home/frederic/Documents/data/hpc_png_dugong/train/'
        print('** flow_from_directory')
        negative_generator = negative_datagen.flow_from_directory(
            directory = negative_dir,  # validation images root directory
            classes = ['negative'],
            target_size = [cnn_input_side,cnn_input_side],
            class_mode = None,
            batch_size = 6,
            #save_to_dir='/home/frederic/Documents/data/dugong/keras_dugong_dir',
            seed=1)
        
        X = next(negative_generator)
        print(X.shape)

                
# =============================================================================
if __name__ == '__main__':
    pass
#    filename = '/home/frederic/Documents/data/hpc_png_dugong/train/negative'
#
#    a,b = os.path.split(filename)
#
##    print(os.path.basename(filename))
#    print(a)
#    print(b)
    
    mdbg = MaskedDugongBatchGenerator(batch_size = 6)
    for k in range(2):
        x_batch, m_batch = mdbg._get_batches_of_transformed_samples()   
        print('*******  view batch {} ******'.format(k))
        cv.imshow('debug',x_batch[4])
        debug_view_all(x_batch,m_batch)

    # print documentation
#    print(__doc__)
#    makeXpositiveXmask()
# =============================================================================
#
    
    
#    ++++++++++++++++++++++    debug leftovers    ++++++++++++++++++++++ 
#    
#    pprint.pprint(positive_list)
#    pprint.pprint(mask_list)
    
    