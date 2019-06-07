# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:50:11 2019

@author: Dima
"""

import multiprocessing as mp
from PIL import Image,ExifTags
import nbimporter # to import methods from another notebook
import manta_file_processing as mfp
import numpy as np
import random,copy,time,pickle
import pandas as pd

   
def rotate90(image,heatmaps,k):
    '''
    rotates image and heatmaps by 90,180 or 270 degrees for k=1,2,3
    '''
    rot_image=np.rot90(image,k=k)
    rot_hm=np.rot90(heatmaps,k=k)
    return rot_image,rot_hm
#-------------------------------------------
def crop_image(image,BLM,heatmap_order,bound_box=0.15,rs=None,verbose=False):
    '''
	crops the image. leaves external context (by default - 15%)
	inputs: image, RAW body landmarks, heatmap order
	
	returns image and RAW body landmarks
	
	'''
    if rs==None: rs=random.random()
        
    centers=[]
    body_landmarks=copy.deepcopy(BLM)
    for blm in body_landmarks["BLM"]:
        if blm[0] in (heatmap_order):
            centers.append(blm[1])
    centers=np.array(centers)

    mins=centers.min(axis=0)
    maxes=centers.max(axis=0)
    
    #set minimal border around landmarks
    min_area=maxes-mins
    delta=min_area*bound_box
    mins=mins-delta
    maxes=maxes+delta
    mins=mins.astype('int')
    maxes=maxes.astype('int')    

    heigh_width=np.array(body_landmarks["height_width"])
    #print("mins ",mins,maxes)

    h_crop=[0,0]
    v_crop=[0,0]
    
    ok=False
    turns=0
    
    if mins[0]<0: mins[0]=0
    if mins[1]<0: mins[1]=0
    if maxes[0]>heigh_width[0]: maxes[0]=heigh_width[0]
    if maxes[1]>heigh_width[1]: maxes[1]=heigh_width[1]
    
    while not ok and turns<10:
        turns+=1
        #horisontal cuts
        
        h_cut1=random.randint(0,mins[0])
        h_cut2=random.randint(maxes[0],heigh_width[0])   
        h_crop=[h_cut1,h_cut2]

        #vertical cuts
        v_cut1=random.randint(0,mins[1])    
        v_cut2=random.randint(maxes[1],heigh_width[1])    
        v_crop=[v_cut1,v_cut2]
        
        #check proportions
        w=v_cut2-v_cut1
        h=h_cut2-h_cut1
        proportion=min((h,w))/max((h,w))
        if proportion>0.7:
            if verbose: print(proportion,turns)
            ok=True
        
    #adjust body_landmark coordinates    
    for blm in body_landmarks["BLM"]:
        blm[1][0]-=h_cut1
        blm[1][1]-=v_cut1
        
    image_cropped=image[h_crop[0]:h_crop[1],v_crop[0]:v_crop[1],:]
    body_landmarks["height_width"]=image_cropped.shape[:2]
    if verbose: return image_cropped, body_landmarks, mins,maxes
    else: return image_cropped, body_landmarks
#=================================================

def exif_rotate(image):
'''
corrects image rotation with exif information
is not used
'''
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        return image


#==================================================

def create_datapoint(params):
'''
generates augmented datapoint. 
inputs: path to image file, path to labels file, sizes of image and heatmaps to be produced and heatmap order
outputs normalized image in numpy and heatmaps
'''
    im_path,blm_path,image_size,heatmap_size,heatmap_order=params
    try:
        image = Image.open(im_path)
        body_landmarks=mfp.extract_body_landmarks(blm_path)
        #if len(body_landmarks["BLM"])<4:
        #    print("not enough blms",blm_path[-60:])
        #    return(None,None)
    except:
        print(print("\nError on "+blm_path))
        return (None,None)
    else:
        #print(".",end="")
        image=np.array(image)[:,:,:3]
        
        #random_crop
        image,body_landmarks=crop_image(image,body_landmarks,heatmap_order)
        image=mfp.to_np(image,image_size)
                
        #generate heatmaps
        heatmaps=mfp.gen_heatmaps(body_landmarks,heatmap_size=heatmap_size)
                
        #random rotate 0,90,180 or 270
        image,heatmaps=rotate90(image,heatmaps,random.randint(0,3))
        norm_image=mfp.normalize(image)
        return (norm_image.astype('float64'),heatmaps.astype('float64'))
#---------------------------------------------------------------------------

   

def image_heatmap_worker(q1,q2,q3,l,log,params):
'''
worker for multiprocessing. is not used
'''
    log.put(" Im in ")
    while True:
        try:
            i,data=q1.get(timeout=1)
        except:
            log.put("except")
            break
        if i==None:
            log.put(" empty queue")
            break
        if i%100==0: log.put(i)
        if i%50==0 and i!=0: log.put("FFF")
        if i%10==0 :log.put(".")
        try:
            image,heatmaps=create_datapoint([*data,*params])
            l.acquire()
            q2.put(image)
            q3.put(heatmaps)
            l.release() 
        except Exception as ex:
            log.put(str(ex))
            break
    log.put("\nIm done ")
    log.put(None)
    time.sleep(2)
                
if __name__ == "__main__":
    #mp.set_start_method('spawn')
    
    batch_data=np.load("mant.npy")
    print (batch_data.shape)
    heatmap_order=("left-eye","right-eye","left-gill","right-gill","tail")
    image_size=[100]*2
    heatmap_size=[29]*2
    
    args=(image_size,heatmap_size,heatmap_order)
    #params=[(md[0],md[1],*args) for md in batch_data[:10]]
    #print(len(params))
    print([*batch_data[2],*args])
    
    data=list(batch_data[2])
    
    image,heatmaps=create_datapoint([*data,*args])
    
    from matplotlib import pyplot as plt
    plt.imshow(mfp.apply_heatmaps(mfp.denormalize(image),heatmaps))
    
    
    
    