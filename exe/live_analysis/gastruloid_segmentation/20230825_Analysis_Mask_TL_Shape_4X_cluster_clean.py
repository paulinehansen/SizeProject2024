# -*- coding: utf-8 -*-


## Importing the mask


#!/usr/bin/python
#coding:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sys # to handle exceptions
import math
import cv2
import pandas as pd
from scipy import ndimage as ndi
import skimage
import skimage.io as skio

import JP_TL_shapeanalysis_functions



print(os.getcwd())

mypath = sys.argv[1]
os.chdir (mypath)

name = sys.argv[2]
print(name)



#%%INPUT

n_pts = sys.argv[3]
Condition = sys.argv[4]
t_start = float(sys.argv[5])

filename_json_in = name+'Analysis_SAM_'+Condition+'_npts'+str(n_pts)+'from_'+str(t_start)+'.json'

filename_out =name+'Analysis_SAM_'+Condition+'_npts'+str(n_pts)+'_shape.json'

d = JP_TL_shapeanalysis_functions.load_json_results(filename_json_in)


directory_visual = "Visual_outputs"
JP_TL_shapeanalysis_functions.create_directory(directory_visual)

#%%LOADING DATA

image = skio.imread(name, plugin="tifffile")
pxsize = d.at[0,'Pixel_size']

#%% For each frame

for i in range(0,len(d)):
    
    try:
        
        im, Mask_init, Cont = JP_TL_shapeanalysis_functions.initialize_time_pt(image, d, i)

        d = JP_TL_shapeanalysis_functions.get_shape_descriptors(Mask_init, i, d)
        d = JP_TL_shapeanalysis_functions.get_info_intensity(im, Mask_init, d, i)
        
        Mask_smooth = JP_TL_shapeanalysis_functions.prepare_contour_medial_axis(im, Cont)
        
        try:
            Skel_full = JP_TL_shapeanalysis_functions.get_full_medial_axis(Mask_smooth)
            Cum_dist = JP_TL_shapeanalysis_functions.get_medial_axis_length(Skel_full, d, i)
        except Exception as e:
            print(e)
            
            Cum_dist = np.nan
            Skel_full = np.nan            
            print('Problem for Medial Axis at time '+str(i))

        d = d.astype({"MedialAxis": object})  
        d.at[i,'MedialAxis'] = [Skel_full]
        d.at[i,'MA_length_um'] = Cum_dist  
    
    except:
        print('Problem at time '+str(i))    

            
######################################################################


df = d.copy()

df = df.to_json(filename_out)
