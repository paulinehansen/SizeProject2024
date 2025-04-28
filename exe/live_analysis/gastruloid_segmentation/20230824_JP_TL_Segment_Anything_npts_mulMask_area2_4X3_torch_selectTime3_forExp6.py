# -*- coding: utf-8 -*-


#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys # to handle exceptions
import math
import cv2
cv2.useOptimized()
from segment_anything import build_sam, SamPredictor
import pandas as pd
from scipy import ndimage as ndi

from func_timeout import func_timeout, FunctionTimedOut

import skimage
import skimage.io as skio
import torch
import torchvision
import tifffile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import JP_TL_Mask_functions

print(os.getcwd())


checkpoint = "sam_vit_h_4b8939.pth"
Use_area_evolution = sys.argv[13]
mypath = sys.argv[1]

name = sys.argv[2]


os.chdir(mypath)
directory_mask = "Mask"
JP_TL_Mask_functions.create_directory(directory_mask)
#%%

image = JP_TL_Mask_functions.load_image_TL_BF(name)


# Read the postfix_time argument from the command line
#postfix_time = sys.argv[12]

# Construc#t the filename using the base name and postfix_time
#base_name = '50cells_time'
filename = '50cells_time.csv'

# Read the CSV file into a DataFrame
df_time = pd.read_csv(filename)
  
# postfix_time = sys.argv[12]
# df_time = pd.read_csv(r''+name[2:-4]+postfix_time)
Time = df_time.Time

n_times = len(Time)

#%%Info about acquisition

obj = sys.argv[10] # objective used 
pxsize = sys.argv[3] #in microns
ExpDate = sys.argv[4]
Microscope = sys.argv[5]
Condition = sys.argv[6]
Replicate = sys.argv[7]
dt = int(sys.argv[8])
n_file = int(sys.argv[9])
n_pts = int(sys.argv[11]) #Number of seed points
t_start = float(sys.argv[14])

filename = name+'Analysis_SAM_'+Condition+'_npts'+str(n_pts)+'from_'+str(t_start)+'.json'

#def JP_2DGastru_Profile (filesource,  name, imgtype, obj, n_ch, channel_use_mask, IsMask, pxsize, Channels_list, post_mark, df, n_file, to_analyze ):    
df = pd.DataFrame({'Exp_Date': pd.Series(dtype='str'),
                     'Path_data': pd.Series(dtype='str'),
                     'Name': pd.Series(dtype='str'),
                     'Well_ID': pd.Series(dtype='str'),
                     'Condition': pd.Series(dtype='str'),
                     'Replicate': pd.Series(dtype='str'),
                     'Keep_image': pd.Series(dtype='object'),
                     'Microscope': pd.Series(dtype='str'),
                     'Objective': pd.Series(dtype='int'),
                     'Pixel_size': pd.Series(dtype='float'),
                     'TimePt': pd.Series(dtype='object'),
                     'Border': pd.Series(dtype='str'),
                     'Mask': pd.Series(dtype='object'),
                     'Point_in': pd.Series(dtype='object'),
                     'Segments': pd.Series(dtype='object'),
                     'MedialAxis': pd.Series(dtype='object'),
                     'MA_length_um': pd.Series(dtype='float'),
                     'Solidity': pd.Series(dtype='object'),
                     'AR': pd.Series(dtype='object'),
                     'Circularity': pd.Series(dtype='object'),
                     'Area': pd.Series(dtype='object'),
                     'Perimeter': pd.Series(dtype='object'),
                     'Major_Axis_um': pd.Series(dtype='object'),
                     'Avg_Intensity_in_mask': pd.Series(dtype='object'),
                     'Avg_Intensity_of_im': pd.Series(dtype='object'),
                     'Ratio_avg_int_in_vs_tot': pd.Series(dtype='object')
})

df['Exp_Date'] = np.full(n_times, ExpDate)
df['Path_data'] = np.full(n_times, mypath)
df['Name'] = np.full(n_times, name)
df['Condition'] = np.full(n_times, Condition)
df['Replicate'] = np.full(n_times, Replicate)
df['Microscope'] = np.full(n_times, [Microscope])
df['Objective'] = np.full(n_times,obj)
df['Pixel_size'] = np.full(n_times,pxsize)
df['Well_ID'] =  np.full(n_times,name.split('[')[1].split('_')[0])####
df['N_file'] = np.full(n_times,n_file)
df['TimePt'] = Time
df = df.assign(Keep_image='Y')



#%% Looping for each frame
mask_i = None
Point_in_tot =  0*np.empty([n_pts,2,image.shape[0]], dtype=float)
r_mask = np.zeros_like(image,dtype='bool')
mask_thresh = np.zeros_like(image,dtype='bool')

#for i in range(image.shape[0]):
for i in range(len(Time)):
    
    
    if (df.at[i,'TimePt']>=t_start):
        
        if (Use_area_evolution=='N'):
            mask_i = None

        #im = image[i, :, :]
        im = image[i,:,:] #For exp1
        #im = (imi/256).astype('uint8') #For exp1
        try:
            #Mask2 = func_timeout(50, JP_TL_Mask_functions.Get_2D_Mask_Yen_BF4X, args=(im,1000,2,))
            Mask2 = func_timeout(50, JP_TL_Mask_functions.Get_2D_Mask_Otsu_BF4X2, args=(im,1000,2,))
#            tifffile.imwrite('Mask'+name[1:]+'Mask_0_'+str(i)+'.tiff', Mask2)

            #Mask1 = func_timeout(50, JP_TL_Mask_functions.binarize, args=(im,))
        except FunctionTimedOut:
            print("could not complete within 50seconds and was terminated. 1")
            mask_i = None
        except:
            print("other problem 1")
            mask_i = None

        try:
            M = func_timeout(50, JP_TL_Mask_functions.Remove_border_touching_objects2, args=(Mask2,))
            M=M.astype(np.uint8)
            #tifffile.imwrite('Mask'+name[1:]+'Mask_1_'+str(i)+'.tiff', M)

        except FunctionTimedOut:
            print("could not complete within 50seconds and was terminated. 3")
            mask_i = None
        except:
            print("other problem 3")
            mask_i = None

        try:
            ratio=0.2
            Point_in, A_in, ratio, M_thresh = func_timeout(360, JP_TL_Mask_functions.Get_mask_seeding_points, args=(M,n_pts,i,ratio,))

            mask_thresh[i, :, :] = M_thresh
            Point_in_tot[:, :, i] = Point_in
            df.at[i, 'Point_in'] = [Point_in]
            #tifffile.imwrite('Mask'+name[1:]+'Mask_2_'+str(i)+'.tiff', M_thresh)

        except FunctionTimedOut:
            print("could not complete within 360seconds and was terminated. 4 " + str(i))
            mask_i = None
        except:
            print("other problem 4")
            mask_i = None

    
    ###### Do the segmentation using SAM
        try:
            masks, scores, logits = func_timeout(120, JP_TL_Mask_functions.SAM_prediction_from_im_pts_gpu, args=(im,Point_in,checkpoint,device,))
            #np.save('Mask'+name[1:]+'t_'+str(i)+'_masks.npy', masks)
            #np.save('Mask'+name[1:]+'t_'+str(i)+'_scores.npy', scores)
                #masks, scores, logits = JP_TL_Mask_functions.SAM_prediction_from_im_pts_gpu(im,Point_in,checkpoint,device)
        except FunctionTimedOut:
            print("could not complete within time and was terminated - GPU step.")
            mask_i = None
        except:
            print("other problem  GPU")
            mask_i = None

        try:
            ratio_up = 2.3+ratio
            mask_i = func_timeout(120, JP_TL_Mask_functions.Filter_Mask_by_Area3, args=(A_in,mask_i,masks,scores,i,ratio_up,0.9,1.5) )

           #     mask_i = JP_TL_Mask_functions.Filter_Mask_by_Area2(A_in, mask_i, masks, scores,i, ratio_upper_segment=(3+ratio), ratio_lower = 0.9, ratio_upper_previous=1.5)
        except:
            print('Problem Filter mask '+str(i))
            mask_i = None

        r_mask[i,:,:] = mask_i
             

         
         
        cnt= JP_TL_Mask_functions.get_img_contour_array(mask_i)
        
        df.at[i,'Mask'] = [cnt]
        print(str(i))



    

r_mask = np.zeros_like(image,dtype='uint8')

for j in range(image.shape[0]):
    
    try:
        Mask_cnt = np.squeeze(np.array(df.at[j,'Mask']))
        r_mask[j,Mask_cnt[:,0],Mask_cnt[:,1]]=255
        r_mask[j,:,:]=255*ndi.binary_fill_holes(r_mask[j,:,:])

    except:
        print('pb')
        

import tifffile

tifffile.imwrite('Mask/'+name+'Mask.tiff', r_mask) ### sep


#tifffile.imwrite('Mask'+name[1:]+'Mask.tiff', r_mask)
tifffile.imwrite('Mask/'+name+'Maskthresh.tiff', mask_thresh) ##sep


# save the Point_in array to a file
np.save('Mask/'+name+'Points_in.npy', Point_in_tot) #sep

df= df.to_json(filename)



