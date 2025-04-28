
#Functions to make a mask of gastruloid for Timelapse BF


#%%Importing base modules
import os
import numpy as np
import matplotlib.pyplot as plt
#    % matplotlib inline
import skimage
import skimage.io as skio
import time
import skimage as ski
import skimage.io as ski_io
from skimage import filters
from matplotlib.animation import ArtistAnimation
from matplotlib.image import AxesImage
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
# import cupyx.scipy.ndimage as cp_ndi
#make sure to pip install

from skimage.morphology import binary_erosion, disk, closing, binary_dilation, remove_small_objects

import matplotlib.pyplot as plt
import numpy as np


#%% Function to create a new directory

def create_directory(path2directory):
    
    try:
        os.makedirs(path2directory, exist_ok = True)
        print("Directory '%s' created successfully" % path2directory)
    except OSError as error:
            print("Directory '%s' can not be created" % path2directory)
    return
#%% Function load_image_TL_BF to load image and check to make it 8bit if it is 16bit

def load_image_TL_BF(image_name):
    '''
    
    Loads a TIFF image from a file, as a numpy.ndarray
    Used with 2D+time images, time should be the first axis
    
    Parameters
    ----------
    image_name : name of the image to be loaded from the directory (str) - must be a tif image

    Returns
    -------
    image : the image as a numpy.ndarray

    '''
    image_raw = skio.imread(image_name, plugin="tifffile")
    if(image_raw.dtype=='<u2'):
        image = (image_raw/256).astype('uint8')
    elif(image_raw.dtype=='<u1'):
        image = image_raw
    else:
        print('Unknown data type - accepted data types are UINT8 or UINT16')
    return image

#%% Functions to generate a first mask

def Get_2D_Mask_Otsu_BF4X(im,size_smallobj):
    
    # After picking threshold
    import skimage.filters as skfilt
    from skimage import morphology
    from scipy import ndimage
    import cv2
    # Do Threshold 
    im_smooth = cv2.blur(im,(2,2))

    thresh_val = skfilt.threshold_otsu(im_smooth)
    new_mask = im_smooth < thresh_val
    # Remove small clumps of cells and fill holes, then smooth
    Mask1 = morphology.remove_small_objects(new_mask, size_smallobj, connectivity=2)
    Mask2 = ndimage.binary_fill_holes(Mask1).astype(int)

    return Mask2

def Get_2D_Mask_Otsu_BF4X2(im,size_smallobj,smooth):
    
    # After picking threshold
    import skimage.filters as skfilt
    from skimage import morphology
    from scipy import ndimage
    import cv2
    # Do Threshold 
    im_smooth = cv2.blur(im,(smooth,smooth))

    thresh_val = skfilt.threshold_otsu(im_smooth)
    new_mask = im_smooth < thresh_val
    # Remove small clumps of cells and fill holes, then smooth
    Mask1 = morphology.remove_small_objects(new_mask, size_smallobj, connectivity=2)
    Mask2 = ndimage.binary_fill_holes(Mask1).astype(int)

    return Mask2

def Get_2D_Mask_Otsu_BF4X3(im,size_smallobj,smooth):
    
    # After picking threshold
    import skimage.filters as skfilt
    from skimage import morphology
    from scipy import ndimage
    from skimage import exposure

    import cv2
    # Do Threshold 
    im_eq = exposure.adjust_gamma(im,0.75)

    im_smooth = cv2.blur(im_eq,(smooth,smooth))

    thresh_val = skfilt.threshold_otsu(im_smooth)
    new_mask = im_smooth < thresh_val
    # Remove small clumps of cells and fill holes, then smooth
    Mask1 = morphology.remove_small_objects(new_mask, size_smallobj, connectivity=2)
    Mask2 = ndimage.binary_fill_holes(Mask1).astype(int)
    Mask2 = Mask2.astype(np.uint8)

    return Mask2
    


def Get_2D_Mask_Yen_BF4X(im,size_smallobj,smooth):
    
    # After picking threshold
    import skimage.filters as skfilt
    from skimage import morphology
    from scipy import ndimage
    import cv2
    # Do Threshold 
    im_smooth = cv2.blur(im,(smooth,smooth))

    thresh_val = skfilt.threshold_yen(im_smooth)
    new_mask = im_smooth < thresh_val
    # Remove small clumps of cells and fill holes, then smooth
    Mask1 = morphology.remove_small_objects(new_mask, size_smallobj, connectivity=2)
    Mask2 = ndimage.binary_fill_holes(Mask1).astype(int)

    return Mask2


def Remove_border_touching_objects(mask):
    
    import cv2
    
    mask = mask.astype(np.uint8)
    mask2 = np.full(mask.shape,255)-mask
    mask2 = mask2.astype(np.uint8)
    h, w = mask.shape
    M = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(mask2, M, (0,0), 255)
        
    cv2.floodFill(mask2, M, (w-1,0), 255)

    M_inv = np.full(mask2.shape,255)-mask2
    M_inv = M_inv.astype(np.uint8)
    M = mask | M_inv
    
    return M

def Remove_border_touching_objects2(mask):
    
    from skimage.segmentation import clear_border

    M = clear_border(mask)
    M.astype(np.uint8)
    
    return M


#%%Other mask making functions (partly from Melody)

def binarize(image):

    imeq = (image - np.min(image) ) / ( np.max(image) - np.min(image))

    from skimage import filters
    from skimage import morphology
    from skimage import exposure
    import cv2
    
    image = exposure.adjust_gamma(imeq,0.75)
    #image = filters.sobel(image)
    
    Mx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    My = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    
    grad = np.sqrt((Mx+My)**2)
    img = filters.gaussian(grad, sigma=5)

    val = filters.threshold_otsu(img)

    binary = np.zeros_like(img, dtype=int)
    binary[img > val] = 1

    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary

    filled = morphology.reconstruction(seed, mask, method='erosion')

    return np.int0(filled)

def clean_binarized(filled):

    from skimage import morphology
    from skimage.measure import label
    from skimage.segmentation import clear_border

    labels = label(filled)
    eroded = morphology.binary_erosion(labels, footprint=morphology.disk(15))
    cleared = morphology.remove_small_objects(eroded, 10000)
    dilated = morphology.binary_dilation(cleared, footprint=morphology.disk(15))
    img=np.zeros_like(filled)
    img[dilated[:]] = 1
    img= clear_border(img)


    return img

def clean_binarized2(filled,sizesmallobj,sizedisk):

    from skimage import morphology
    from skimage.measure import label
    from skimage.segmentation import clear_border

    labels = label(filled)
    eroded = morphology.binary_erosion(labels, footprint=morphology.disk(sizedisk))
    cleared = morphology.remove_small_objects(eroded, sizesmallobj)
    dilated = morphology.binary_dilation(cleared, footprint=morphology.disk(sizedisk))
    img=np.zeros_like(filled)
    img[dilated[:]] = 1
    img= clear_border(img)


    return img
#%% Function to define seeding points

def define_seeding_pts(mask,n_pts_seed,i):
    import cv2
    import numpy as np
    
    Point_in = 0*np.empty([n_pts_seed,2], dtype=float)

    connectivity = 8
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
# Get the results
    num_labels = output[0]# The first cell is the number of labels
    labels = output[1] # The second cell is the label matrix
    stats = output[2] # The third cell is the stat matrix
    centroids = output[3] # The fourth cell is the centroid matrix
    
    try:
        n_max = 1+np.argmax(stats[1:,4])  #Making sure you get the biggest labelled area - except for background (label 0)
        A_max = max(stats[1:,4])
        [x_g, y_g] = centroids[n_max] 
            
        Point_in[0,:] = centroids[n_max]
    except:
        print('problem when defining centroid point '+str(i))
    
    ###### #########SELECTING RANDOM POINTS
    if (n_pts_seed>1):
        try:     
            Maskii=labels==n_max
            idx = np.argwhere(Maskii)
            random_idx = idx[np.random.choice(len(idx),size=n_pts_seed-1,replace=False)]
            random_pts = np.fliplr(random_idx)
                
            Point_in[1:n_pts_seed,:]= random_pts   
        
        except:
            print('problem when selecting random points '+str(i))
        
    return Point_in, A_max





#%%         Point_in = JP_TL_Mask_functions.Get_mask_seeding_points(M, n_pts, i)

def Erode_mask_depending_on_size(mask,ratio):
    
    import cv2
    import math
    
    #ERODE DEPENDING ON MASK SIZE
    mask = mask.astype(np.uint8)
    
    connectivity = 8
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    # Get the results
    num_labels = output[0] # The first cell is the number of labels

    labels = output[1]# The second cell is the label matrix

    stats = output[2] # The third cell is the stat matrix

    n_max = 1+np.argmax(stats[1:,4])  #Making sure you get the biggest labelled area - except for background (label 0)
    A_max = max(stats[1:,4])
    r = math.sqrt(A_max/math.pi)

    kernel = np.ones((int(r*ratio),int(r*ratio)), np.uint8)

    M = cv2.erode(mask, kernel)
    
    return M

def Get_mask_seeding_points(M, n_pts, i,ratio_erosion):
    
    #ratio_erosion = 0.4 #starting point
    Pt_in = None
    
    while Pt_in is None:
        try:
            M1 = Erode_mask_depending_on_size(M, ratio_erosion)
            Pt_in, A_in = define_seeding_pts(M1, n_pts, i)
        except:
            ratio_erosion = ratio_erosion-0.1
    return Pt_in, A_in, ratio_erosion, M1

def Get_mask_seeding_points_no_erosion(M, n_pts, i):
    
    ratio_erosion = 0 #starting point
    Pt_in = None
    
    while Pt_in is None:
        try:
            M1 = M
            Pt_in, A_in = define_seeding_pts(M1, n_pts, i)
        except:
            print("problem when defining CoM")
    return Pt_in, A_in, ratio_erosion, M1
#%% Function for SAM prediction

def SAM_prediction_from_im_pts(im,point_in,model_checkpoint):
    
    from segment_anything import build_sam, SamPredictor
    import cv2 
    
    imrgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    predictor = SamPredictor(build_sam(checkpoint=model_checkpoint))

    predictor.set_image(imrgb)
    input_point = point_in[:,:]
    input_label = np.ones(len(point_in))

    #mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, scores, logits  = predictor.predict(point_coords=input_point,point_labels=input_label, multimask_output=True)

    return masks, scores, logits

def Get_Area_predict(masks, scores):
    import numpy as np
    n = len(scores)
    Areas = np.zeros(n)
    for i in range(n):
        Mi = masks[i,:,:]
        Mi = Mi.astype(np.uint8)
        if (Mi.max()>0):
            Areas[i] = Mi.sum()/Mi.max()
            
    
    return Areas


def SAM_prediction_from_im_pts_gpu(im,point_in,model_checkpoint, device):
    
    from segment_anything import build_sam, SamPredictor, sam_model_registry
    import cv2 
    
    imrgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    model_type = "vit_h"
#    sam_checkpoint = "sam_vit_h_4b8939.pth"
#    model_type = "vit_h"

#device = "cuda"

#    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)
    sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
#    sam = build_sam(checkpoint=model_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    #predictor = SamPredictor(sam)
    
#    sam = sam_model_registry[model_type](checkpoint="/home/ubuntu/pallawi/metaai/segment-anything/sam_vit_h_4b8939.pth")
#    
#    sam.to(device=device)
    predictor.set_image(imrgb)
    input_point = point_in[:,:]
    input_label = np.ones(len(point_in))

    #mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, scores, logits  = predictor.predict(point_coords=input_point,point_labels=input_label, multimask_output=True)

    return masks, scores, logits

def Get_Area_predict(masks, scores):
    import numpy as np
    n = len(scores)
    Areas = np.zeros(n)
    for i in range(n):
        Mi = masks[i,:,:]
        Mi = Mi.astype(np.uint8)
        if (Mi.max()>0):
            Areas[i] = Mi.sum()/Mi.max()
            
    
    return Areas
#%% get_img_contour_array - Get array with contour of a mask

def get_img_contour_array(mask) :
    '''
    

    Parameters
    ----------
    mask : a binary image (mask of the organoid) as a 2D numpy.ndarray

    Returns
    -------
    cnt : the list of coordinates of the contour of the mask as a 2D numpy.ndarray

    '''
    
    import cv2
    r_cont = np.zeros_like(mask, dtype='bool')

    try:
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cont_i = np.squeeze(contours[0])
        r_cont[(cont_i[:,1]),(cont_i[:,0])]=255
        cnt = np.argwhere(r_cont) # Get coordinates of contour of gastruloid
        return cnt

    except:
        print('problem when defining contour ')
    
#%% Selecting which mask depending on area criteria

def Filter_Mask_by_Area(A_max, mask_i, masks, scores,i, ratio_upper_segment=2.5, ratio_lower = 0.9, ratio_upper_previous=2):
    
    if mask_i is not None: #if there is a previously existing mask from another time point (and you opted to take it into account)
        if (mask_i.max()>0): #if this mask is not empty
            A_old = mask_i.sum()/mask_i.max() #Area of the old mask
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh = Areas>(A_old*ratio_lower) # Areas that are larger than 90% of the old area
            Ar_thresh_high = Areas<(A_old*ratio_upper_previous) # Areas that are smaller than 2.5*old area 
            Ar_thresh_high2 = Areas<(A_max*ratio_upper_segment) # Areas that are smaller that 2.5*the area of the initially determined mask (the one used to find the centroid)
            scores_f = scores*Ar_thresh*Ar_thresh_high*Ar_thresh_high2 #filter the scores of the masks by which meets the 3 criterias above
            mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
            print('Using area of t-1 for filtering '+str(i))
            print('A_old  '+str(A_old)+', A_mask eroded '+str(A_max)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] #if the existing mask was empty, just keep the one with the highest score

            
    else:  #If you do not take into account the mask at t-1
        if (A_max>0): #If the initial mask (without SAM) was not empty
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh_high = Areas<(A_max*ratio_upper_segment)  ## Areas that are smaller than 2.5* the area of the initially determined mask (changed from 1.1 to 2.5)
            scores_f = scores*Ar_thresh_high 
            mask_i = masks[np.argmax(scores_f), :, :]  # select the best mask, with the highest score, meeting the area criteria
            print('Using area of initial mask for filtering '+str(i))

        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] 

                
    mask_i = np.squeeze(mask_i)
    mask_i = mask_i.astype(np.uint8)
    
    return mask_i


#%% Selecting which mask depending on area criteria V2

def Filter_Mask_by_Area2(A_in, mask_i, masks, scores,i, ratio_upper_segment=2.5, ratio_lower = 0.9, ratio_upper_previous=2):
    
    if mask_i is not None: #if there is a previously existing mask from another time point (and you opted to take it into account)
        if (mask_i.max()>0): #if this mask is not empty
            A_old = mask_i.sum()/mask_i.max() #Area of the old mask
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh = Areas>(A_old*ratio_lower) # Areas that are larger than 90% of the old area
            Ar_thresh_high = Areas<(A_old*ratio_upper_previous) # Areas that are smaller than 2*old area
            Ar_thresh_high2 = Areas<(A_in*ratio_upper_segment) # Areas that are smaller that 2.5*the area of the initially determined mask (the one used to find the centroid)
            scores_f = scores*Ar_thresh*Ar_thresh_high*Ar_thresh_high2 #filter the scores of the masks by which meets the 3 criterias above
            
            if np.max(scores_f)>0:
                mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                print('Using area of t-1 for filtering '+str(i))
                print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
            else:
                
                scores_f = scores*Ar_thresh*Ar_thresh_high
                
                if (np.max(scores_f)>0):
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Using area of t-1 but not A_in_mask for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
                else:
                    scores_f=scores
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Could not use previous area or eroded mask area for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

                    
        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] #if the existing mask was empty, just keep the one with the highest score

            
    else:  #If you do not take into account the mask at t-1
        if (A_in>0): #If the initial mask (without SAM) was not empty
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh_high = Areas<(A_in*ratio_upper_segment)  ## Areas that are smaller than 2.5* the area of the initially determined mask (changed from 1.1 to 2.5)
            scores_f = scores*Ar_thresh_high 
            mask_i = masks[np.argmax(scores_f), :, :]  # select the best mask, with the highest score, meeting the area criteria
            print('Using area of initial mask for filtering '+str(i))
            print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))


        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] 

                
    mask_i = np.squeeze(mask_i)
    mask_i = mask_i.astype(np.uint8)
    
    return mask_i

#%% Selecting which mask depending on area criteria V3

def Filter_Mask_by_Area3(A_in, mask_i, masks, scores,i, ratio_upper_segment=2.5, ratio_lower = 0.9, ratio_upper_previous=2):
    
    if mask_i is not None: #if there is a previously existing mask from another time point (and you opted to take it into account)
        if (mask_i.max()>0): #if this mask is not empty
            A_old = mask_i.sum()/mask_i.max() #Area of the old mask
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh = Areas>(A_old*ratio_lower) # Areas that are larger than 90% of the old area
            Ar_thresh_high = Areas<(A_old*ratio_upper_previous) # Areas that are smaller old area * ratio
            Ar_thresh_high2 = Areas<(A_in*ratio_upper_segment) # Areas that are smaller than ratio*ratio*area of the initially determined mask (the one used to find the centroid)
            scores_f = scores*Ar_thresh*Ar_thresh_high*Ar_thresh_high2 #filter the scores of the masks by which meets the 3 criterias above
            
            if np.max(scores_f)>0:
                mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                print('Using area of t-1 for filtering '+str(i))
                print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
            else: 
                
                scores_f = scores*Ar_thresh*Ar_thresh_high
                
                if (np.max(scores_f)>0):
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Using area of t-1 but not A_in_mask for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
                else:
                    scores_f=scores
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Could not use previous area or eroded mask area for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

                    
        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] #if the existing mask was empty, just keep the one with the highest score

            
    else:  #If you do not take into account the mask at t-1
        if (A_in>0): #If the initial mask (without SAM) was not empty
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh_high = Areas<(A_in*ratio_upper_segment)  ## Areas that are smaller than ratio* the area of the initially determined mask 
            #Ar_thresh_low = Areas > A_in*(ratio_upper_segment-1)
            scores_f = scores*Ar_thresh_high
            mask_i = masks[np.argmax(scores_f), :, :]  # select the best mask, with the highest score, meeting the area criteria
            print('Using area of initial mask for filtering '+str(i))
            print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] 
            print('Not using any area for filtering mask '+str(i))
            print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

                
    mask_i = np.squeeze(mask_i)
    mask_i = mask_i.astype(np.uint8)
    
    return mask_i

#%% Selecting which mask depending on area criteria V4


def Filter_Mask_by_Area4(A_in, mask_i, masks, scores,i, ratio_upper_segment=2.5, ratio_lower = 0.9, ratio_upper_previous=2):
    
    if mask_i is not None: #if there is a previously existing mask from another time point (and you opted to take it into account)
        if (mask_i.max()>0): #if this mask is not empty
            A_old = mask_i.sum()/mask_i.max() #Area of the old mask
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh = Areas>(A_old*ratio_lower) # Areas that are larger than 90% of the old area
            Ar_thresh_high = Areas<(A_old*ratio_upper_previous) # Areas that are smaller than 2*old area
            Ar_thresh_high2 = Areas<(A_in*ratio_upper_segment) # Areas that are smaller that ratio*the area of the initially determined mask (the one used to find the centroid)
            scores_f = scores*Ar_thresh*Ar_thresh_high*Ar_thresh_high2 #filter the scores of the masks by which meets the 3 criterias above
            
            if np.max(scores_f)>0:
                mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                print('Using area of t-1 for filtering '+str(i))
                print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
            else:
                
                scores_f = scores*Ar_thresh*Ar_thresh_high
                
                if (np.max(scores_f)>0):
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Using area of t-1 but not A_in_mask for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
                else:
                    scores_f=scores
                    mask_i = masks[np.argmax(scores_f), :, :] # select the best mask, with the highest score, meeting all criterias
                    print('Could not use previous area or eroded mask area for filtering '+str(i))
                    print('A_old  '+str(A_old)+', A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

                    
        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] #if the existing mask was empty, just keep the one with the highest score

            
    else:  #If you do not take into account the mask at t-1
        if (A_in>0): #If the initial mask (without SAM) was not empty
            Areas = Get_Area_predict(masks, scores) #get the areas of the predicted masks
            Ar_thresh_high = Areas<(A_in*ratio_upper_segment)  ## Areas that are smaller than 2.3+ratio * the area of the initially determined mask 
            Ar_thresh_low = Areas > A_in*(ratio_upper_segment-1)
            scores_f = scores*Ar_thresh_high*Ar_thresh_low
            if np.max(scores_f)>0:
                mask_i = masks[np.argmax(scores_f), :, :]  # select the best mask, with the highest score, meeting the area criteria
                print('Using area of initial mask for filtering upper and lower bounds '+str(i))
                print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
            else:
                scores_f = scores*Ar_thresh_high
                if np.max(scores_f)>0:
                    mask_i = masks[np.argmax(scores_f), :, :]  # select the best mask, with the highest score, meeting the area criteria
                    print('Using area of initial mask for filtering upper bounds '+str(i))
                    print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
                else:
                    scores_f=scores
                    mask_i = masks[np.argmax(scores_f), :, :] 
                    print('Not using any area for filtering mask '+str(i))
                    print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))
 
        else:
            scores_f=scores
            mask_i = masks[np.argmax(scores_f), :, :] 
            print('Not using any area for filtering mask '+str(i))
            print('A_mask eroded '+str(A_in)+', Areas ' +str(Areas)+', scores_f '+str(scores_f)+', ratio_upper_segment '+str(ratio_upper_segment))

                
    mask_i = np.squeeze(mask_i)
    mask_i = mask_i.astype(np.uint8)
    
    return mask_i

#%% Maybe not needed ?

#%% Function Generate enhanced contrast
def Generate_enhanced_contrast(im):
    import cv2
    
    
    im_BG =  cv2.GaussianBlur(im,(41,41),0)
    im_unBG = im-im_BG
    Iinv = cv2.bitwise_not(im_unBG)
    #plt.imshow(Iinv)
    im_en =  cv2.GaussianBlur(Iinv,(11,11),0)
    #plt.imshow(im_en)

    return im_en





