
#Functions to characterize gastruloid shape/images in BF


#%%Importing base modules
import os
import numpy as np
import matplotlib.pyplot as plt
#    % matplotlib inline
import skimage
import skimage.io as skio
import math


#%% Function to create a new directory

def create_directory(path2directory):
    
    try:
        os.makedirs(path2directory, exist_ok = True)
        print("Directory '%s' created successfully" % path2directory)
    except OSError as error:
            print("Directory '%s' can not be created" % path2directory)
    return

#%% Function to load json file
def load_json_results(json_name):
    import pandas as pd
    import os
    
    if os.path.isfile(json_name):
        df_plate = pd.read_json(json_name, orient='columns')    
        d = df_plate.copy()
        return d
    
    else:
        print('No analysis file found for '+ json_name)


#%% Function initialize time point

def initialize_time_pt(image, d, i):
    
    import numpy as np
    from scipy import ndimage as ndi

    im = image[i,:,:]

    Mask_init = np.zeros_like(im, dtype='uint8')
    


    Cont = d.at[i,'Mask'][0] 
#re-create the initial mask for shape measurements
    C = np.asarray(Cont)
    Mask_init[np.round(C[:,0]).astype('int'), np.round(C[:,1]).astype('int')] = 255
    Mask_init[:,:] = ndi.binary_fill_holes(Mask_init[:,:])
    Cont_2 = skimage.measure.find_contours(Mask_init[:,:])
    
    return im, Mask_init, Cont_2

#%% Function shape descriptors

def get_shape_descriptors(Mask, i, df):
    
       from skimage import measure

       label, num_lab= measure.label(Mask,return_num=True,connectivity=2)
       Prop = measure.regionprops(label)      
       Moments_Hu = Prop[0].moments_hu
       
       pxsize = df.at[i,'Pixel_size']
       
       df.at[i, 'Solidity'] = Prop[0].solidity
       df.at[i, 'Perimeter'] = pxsize*Prop[0].perimeter
       df.at[i, 'AR'] = Prop[0].axis_major_length/Prop[0].axis_minor_length
       df.at[i,'Major_Axis_um'] = [pxsize*Prop[0].axis_major_length]
       df.at[i, 'Area'] = pxsize*pxsize*Prop[0].area
       df.at[i, 'Circularity'] = 4*math.pi*Prop[0].area/(Prop[0].perimeter*Prop[0].perimeter)


       
       return df
   
#%% Function get_info_intensity


def get_info_intensity(im, Mask, df,i):
    
    from skimage import measure
    label, num_lab= measure.label(Mask,return_num=True,connectivity=2)
    Prop_int = measure.regionprops(label, intensity_image=im)
    df.at[i,'Avg_Intensity_in_mask'] = Prop_int[0].intensity_mean
    df.at[i,'Avg_Intensity_of_im'] = np.mean(im)
    df.at[i,'Ratio_avg_int_in_vs_tot'] = Prop_int[0].intensity_mean/np.mean(im)
    
    return df

#%% Function prepare for medial axis

def prepare_contour_medial_axis(im, Cont_2):
    
    import skimage
    from scipy import ndimage as ndi
    import numpy as np
    
    Mask_smooth = np.zeros_like(im, dtype='bool')
    ####### Smoothing the contourCont to make a smoothed mask, for medial axis
    Im_smooth = skimage.filters.gaussian(im, 5, preserve_range=False)
    
    init = (max(Cont_2, key=len))
    snake = skimage.segmentation.active_contour(Im_smooth,init, alpha=0.01, beta=10, gamma=0.5, w_line=0 , w_edge=0)

#Increase the number of points in snake
    from scipy import interpolate
            
    Npoints = 20000
    tck, u = interpolate.splprep([snake[:,0], snake[:,1]], per=1)
    xi, yi = interpolate.splev(np.linspace(0, 1, Npoints), tck)
    snake_2=np.array([xi,yi]).transpose()
              
# Create a contour image by using the contour coordinates rounded to their nearest integer value
    Mask_smooth[np.round(snake_2[:, 0]).astype('int'), np.round(snake_2[:, 1]).astype('int')] = 1
# Fill in the hole created by the contour boundary
    Mask_smooth[:,:] = ndi.binary_fill_holes(Mask_smooth[:,:])

    return Mask_smooth


#%% Function get_medial_axis

def get_medial_axis(mask):

    from skimage.morphology import medial_axis
      
    # Look for Medial axis
    
    sk, distance = medial_axis(mask, return_distance=True)
    
    
    plt.imshow(sk*100, cmap='magma',vmin=0, vmax=1)
    plt.contour(mask, [0.5], colors='w')
    
    return sk
#%% Get image with contour of a mask

def get_img_contour(im) :
    from skimage import measure

    contours = measure.find_contours(im, 0.8)
    cont = np.squeeze(contours)
    contint = np.array(cont, dtype=int)
    Img_Cont = np.zeros((im.shape[0], im.shape[1]), np.uint8)
    Img_Cont[contint[:,0],contint[:,1]]=255
    
    return Img_Cont

#%% find_longest_edge

def find_longest_edge(l, T):
    '''  '''
    e1 = T[l[0]][l[1]]['weight']
    e2 = T[l[0]][l[2]]['weight']
    e3 = T[l[1]][l[2]]['weight']
    E=[e1,e2,e3]

    # if any(e ==0 for e in E):
    #     return()
    if e2 < e1 > e3:
        return (l[0], l[1])
    elif e1 < e2 > e3:
        return (l[0], l[2])
    elif e1 < e3 > e2:
        return (l[1], l[2])

#%% orderPoints
def orderPoints( C_skel):
    ''' Reorder the coordinates skel of the skeleton '''
    #20220726 : Judith trying to comment to explain the code
    from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
    import networkx as nx

    G = kneighbors_graph(C_skel, 2, mode='distance')
#    T = nx.from_scipy_sparse_matrix(G)
    T = nx.from_scipy_sparse_array(G)

    degrees = np.array([val for (node, val) in T.degree()]) # JP: Number of links from each node
    endidx = np.argwhere(degrees == 1).squeeze() #JP: Selecting the indexes where degree=1, so only 1 link, meaning this is an extremity
    

    end_cliques = [i for i in list(nx.find_cliques(T)) if len(i) == 3] # JP: Look for cliques of 3 nodes
    edge_lengths = [find_longest_edge(i, T) for i in end_cliques] #JP: in the cliques of 3 nodes look for longest edge
    T.remove_edges_from(edge_lengths) #JP: remove these edges to have only cliques of 2 (?)

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(C_skel))]
    mindist = np.inf
    minidx = 0

    lenp=[]
    costs=[]
    for i in range(len(C_skel)):
        p = paths[i]           # order of nodes
        lenp.append(len(p))
        ordered = C_skel[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        costs.append(cost)
        if cost < mindist:
            mindist = cost
            minidx = i

    opt_order = paths[minidx]
    C_skel_O = C_skel[opt_order, :]

    return C_skel_O, T


#%%interpol

def interpol(x,y):
    ''' '''

    m, p = np.polyfit(x, y, 1)
    n=-1

    return [m,n,p]
#%% interpolateEnds
def interpolateEnds( C_skel, ratio=3, lenMin=2):
    ''' '''

    xSkel, ySkel = C_skel.T
    if len(xSkel)<2*lenMin+1:  #lenMin sets a minimum number of points for interpolation
        indBeg = np.min([lenMin, len(xSkel)])
        indEnd = np.min([0, np.abs(len(xSkel)- lenMin)])
    else:
#        indBeg = np.max( [np.int(np.floor(len(ySkel)*ratio/100)), lenMin])
        indBeg = np.max( [int(np.floor(len(ySkel)*ratio/100)), lenMin])

#        indEnd = np.min( [np.int(np.ceil(len(ySkel)*(100-ratio)/100)), len(xSkel)- lenMin])
        indEnd = np.min( [int(np.ceil(len(ySkel)*(100-ratio)/100)), len(xSkel)- lenMin])

    xBeg = xSkel[:indBeg]
    yBeg = ySkel[:indBeg]

    xEnd = xSkel[indEnd:]
    yEnd = ySkel[indEnd:]

    fB = interpol(xBeg,yBeg)
    fE = interpol(xEnd,yEnd)

    return fB, fE  

#%% intersect
def intersect(C_cnt, f, P_skel, no=[]):
    '''  '''
    
    import networkx as nx
    
    m,n,p = f

    xCnt,yCnt = C_cnt.T
    finter = lambda x,y: m*x+n*y+p 
    res = [ finter(xCnt[i], yCnt[i]) for  i in range(len(xCnt))] 
    ind = np.argwhere(np.abs(res)< np.sqrt(2)*4)
    
    test_cnt = np.c_[xCnt[ind], yCnt[ind]] 
    
    from sklearn.neighbors import radius_neighbors_graph 

    G = radius_neighbors_graph(test_cnt, radius =10)
#    T = nx.from_scipy_sparse_matrix(G)
    T = nx.from_scipy_sparse_array(G)

    end_cliques =[i for i in list(nx.find_cliques(T))] 

    I =[]
    for i, a in enumerate(end_cliques):
        A = [ ind[a[j]][0] for j in range(len(a))] 
        resA = [res[j] for j in A] 
        if all(a not in no for a in A) : 
            I.append(A[np.argmin(np.abs(resA))]) 

    m = [ np.sum( (xCnt[a]- P_skel[0] )**2 + (yCnt[a]- P_skel[1] )**2) for a in I]
    intersectI = I[np.argmin(m)]

    return intersectI 


#%%appendMedialAxis
def appendMedialAxis( C_skel,  C_endpoint, order = 'after' ):
    
    
    xSkel, ySkel = C_skel.T
    if order == 'after':
        ind = -1
    elif order == 'before':
        ind = 0
    mainDir = np.argmax( [ np.abs(xSkel[ind]- C_endpoint[0]), np.abs(ySkel[ind]- C_endpoint[1])] )
    #: if mainDir ==0, the line is mostly in the x axis, 
        # if mainDir==1, the line is mostly in the y axis
    if mainDir == 0:
        xa = np.arange(C_endpoint[0], xSkel[ind], 0.8*np.sign(xSkel[ind]- C_endpoint[0]))
        # , dtype=int)
        # create x values from C_endpoint to the extremity of the skeleton (start or end depending on ind)
            # with a spacing of 0.8 or -0.8 depending on with x extremity is larger than the other
        ya = np.linspace(C_endpoint[1], ySkel[ind], num= len(xa))
        # , dtype=int)
        # create the same number of y values
    elif mainDir == 1: 
        ya = np.arange(C_endpoint[1], ySkel[ind], 0.8*np.sign(ySkel[ind]- C_endpoint[1]))
        # , dtype=int)
        xa = np.linspace(C_endpoint[0], xSkel[ind],num= len(ya))
                         # , dtype=int)

    if order == 'after':
        xMA = np.concatenate((xSkel[:-1], np.flip(xa)), axis=0)
        yMA = np.concatenate((ySkel[:-1], np.flip(ya)), axis=0)
    elif order == 'before':
        xMA = np.concatenate((xa, xSkel[1:]), axis=0)
        yMA = np.concatenate((ya, ySkel[1:]), axis=0)

    C_axis = np.c_[xMA,yMA]

    return C_axis 

#%% expand_skel_return
def expand_skel_return(C_skel, C_cnt, ratio=3, lenMin=2):
    '''  '''

    if len(C_skel)>3:
        C_skel, _ = orderPoints(C_skel) #JP: safety, arranging the order of points in C_skel
    
    
    fb,fe = interpolateEnds(C_skel, ratio, lenMin)
    
   
    Ib = intersect(C_cnt, fb, C_skel[0])
    if fb==fe:
        Ie = intersect(C_cnt, fe, C_skel[-1], no=[Ib])
    else:
        Ie = intersect(C_cnt, fe, C_skel[-1])
    
    # Appending the extremities
    C_axis = appendMedialAxis(C_skel, C_cnt[Ib], order='before')
    C_axis = appendMedialAxis(C_axis, C_cnt[Ie], order='after')
    if len(C_axis)>3:
        C_axis, _ = orderPoints(C_axis)#making sure points are well ordered
    return C_axis, Ib, Ie # returning the list of coordinates of the prolonged medial axis, and the indexes of the crossing points in the contour


def get_full_medial_axis(Mask_s):
    
    import numpy as np
                  ############################## GET MEDIAL AXIS FOR THE LAST 24 TIME POINTS
    skel = get_medial_axis(Mask_s)
    skel = np.argwhere(skel) # Get coordinates of medial axis
    Img_Cont = get_img_contour(Mask_s)
    cnt = np.argwhere(Img_Cont) # Get coordinates of contour of gastruloid
                  
    Skel_full, ie, ib = expand_skel_return(skel, cnt, ratio=4, lenMin=10) 
    
    return Skel_full

#%% Get medial axis length

def get_medial_axis_length(skelfull, df, i):
    
    import math
    Cum_dist = 0
    pxsize = df.at[i,'Pixel_size']
    
    for l in range(1,(len(skelfull))):
                  
        x_med = skelfull[l,0]
        y_med = skelfull[l,1]
        x_med_prev = skelfull[(l-1),0]
        y_med_prev = skelfull[(l-1),1]    
        Cum_dist = Cum_dist + pxsize*math.sqrt((x_med-x_med_prev)**2 + (y_med-y_med_prev)**2)
    
    return Cum_dist

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

def Remove_border_touching_objects(mask):
    
    import cv2
    
    mask = mask.astype(np.uint8)
    mask2 = np.full(mask.shape,255)-mask
    mask2 = mask2.astype(np.uint8)
    h, w = mask.shape
    M = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(mask2, M, (0,0), 255)
        
    M_inv = np.full(mask2.shape,255)-mask2
    M_inv = M_inv.astype(np.uint8)
    M = mask | M_inv
    
    return M

def Erode_mask_depending_on_size(mask):
    
    import cv2
    import math
    
    #ERODE DEPENDING ON MASK SIZE
    connectivity = 8
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    # Get the results
    num_labels = output[0] # The first cell is the number of labels

    labels = output[1]# The second cell is the label matrix

    stats = output[2] # The third cell is the stat matrix

    n_max = 1+np.argmax(stats[1:,4])  #Making sure you get the biggest labelled area - except for background (label 0)
    A_max = max(stats[1:,4])
    r = math.sqrt(A_max/math.pi)

    kernel = np.ones((int(r*0.2),int(r*0.2)), np.uint8)

    M = cv2.erode(mask, kernel)
    
    return M

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

def Filter_Mask_by_Area(A_max, mask_i, masks, scores, logits,i, ratio_upper_segment=2.5, ratio_lower = 0.9, ratio_upper_previous=2):
    
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





