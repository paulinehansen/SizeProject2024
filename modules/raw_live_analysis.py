import os
import cv2
import sys
import math
import time
import warnings
import numpy as np
import pandas as pd
import skimage as ski
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy import interpolate
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
#from skimage import data, img_as_float

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)

from modules.raw_IF_analysis import expand_skel_return
from modules.processed_data_analysis import load_processed_df #, prepare_multipolar_df
#from napari_segment_blobs_and_things_with_membranes import voronoi_otsu_labeling, threshold_otsu


# functions for best segmentation
def read_movie(mov_path):
    """
    Reads a movie file from the given path, calculates the time taken for reading, and returns the loaded movie data.

    The function utilizes the tifffile plugin to load the movie and prints the movie dimensions
    as well as the time taken to read the file.

    :param mov_path: The file path to the movie to be read.
    :type mov_path: str
    :return: The loaded movie data.
    :rtype: numpy.ndarray
    """
    start_time = time.time()
    movie = ski.io.imread(mov_path, plugin='tifffile')
    mov_time = time.time()
    print('Movie dim: ', movie.shape)
    print(" movie reading time --- %s seconds ---" % (mov_time - start_time))

    return movie


# function deployed after segmentation
def initialize_time_pt(image, d, i):
    """
    Initialize the time point for the given image and related data, extracting a single
    image slice, creating an initial mask based on provided contour points, and
    generating filled contours for further processing.

    This function primarily deals with generating initial contours and masks for a time
    point slice of a 3D image input by filling holes in the mask derived from the contour
    coordinates. It is useful in initializing and preparing data for shape measurements
    or related analysis.

    :param image: A 3D numpy array representing the input image stack where the first
        dimension corresponds to different time or z-index points.
    :param d: A pandas DataFrame containing metadata or auxiliary data about the input
        image. The DataFrame must include a 'Mask' column, where each entry is a list
        with coordinate points for contours.
    :param i: An integer index pointing to the specific time point or z-index slice of
        the image array and DataFrame to process.
    :return: A tuple containing:
        - The 2D image slice extracted from the 3D image array.
        - The binary mask initialized based on the contours from the 'Mask' column.
        - The filled contours detected from the initialized binary mask.
    """
    im = image[i, :, :]

    Mask_init = np.zeros_like(im, dtype='uint8')

    Cont = d.at[i, 'Mask'][0]  ###list coord
    # re-create the initial mask for shape measurements
    C = np.asarray(Cont)
    Mask_init[np.round(C[:, 0]).astype('int'), np.round(C[:, 1]).astype('int')] = 255
    Mask_init[:, :] = ndi.binary_fill_holes(Mask_init[:, :])
    Cont_2 = ski.measure.find_contours(Mask_init[:, :])

    return im, Mask_init, Cont_2


def initialize_time_pt_image(image, d, i):
    """
    Reinitializes a binary mask from contour coordinates and provides updated contour
    measurements for further processing.

    This function takes an initial image and reconstructs a binary mask from the contour
    coordinates provided in a dataframe. The mask is filled and the contours are recalculated,
    returning both the binary mask and the updated contour coordinates.

    :param image: Input image array which serves as the dimensional base for the mask.
    :type image: numpy.ndarray
    :param d: DataFrame containing contour coordinates.
    :type d: pandas.DataFrame
    :param i: Index in the dataframe `d` pointing to a row with mask contour coordinates.
    :type i: int
    :return: Tuple consisting of the reinitialized binary mask and the recalculated contours.
    :rtype: Tuple[numpy.ndarray, list]
    """
    Mask_init = np.zeros_like(image, dtype='uint8')

    Cont = d.at[i, 'Mask']  ###list coord
    # re-create the initial mask for shape measurements
    C = np.asarray(Cont)
    Mask_init[np.round(C[:, 0]).astype('int'), np.round(C[:, 1]).astype('int')] = 255
    Mask_init[:, :] = ndi.binary_fill_holes(Mask_init[:, :])
    Cont_2 = ski.measure.find_contours(Mask_init[:, :])

    return Mask_init, Cont_2


def get_shape_descriptors(Mask, i, df):
    """
    Calculate shape descriptors for a given binary mask and update the provided DataFrame.

    This function processes a binary mask using connected-component labeling and extracts
    region properties using the ``regionprops`` function. The shape descriptors, such as
    solidity, perimeter, aspect ratio (AR), major axis length, area, and circularity, are
    calculated based on the labeled region and pixel size. These descriptors are then
    updated in the provided DataFrame for the corresponding index.

    :param Mask: Binary mask image where connected components will be analyzed.
    :type Mask: numpy.ndarray
    :param i: Index of the row in the DataFrame to update with the computed shape descriptors.
    :type i: int
    :param df: DataFrame containing at least the 'Pixel_size' column. The computed shape
        descriptors will be stored in this DataFrame.
    :type df: pandas.DataFrame
    :return: Updated DataFrame containing the calculated shape descriptors for the given index.
    :rtype: pandas.DataFrame
    """
    label, num_lab = ski.measure.label(Mask, return_num=True, connectivity=2)
    Prop = ski.measure.regionprops(label)
    #Moments_Hu = Prop[0].moments_hu

    pxsize = df.at[i, 'Pixel_size']

    df.at[i, 'Solidity'] = Prop[0].solidity
    df.at[i, 'Perimeter'] = pxsize * Prop[0].perimeter
    df.at[i, 'AR'] = Prop[0].axis_major_length / Prop[0].axis_minor_length
    df.at[i, 'Major_Axis_um'] = [pxsize * Prop[0].axis_major_length]
    df.at[i, 'Area'] = pxsize * pxsize * Prop[0].area
    df.at[i, 'Circularity'] = 4 * math.pi * Prop[0].area / (Prop[0].perimeter * Prop[0].perimeter)

    return df

def get_info_intensity(im, Mask, df, i):
    """
    Extracts intensity information of a given mask applied on an image and updates the relevant
    details in a DataFrame.

    The function processes a mask to calculate intensity-based metrics of the image and updates
    a provided DataFrame with the calculated values. Specifically, the function computes the average
    intensity within the mask region, the average intensity of the complete image, the ratio between
    the average intensity in the mask and the entire image's average, and the sum of the intensity
    values within the mask region.

    :param im: The input image as a NumPy array used for intensity calculation.
    :type im: numpy.ndarray
    :param Mask: The binary mask to segment regions in the image for intensity analysis.
    :type Mask: numpy.ndarray
    :param df: The DataFrame that is updated with calculated intensity statistics.
    :type df: pandas.DataFrame
    :param i: The index of the row in the DataFrame where the calculated results are stored.
    :type i: int
    :return: The updated DataFrame with added intensity metrics corresponding to the given index.
    :rtype: pandas.DataFrame
    """
    label, num_lab = ski.measure.label(Mask, return_num=True, connectivity=2)
    Prop_int = ski.measure.regionprops(label, intensity_image=im)
    coords = Prop_int[0].coords
    df.at[i, 'Avg_Intensity_in_mask'] = Prop_int[0].intensity_mean
    df.at[i, 'Avg_Intensity_of_im'] = np.mean(im)
    df.at[i, 'Ratio_avg_int_in_vs_tot'] = Prop_int[0].intensity_mean / np.mean(im)
    df.at[i, 'Sum_Intensity_in_Mask'] = im[coords[:, 0], coords[:, 1]].sum()

    return df


# functions used to get the medial axis
def prepare_contour_medial_axis(im, Cont_2):
    """
    Prepares a smoothed mask along the medial axis by processing the input contour and image data.

    :param im: Input image array used for processing.
    :type im: numpy.ndarray
    :param Cont_2: Contour points as a list of coordinates used for smoothing and masking.
    :type Cont_2: list
    :return: A binary mask with smoothed contour and filled holes.
    :rtype: numpy.ndarray
    """
    Mask_smooth = np.zeros_like(im, dtype='bool')
    ####### Smoothing the contourCont to make a smoothed mask, for medial axis
    Im_smooth = ski.filters.gaussian(im, 5, preserve_range=False)

    init = (max(Cont_2, key=len))
    snake = ski.segmentation.active_contour(Im_smooth, init, alpha=0.01, beta=10, gamma=0.5, w_line=0, w_edge=0)

    # Increase the number of points in snake
    Npoints = 20000
    tck, u = interpolate.splprep([snake[:, 0], snake[:, 1]], per=1)
    xi, yi = interpolate.splev(np.linspace(0, 1, Npoints), tck)
    snake_2 = np.array([xi, yi]).transpose()

    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    Mask_smooth[np.round(snake_2[:, 0]).astype('int'), np.round(snake_2[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    Mask_smooth[:, :] = ndi.binary_fill_holes(Mask_smooth[:, :])

    return Mask_smooth

def get_img_contour(im):
    """
    Extracts and returns the contour of an image based on a given intensity threshold.
    The function locates the contours in the input image, processes them to integer
    coordinates, and creates a binary image where the contours are highlighted.

    :param im: ndarray
        The input image from which contours are to be extracted.
    :return: ndarray
        A binary image of the same size as the input, with the extracted contours
        highlighted in white (255) against a black (0) background.
    """
    contours = ski.measure.find_contours(im, 0.8)
    cont = np.squeeze(contours)
    contint = np.array(cont, dtype=int)
    Img_Cont = np.zeros((im.shape[0], im.shape[1]), np.uint8)
    Img_Cont[contint[:, 0], contint[:, 1]] = 255

    return Img_Cont

def get_full_medial_axis(Mask_s):
    """
    Computes the full extended medial axis of a smoothened mask. The function computes the medial axis for
    the given input mask, retrieves its coordinate points, extracts the contour of the input mask, and
    prolongs the medial axis to the gastruloid contour using specified fitting parameters (ratio and
    minimum length). The extended medial axis and intersection indices are then returned as output.

    :param Mask_s: A numpy array representing the binary smoothened mask used for medial axis computation.
    :type Mask_s: numpy.ndarray
    :returns: Skel_full - An extended medial axis prolonged to the contour.
    :rtype: numpy.ndarray
    :returns: ie - Index of the intersection point in the contour coordinates where the medial axis ends.
    :rtype: int
    :returns: ib - Index of the intersection point in the contour coordinates where the medial axis begins.
    :rtype: int
    """
    ## input smoothened mask
    ############################## GET MEDIAL AXIS FOR THE LAST 24 TIME POINTS
    skel, _ = ski.morphology.medial_axis(Mask_s, return_distance=True)
    skel = np.argwhere(skel)  # Get coordinates of medial axis
    Img_Cont = get_img_contour(Mask_s)
    cnt = np.argwhere(Img_Cont)  # Get coordinates of contour of gastruloid

    # Mélody version: Prolong medial axis to the gastruloid contour

    # ratio = percentage of extremities used for the fit
    # lenMin = minimum number of point for the fit (if the percentage gives a lower number, it will use this)
    # output: prolonged medial axis, and ie and ib (index in the contour coordinates of the intersections with medial axis)
    Skel_full, ie, ib = expand_skel_return(skel, cnt, ratio=4, lenMin=10)

    return Skel_full

def get_medial_axis_length(skelfull, df, i):
    """
    Calculate the medial axis length of a skeletonized structure.

    This function computes the cumulative distance (length) of the medial axis
    based on the coordinates of its skeleton pixels. The computation takes into
    account the scaling factor provided in the data frame's 'Pixel_size' column.

    :param skelfull: An array containing the coordinates (x, y) of the skeleton
        pixels. Each element represents the [x, y] coordinates of a single pixel.
    :param df: A pandas DataFrame containing additional metadata. The 'Pixel_size'
        column is accessed for scaling purposes.
    :param i: Index of the specific row in the DataFrame `df` from which
        the 'Pixel_size' value is retrieved.
    :return: The cumulative length of the medial axis, scaled using the 'Pixel_size'
        factor.
    :rtype: float
    """
    Cum_dist = 0
    pxsize = df.at[i, 'Pixel_size']

    for l in range(1, (len(skelfull))):
        x_med = skelfull[l, 0]
        y_med = skelfull[l, 1]
        x_med_prev = skelfull[(l - 1), 0]
        y_med_prev = skelfull[(l - 1), 1]
        Cum_dist = Cum_dist + pxsize * math.sqrt((x_med - x_med_prev) ** 2 + (y_med - y_med_prev) ** 2)

    return Cum_dist


# big function combining the live analysis
def compute_multi_tp_mask_analysis(df, save_path, save=True):
    """
    Analyze multi-time point data for a pandas DataFrame, including shape descriptors
    and medial axis computations. For each time point, processes associated image data
    (if available) by extracting key morphological characteristics and saves the updated
    DataFrame. Handles errors gracefully during the computational steps and provides
    options to save results to a specified path.

    :param df: A pandas DataFrame containing time-point indexed data. Each entry should
        have a path to the associated image or movie file under the 'Path_data' column.
        Additional columns such as 'Name' will be used for logging purposes.
    :type df: pandas.DataFrame
    :param save_path: File path where the updated DataFrame will be serialized. Supported
        format is JSON.
    :type save_path: str
    :param save: A boolean flag indicating if the resulting DataFrame should be saved
        to a file under the specified save_path.
    :type save: bool
    :return: Updated pandas DataFrame containing computed shape descriptors and medial
        axis information for each time point.
    :rtype: pandas.DataFrame
    """
    # starting the analysis for every time point in the df
    for j, i in enumerate(df.index):
        
        if os.path.isfile(df.at[i, 'Path_data']):
            image = ski.io.imread(df.at[i, 'Path_data'], plugin='tifffile')[:,:,0]
        
            try:
                Mask_init, Cont = initialize_time_pt_image(image, df, i)
                df = get_shape_descriptors(Mask_init, i, df)
                Mask_smooth = prepare_contour_medial_axis(image, Cont)
    
                try:
                    Skel_full = get_full_medial_axis(Mask_smooth)
                    Cum_dist = get_medial_axis_length(Skel_full, df, i)
                except Exception as e:
                    Cum_dist = np.nan
                    Skel_full = np.nan
                    print(e, 'Problem for Medial Axis at time ' + str(j))
    
                df = df.astype({"MedialAxis": object})
                df.at[j, 'MedialAxis'] = [Skel_full]
                df.at[j, 'MA_length_um'] = Cum_dist

            except:
                print('Problem with image ' + df.at[i, 'Name'])

        else:
            print('No image or movie file could be found for: ', df.at[i, 'Path_data'])

    # save df that now contains shape and intensity information
    if save:
        df.to_json(save_path, orient='split')
        print(f'Data saved under ', save_path)

    return df


def compute_live_movie_mask_analysis(df, save_path, save=True):
    """
    Process and analyze live movie image data to extract shape descriptors, intensity
    information, and medial axis data. The function iterates over all time points in
    the provided data frame, performs analysis, and updates the data frame with
    extracted information. The processed data can optionally be saved to a specified path.

    :param df: A pandas DataFrame containing image analysis metadata. Expected to
               include columns for paths to movie and other necessary information
               at different time points.
    :type df: pandas.DataFrame
    :param save_path: The file path where the updated data frame should be saved, if
                      saving is enabled.
    :type save_path: str
    :param save: A boolean flag indicating whether or not to save the updated data
                 frame to the specified `save_path`. Defaults to True.
    :type save: bool
    :return: Updated pandas DataFrame containing additional columns for shape
             descriptors, intensity information, and medial axis analysis.
    :rtype: pandas.DataFrame
    """
    # check whether movie exists in path, otherwise skip analysis
    mov_path = df['Path_data'].unique()[0] + df['Name'].unique()[0]
    if os.path.isfile(mov_path):
        movie = read_movie(mov_path)
        image = movie[:, :, :, 0]

        # starting the analysis for every time point in the df
        for j in range(len(df)):
            try:
                im, Mask_init, Cont = initialize_time_pt(image, df, j)
                df = get_shape_descriptors(Mask_init, j, df)
                df = get_info_intensity(im, Mask_init, df, j)
                Mask_smooth = prepare_contour_medial_axis(im, Cont)

                try:
                    Skel_full = get_full_medial_axis(Mask_smooth)
                    Cum_dist = get_medial_axis_length(Skel_full, df, j)
                except Exception as e:
                    Cum_dist = np.nan
                    Skel_full = np.nan
                    print(e, 'Problem for Medial Axis at time ' + str(j))

                df = df.astype({"MedialAxis": object})
                df.at[j, 'MedialAxis'] = [Skel_full]
                df.at[j, 'MA_length_um'] = Cum_dist

            except:
                print('Problem at time ' + str(j))

    else:
        print('No image or movie file could be found for: ', mov_path)


    # save df that now contains shape and intensity information
    if save:
        df.to_json(save_path, orient='split')
        print(f'Data saved under ', save_path)

    return df

def compute_cumulated_live_movie_mask_analysis(df_list, df_save_list=None, save=True):
    """
    Computes and processes cumulated live movie mask analysis for a list of dataframes.

    This function takes a list of dataframe file paths, processes each dataframe
    to ensure data consistency, and subsequently performs live movie mask analysis.
    The processed data can be saved optionally to specified paths.

    :param df_list: The list of file paths for the input dataframes to be processed.
    :type df_list: list[str]
    :param df_save_list: Optional list of file paths where processed dataframes will
        be saved. If not provided, the original paths will be used to save data.
    :type df_save_list: list[str], optional
    :param save: Flag to indicate whether the processed dataframes should be saved
        to disk. Defaults to True.
    :type save: bool
    :return: None
    """
    for i in range(len(df_list)):

        # reading in the df
        df_path = df_list[i]
        df_save_path = df_path
        df = load_processed_df(df_path)

        # some code I had to implement because the data was opened/saved in weird format,
        try:
            df.index = [int(i) for i in df.index]
            df['Pixel_size'] = df['Pixel_size'].astype('float')

        except Exception as e:
            print(e)

        if (save and df_save_list):
                df_save_path = df_save_list[i]

        compute_live_movie_mask_analysis(df, df_save_path, save=save)


# functions for fluorecent movie analysis
def filter_fluo_img(img, k):
    """
    Apply a median filter on the given image for noise reduction.

    This function performs a median filtering operation on the input image using
    a square kernel of size `k x k`. Median filtering is commonly used to reduce
    noise in images while preserving edges. The filter computes the median of
    the neighboring pixels within the kernel for each pixel in the image.

    :param img: The input image to be filtered, provided as a 2D numpy array.
    :param k: The size of the square kernel to be applied for filtering.
    :return: The filtered image as a 2D numpy array after the median filter is applied.

    """
    kernel = np.zeros((k, k)) + 1
    smooth = ski.filters.rank.median(img, footprint=kernel)

    return smooth

def get_filtered_fluo_movie_mesp(movie, ch, k):
    """
    Processes a 4D movie array to filter the fluorescence images channel-wise.

    This function extracts a specified channel from a given 4D movie array, applies a fluorescence image
    filter to each 2D image in the time dimension of the channel, and then returns the filtered 4D movie
    array. Each frame in the specified channel is modified in-place after processing.

    :param movie: A 4D array containing fluorescence image data in the form of
                  (time, height, width, channels) for a multi-dimensional movie.
    :param ch: int
        The channel index to filter within the movie.
    :param k: float
        The parameter controlling the behavior of the filter applied to
        each 2D frame in the specified channel.
    :return: The filtered 4D movie array with the same dimensions as the input movie.
    :rtype: numpy.ndarray
    """
    fluo_movie = movie[:, :, :, ch]
    for t in range(fluo_movie.shape[0]):
        fluo_img = fluo_movie[t, :, :]
        fluo_movie[t, :, :] = filter_fluo_img(fluo_img, k)

    return fluo_movie

def compute_img_threshold(img):
    """
    Compute threshold value for an image using Otsu's method.

    This function calculates the optimal threshold value for a given image
    using Otsu's method from skimage.filters module. Otsu's method finds the
    threshold that minimizes the intra-class variance, making it suitable
    for segmenting grayscale images into binary states (foreground and
    background).

    :param img: Grayscale input image for which the threshold value is to be
        calculated.
    :type img: numpy.ndarray
    :return: Computed threshold value for the input image.
    :rtype: float
    """
    th = ski.filters.threshold_otsu(img)

    return th

def compute_movie_threshold(movie, metric=0):
    """
    Computes a threshold value for the input movie based on a specified metric.

    This function processes each frame of the input movie to calculate a threshold
    value for fluorescence intensity using the `compute_img_threshold` function.
    The function subsequently combines these thresholds across all frames using
    either the median or mean, depending on the specified metric.

    :param movie: A 3D NumPy array representing the movie data, where the first
                  dimension corresponds to time (frames), and the other two dimensions
                  are spatial dimensions for each frame.
    :param metric: An integer specifying the method to compute the final threshold.
                   If 0, the median of the thresholds from all frames is used.
                   If 1, the mean of the thresholds from all frames is used.
                   Default is 0.
    :return: The computed threshold value, either the median or mean of the per-frame
             thresholds based on the provided metric.
    :rtype: float
    """
    warnings.simplefilter('ignore')
    Nt = movie.shape[0]
    ths = list()

    for t in range(Nt):
        fluo = movie[t, :, :]
        ths.append(compute_img_threshold(fluo))

    movie_th = 0
    if metric == 0:
        movie_th = np.median(ths)
    elif metric == 1:
        movie_th = np.mean(ths)

    return movie_th

def snake_to_bw(snake,shape):
    """
    Convert a sequence of coordinates in a snake to a binary image with specified shape.

    The function takes a list or array representing the coordinates of the snake,
    draws filled circles at each coordinate position on a blank binary image. Holes
    within the binary result are then filled to ensure the shape is represented as
    a continuous region.

    :param snake: A sequence of 2D coordinates representing the snake's positions. Each
                  coordinate is expected to be a list or array-like [y, x].
    :param shape: The shape of the binary output image. Must be a tuple of two integers,
                  specifying (height, width).
    :return: A binary image of the specified shape with the snake positions marked
             and filled as a continuous region.
    :rtype: numpy.ndarray
    """

    bw_snake = np.zeros(shape)
    snake = np.int0(snake)
    for i in range(np.size(snake,0)):
        cv2.circle(bw_snake,(snake[i,1], snake[i,0]), radius=2, color=(255,255,255), thickness=-1)

    bw_snake = binary_fill_holes(bw_snake)

    return bw_snake

def bw_to_snake(bw):
    """
    Converts a given binary (black and white) image to a snake representation
    using image contours. A snake representation is essentially a collection of
    points outlining the shape from the contours of the provided input image.

    :param bw: Binary (black and white) image represented as a 2D numpy array.
               The input matrix should contain binary values (0 and 255 or
               similar) where the contours of the object are defined.
    :return: An array of snakes (contours), where the snake is an array of x and
             y coordinates that define the contour of the object. If more than one
             contour is found, only the first one is considered and reshaped.
    :rtype: numpy.ndarray
    """
    snake = ski.measure.find_contours(bw, 0.5)
    if len(snake) > 1:
        snake = np.array(snake[0])
        snake.reshape((snake.shape[0], snake.shape[1]))
    else:
        snake = np.array(snake)
        snake = snake.reshape((snake.shape[1], snake.shape[2]))

    return snake


def get_fluorescent_area(fluo, th=150):
    """
    Computes fluorescent areas in the given image using a threshold and several
    morphological transformations. The method performs binarization, reconstruction,
    cleanup of small objects, erosion, and labeling on the input fluorescence image.
    This method identifies and labels distinct regions of fluorescent areas in the
    image.

    :param fluo: 2D numpy array of grayscale fluorescence image to be processed.
    :param th: Integer threshold value (default is 150) to binarize the image.
    :return: Tuple containing:
        - 2D numpy array where each distinct fluorescent region is labeled
          with a unique integer.
        - 1D numpy array of unique labels representing detected regions.
    """
    binary = (fluo > th).astype(bool)
    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary
    filled = ski.morphology.reconstruction(seed, mask, method='erosion')
    label = ski.measure.label(filled)

    # specify disk size, the larger the longer the computation
    cleared = ski.morphology.remove_small_objects(label, 5000)
    eroded = ski.morphology.binary_erosion(cleared, footprint=ski.morphology.disk(10))
    label = ski.measure.label(eroded)
    
    # labels per identified fluor filled area
    labels = np.unique(label)

    return label, labels


def get_fluo_shape_descriptors(Mask, l, i, df):
    """
    Analyzes a binary mask to compute and store fluorescence shape descriptors for a given object
    and updates the provided DataFrame with calculated properties. Includes geometric metrics like
    solidity, perimeter, axes, area, and circularity.

    :param Mask: A binary mask image used for object analysis.
    :type Mask: numpy.ndarray
    :param l: Index or key used to access nested data in the DataFrame.
    :type l: int or str
    :param i: Row index in the DataFrame corresponding to the current object.
    :type i: int
    :param df: DataFrame containing data, which will be updated with the calculated shape descriptors.
    :type df: pandas.DataFrame
    :return: Updated DataFrame after adding the fluorescence shape descriptors.
    :rtype: pandas.DataFrame
    """
    Prop = ski.measure.regionprops(Mask)
    pxsize = df.at[i, 'Pixel_size']
    try:
        df.at[i, 'Mask_fluo'][l] = [bw_to_snake(Mask).tolist()]
        df.at[i, 'Solidity_fluo'][l] = Prop[0].solidity
        df.at[i, 'Perimeter_fluo'][l] = pxsize * Prop[0].perimeter
        df.at[i, 'AR_fluo'][l] = Prop[0].axis_major_length / Prop[0].axis_minor_length
        df.at[i, 'Major_Axis_um_fluo'][l] = [pxsize * Prop[0].axis_major_length]
        df.at[i, 'Area_fluo'][l] = pxsize * pxsize * Prop[0].area
        df.at[i, 'Circularity_fluo'][l] = 4 * math.pi * Prop[0].area / (Prop[0].perimeter * Prop[0].perimeter)
    except:
        print('something went wrong')
    return df


# big function combining the multipol analysis
def compute_multipol_analysis_per_df(fluo_movie, th_fluo, df_fluo):
    """
    Computes the fluorescence multipolarity analysis for the given dataframe. This
    function processes each frame of a fluorescence movie, applies necessary
    masking and filtering operations, performs fluorescence analysis, and updates
    the given dataframe with calculated results such as frame dimensions, centroid
    positions, number of fluorescence poles, and their coordinates.

    :param fluo_movie: 3D numpy array representing the fluorescence movie where
                       each frame is a 2D array.
    :type fluo_movie: numpy.ndarray

    :param th_fluo: Threshold value used for identifying fluorescent areas in the
                    frame.
    :type th_fluo: float

    :param df_fluo: Pandas dataframe containing the input data where each row
                    corresponds to analysis details for a specific frame. This
                    dataframe will be updated with calculated results during the
                    processing.
    :type df_fluo: pandas.DataFrame

    :return: Updated dataframe containing the results of fluorescence multipolarity
             analysis for each frame.
    :rtype: pandas.DataFrame
    """
    for t in df_fluo.index:

        frame_fluo = fluo_movie[t, :, :]
        df_fluo.at[t, "Frame_dim"] = np.array(fluo_movie[t, :, :].shape).astype(object)

        try:
            bf_snake = df_fluo.at[t, 'Mask'][0]
            if len(bf_snake) > 100:
                bf_mask = snake_to_bw(bf_snake, frame_fluo.shape)
                frame_fluo = frame_fluo * bf_mask
       
                Mask = bf_mask*np.ones_like(frame_fluo)
                Prop = ski.measure.regionprops(Mask)
                df_fluo.loc[[t], 'centroid'] = pd.Series([Prop[0].centroid], index=df_fluo.index[[t]])
                
        except:
            print(t, 'no filtering of contour')

        ### start analysis of fluorescence
        cleared, labels = get_fluorescent_area(frame_fluo, th=th_fluo)
        eroded = ski.morphology.binary_erosion(cleared, footprint=ski.morphology.disk(2))

        frame_fluo = ski.filters.gaussian(frame_fluo, sigma=20)
        frame_fluo2 = frame_fluo * eroded
        coords = peak_local_max(frame_fluo2, min_distance=100)
        
        df_fluo.at[t, 'n_fluo_poles'] = len(coords)
        df_fluo.loc[[t], 'fluo_pole_coords'] = pd.Series([coords], index=df_fluo.index[[t]])


    return df_fluo




def compute_multipol_analysis_per_condition(df_list, movie_list, ch_fluo=1, k=7, th=None):
    """
    Computes a multipolarity analysis per condition by processing provided dataframes and movies, applies
    fluorescence thresholding, performs analysis, and saves the processed data back.

    This function iterates over a list of dataframe file paths (`df_list`) and correlates each dataframe
    with a specific movie from `movie_list` by its identifier. Fluorescence intensities are processed to
    identify multipolar configurations. Processed data is saved to the original dataframe files in JSON format.

    :param df_list: List of file paths to the dataframes to be processed.
    :type df_list: list of str
    :param movie_list: List of file paths to the movie files to be analyzed.
    :type movie_list: list of str
    :param ch_fluo: Channel index in the fluorescence images (default is 1).
    :type ch_fluo: int, optional
    :param k: Radius for filtering in fluorescence movie pre-processing (default is set to 7).
    :type k: int, optional
    :param th: Fluorescence threshold to be applied during analysis. If not provided, it is computed automatically.
    :type th: int or None, optional
    :return: This function does not return a value. The processed data is directly saved to the respective
             dataframe files.
    :rtype: None
    """
    # load list of paths to dfs
    for df_path in tqdm(df_list):
        # load df
        print(df_path)
        df = load_processed_df(df_path)
        df.index = [int(i) for i in df.index]
        df['Pixel_size'] = df['Pixel_size'].astype('float')
        df['centroid'] = np.nan
        df['n_fluo_poles'] = np.nan
        df['fluo_pole_coords'] = np.nan

        # load movie
        movie_id = df_path.rsplit('/', 1)[1].split('Analysis_SAM')[0]
        movie_path = [path for path in movie_list if movie_id in path]
        print(movie_path)
        movie = read_movie(movie_path)
        
        # multipolarity analysis
        fluo_movie = get_filtered_fluo_movie_mesp(movie, ch=ch_fluo, k=k)
        
        if th:
            th_fluo = th
        else:
            th_fluo = compute_movie_threshold(fluo_movie, metric=0)
        
        df_fluo = compute_multipol_analysis_per_df(fluo_movie, th_fluo, df)
        print(df_fluo)
        # save data
        
        df_fluo.to_json(df_path, orient='split')
        print('Analysis completed and saved under \n', df_path, '\n')


