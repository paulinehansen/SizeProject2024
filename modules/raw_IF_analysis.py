import os
import cv2
import sys
import math
import tifffile
import numpy as np
import pandas as pd
import skimage as ski
import networkx as nx

from scipy.ndimage import binary_fill_holes
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)

import bioformats
import modules.javaThings as jv
import modules.configs_setup as configs

jv.start_jvm()
jv.init_logger()


# functions working with czi files
def GetMetadata(filename):
    """
    Extracts and processes metadata from a given microscopy image file using the Bio-Formats library.

    This function reads the metadata of a microscopy image file and organizes it into a dictionary.
    The returned metadata dictionary contains various image-related attributes, such as image dimensions,
    number of channels, physical sizes, and pixel type. Additionally, it attempts to retrieve
    the nominal magnification and the number of series from the image metadata. Missing or unavailable
    attributes are replaced with fallback values, where applicable.

    :param filename: File path of the microscopy image whose metadata needs to be extracted.
    :type filename: str

    :return: A dictionary containing metadata of the microscopy image, including attributes like:
        - 'Nx': Number of pixels in the X dimension.
        - 'Ny': Number of pixels in the Y dimension.
        - 'Nz': Number of pixels in the Z dimension.
        - 'Nt': Number of timepoints.
        - 'Nch': Number of channels.
        - 'dtype_str': Data type used for pixel values.
        - 'dx': Physical size of a pixel in the X dimension.
        - 'dxUnit': Measurement unit of the physical size in the X dimension.
        - 'dy': Physical size of a pixel in the Y dimension.
        - 'dyUnit': Measurement unit of the physical size in the Y dimension.
        - 'dz': Physical size of a voxel in the Z dimension.
        - 'dzUnit': Measurement unit of the physical size in the Z dimension.
        - 'Nseries': Number of image series available in the file.
        - 'Magn': Nominal magnification (if available).
    :rtype: dict
    """
    md              = bioformats.get_omexml_metadata(filename)
    metadata = {
        #         "fullmetadata" : md,
        "Nx"        : bioformats.OMEXML(md).image().Pixels.SizeX,
        "Ny"        : bioformats.OMEXML(md).image().Pixels.SizeY,
        "Nz"        : bioformats.OMEXML(md).image().Pixels.SizeZ,
        "Nt"        : bioformats.OMEXML(md).image().Pixels.SizeT,
        "Nch"       : bioformats.OMEXML(md).image().Pixels.SizeC,
        "dtype_str" : bioformats.OMEXML(md).image().Pixels.PixelType,
        "dx"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeX,
        "dxUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeXUnit,
        "dy"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeY,
        "dyUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeYUnit,
        "dz"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeZ,
        "dzUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeZUnit,
        "Nseries"   : bioformats.ImageReader(filename).rdr.getSeriesCount()
    }

    try:
        metadata["Magn"] = bioformats.OMEXML(md).instrument().Objective.NominalMagnification
    except:
        print('no magn found')
    try:
        metadata["Nseries"] = bioformats.ImageReader(filename).rdr.getSeriesCount()
    except:
        metadata['Nseries'] = 1
        print('#series not found')

    return metadata

def get_middleslice(filepath):
    """
    Gets the middle slice of the Z-stack from a multi-dimensional image file.

    This function reads a Z-stack image file and extracts the middle slice
    along the Z-dimension. It uses bioformats for image reading and retrieves the
    necessary metadata to compute the location of the middle slice.

    :param filepath: The file path to the image file. Should be a valid path to
        a Z-stack image.
    :type filepath: str
    :return: The middle slice of the Z-stack image.
    :rtype: numpy.ndarray
    """
    reader = bioformats.ImageReader(filepath)
    metadata = GetMetadata(filepath)
    image = reader.read(rescale=False, z=np.floor(metadata['Nz'] / 2), c=0)

    return image

def maxprojection(czifile, theS=0, theC=0, theT=0):
    """
    Generates a maximum intensity projection across z-dimensions from a given
    CZI (Carl Zeiss Image) file. The function reads the image data for the
    specified series, channel, and timepoint, and calculates the maximum
    value at each pixel over all z-planes. Additionally, it records the z-plane
    index at which the maximum value occurs.

    :param czifile: Path to the CZI file containing image data.
    :type czifile: str
    :param theS: Series index in the CZI file to be processed. Default is 0.
    :type theS: int, optional
    :param theC: Channel index in the series to be processed. Default is 0.
    :type theC: int, optional
    :param theT: Timepoint index in the series to be processed. Default is 0.
    :type theT: int, optional
    :return: A tuple containing the maximum intensity projection and the
        corresponding z-plane index map.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    print("sum projection!")

    reader = bioformats.ImageReader(czifile)
    metadata = GetMetadata(czifile)
    Nz = metadata["Nz"]

    I = reader.read(rescale=False, z=0, c=theC, series=theS, t=theT)

    maxi = np.copy(I)
    indz = np.zeros_like(I)

    for theZ in range(1, Nz):
        I = reader.read(rescale=False, z=theZ, c=theC, series=theS)
        inds = I > maxi
        indz[inds] = theZ
        maxi[inds] = I[inds]  # update the maximum value at each pixel

    return maxi, indz



# functions used for segmentation for contouring
def binarize(image):
    """
    Binarizes an input image using several processing steps involving normalization, gamma
    adjustment, Sobel filter application, Gaussian filtering, Otsu thresholding, and image
    reconstruction. The function processes the input image to create a binary filled
    version, emphasizing specific features of the input image.

    :param image: Input image to be processed.
    :type image: numpy.ndarray

    :return: Binary filled version of the processed image.
    :rtype: numpy.ndarray
    """
    #imeq = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = ski.exposure.adjust_gamma(imeq, 0.75)
    image = ski.filters.sobel(image)
    img = ski.filters.gaussian(image, sigma=5)
    val = ski.filters.threshold_otsu(img)
    binary = np.zeros_like(img, dtype=int)
    binary[img > val] = 1

    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary
    filled = ski.morphology.reconstruction(seed, mask, method='erosion')

    def f(x):
        return int(x)
    f2 = np.vectorize(f)
    filled2 = f2(filled)

    return filled2

def binarize_channel(image):
    """
    Binarizes a single channel of an image using Otsu's thresholding method.

    This function normalizes the input image to a range of 0 to 1, computes
    the Otsu threshold, and produces a binarized version of the image based
    on the threshold. The binarized image contains only 0s and 1s, where 1s represent
    pixels above the threshold.

    :param image: A 2D array representing a single-channel image to be binarized.
    :type image: numpy.ndarray
    :return: A tuple containing the binarized image and the Otsu threshold value.
             The binarized image is a 2D array with the same shape as the input, made up
             of 0s and 1s, and the threshold is a float value representing the computed
             Otsu's threshold.
    :rtype: tuple[numpy.ndarray, float]
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    val = ski.filters.threshold_otsu(image)
    binary = np.zeros_like(image, dtype=int)
    binary[image > val] = 1

    return binary, val

def clean_binarized(filled):
    """
    Cleans a binarized image by applying morphological operations. This process includes
    labeling connected regions, erosion, removal of small objects, dilation, and clearing
    of borders to refine the binary image.

    :param filled: Binary input image as a NumPy array. Nonzero values are treated as
        foreground, and zero values are treated as background.
    :type filled: numpy.ndarray

    :return: A cleaned binary image with refined regions and removed small artifacts.
    :rtype: numpy.ndarray
    """
    labels = ski.measure.label(filled)
    # specify disk size, the larger the longer the computation
    eroded = ski.morphology.binary_erosion(labels, footprint=ski.morphology.disk(15))
    # specifiy max size of small objects
    cleared = ski.morphology.remove_small_objects(eroded, 5000)
    dilated = ski.morphology.binary_dilation(cleared, footprint=ski.morphology.disk(15))
    img = np.zeros_like(filled)
    img[dilated[:]] = 1
    img = ski.segmentation.clear_border(img)

    return img

def bw_to_snake(bw):
    """
    Converts a binary image (black and white) into a snake representation.

    This function takes a binary image as input and finds the contours using
    image processing techniques. The contours are then reshaped to form a
    representation of the snake.

    :param bw: The binary image (black and white) to process.
    :type bw: numpy.ndarray
    :return: A numpy array representing the snake as a set of reshaped contours.
    :rtype: numpy.ndarray
    """

    snake = np.array(ski.measure.find_contours(bw, 0.5))
    snake = snake.reshape((snake.shape[1], snake.shape[2]))

    return snake

def snake_to_bw(snake, shape):
    """
    Converts a given snake (list of coordinates) into a binary black-and-white image
    of the specified shape. Each point in the snake is marked by a filled circle
    with a radius of 2. The resulting binary image will have the snake filled with
    white pixels (value 255), and the rest will have black pixels (value 0). Holes
    inside the snake's contour are filled.

    :param snake: The sequence of 2D coordinates that form the snake.
        Each coordinate is a pair of integers representing (row, column).
    :param shape: The shape of the output binary image as a tuple of two integers
        (height, width), specifying the number of rows and columns respectively.
    :return: A 2D binary image of the specified shape. The image is black (0)
        by default with the snake represented in white (255).
    :rtype: numpy.ndarray
    """

    bw_snake = np.zeros(shape)

    def f(x):
        return int(x)
    f2 = np.vectorize(f)
    snake = f2(snake)

    for i in range(np.size(snake, 0)):
        cv2.circle(bw_snake, (snake[i, 1], snake[i, 0]), radius=2, color=(255, 255, 255), thickness=-1)

    bw_snake = binary_fill_holes(bw_snake)

    return bw_snake

def get_contour(binary, image, alpha=0.1, beta=20, w_line=0, w_edge=0):
    """
    Finds the contour of a binary object in an image using active contour model.

    The function initializes the snake contour from the binary input image, then
    applies the active contour model from scikit-image on the given image to
    compute the final contour. This is often used for edge detection and extracting
    the contours of objects in images.

    :param binary: A binary image that is used to initialize the snake.
    :param image: The grayscale or color input image on which active contouring will
                  be applied.
    :param alpha: A float controlling the tension of the snake. Defaults to 0.1.
    :param beta: A float that controls the rigidity of the snake. Defaults to 20.
    :param w_line: A float that weights the attraction to intensity values. Defaults to 0.
    :param w_edge: A float that weights the attraction to features in the image.
                   Defaults to 0.
    :return: A numpy array of shape `(N, 2)` representing the final coordinates of
             the snake's contour.
    """
    snake_init = bw_to_snake(binary)
    snake = ski.segmentation.active_contour(image, snake_init, alpha=alpha, beta=beta, w_line=w_line, w_edge=w_edge)
    print('active contour done')

    return snake


# functions used to get the medial axis
def find_longest_edge(l, T):
    """
    Find the longest edge in a triangle represented in a weighted graph.

    This function determines the longest edge in a triangle within a given graph.
    The triangle is represented by three vertices, and the graph is represented as
    an adjacency dictionary where the weights of the edges are stored. The function
    compares the weights of the three edges in the triangle and returns the pair of
    vertices that form the longest edge.

    :param l: List of vertices that form the triangle. Should be of length 3.
    :type l: list
    :param T: Graph represented as an adjacency dictionary. The dictionary associates
              vertex pairs with a dictionary, containing the weight of the edge.
    :type T: dict
    :return: A tuple representing the two vertices that form the edge with the highest weight.
    :rtype: tuple
    """
    e1 = T[l[0]][l[1]]['weight']
    e2 = T[l[0]][l[2]]['weight']
    e3 = T[l[1]][l[2]]['weight']

    if e2 < e1 > e3:
        return (l[0], l[1])
    elif e1 < e2 > e3:
        return (l[0], l[2])
    elif e1 < e3 > e2:
        return (l[1], l[2])

def orderPoints(C_skel):
    """
    Order points in the given skeleton representation of points to minimize the sum of Euclidean distances
    in a traversal path. The method employs a graph-based approach to calculate the optimal order of points,
    removing excessive edges, and exploring possible traversal paths to find the minimal cost route.

    :param C_skel: Array of shape (n, d), where n is the number of points and d is the dimensionality of
        each point. Represents the skeleton points to be ordered.
    :type C_skel: numpy.ndarray

    :return: A tuple containing:
        - C_skel_O (numpy.ndarray): Re-ordered array of skeleton points according to the minimal cost path.
        - T (networkx.Graph): Graph representation after reducing edges based on traversal optimization.
    :rtype: tuple
    """

    G = kneighbors_graph(C_skel, 2, mode='distance')
    T = nx.from_scipy_sparse_array(G)

    end_cliques = [i for i in list(nx.find_cliques(T)) if len(i) == 3]
    edge_lengths = [find_longest_edge(i, T) for i in end_cliques]
    T.remove_edges_from(edge_lengths)

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(C_skel))]
    mindist = np.inf
    minidx = 0

    lenp = []
    costs = []
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

def interpol(x,y):
    """
    Computes the parameters of a line in 2D space defined by the given points (x, y).
    The output is a list containing three elements representing the equation of the
    line in the general form: m*x + n*y + p = 0. If the points define a vertical line,
    special handling ensures a proper result.

    :param x: List or array-like of x-coordinates of the points
    :type x: list or numpy.ndarray
    :param y: List or array-like of y-coordinates of the points
    :type y: list or numpy.ndarray
    :return: A list containing [m, n, p] where m, n, p are coefficients of the line
    :rtype: list
    """
    dx = np.sum(np.diff(x))
    dy = np.sum(np.diff(y))

    if dx == 0:
        m = 1
        n = 0
        p = -np.mean(x)
    else:
        m = dy/dx
        n = -1
        p = np.mean(y) - m*np.mean(x)

    return [m, n, p]

def interpolateEnds(C_skel, ratio=3, lenMin=2):
    """
    Determines interpolation functions for the beginning and end segments of a given skeleton curve
    based on a specified ratio and minimum segment length. The function analyzes the positions
    of the skeleton curve coordinates and divides the curve into the beginning and end regions,
    which are then interpolated using a defined interpolation method.

    :param C_skel: 2D numpy array representing the skeleton curve. Each row corresponds to a point,
                   where the first column represents x-coordinates, and the second column represents
                   y-coordinates.
    :param ratio: Integer representing the percentage ratio used to define the separation
                  point for the beginning and end segments of the skeleton curve. Defaults to 3.
    :param lenMin: Integer specifying the minimum length of the segment (in terms of the number
                   of points) to ensure valid interpolation. Defaults to 2.
    :return: A tuple containing two interpolation functions: one for the beginning segment and
             one for the end segment.
    """
    xSkel, ySkel = C_skel.T
    if len(xSkel) < 2*lenMin+1:
        indBeg = np.min([lenMin, len(xSkel)])
        indEnd = np.min([0, np.abs(len(xSkel)- lenMin)])
    else:
        indBeg = np.max([int(np.floor(len(ySkel)*ratio/100)), lenMin])
        indEnd = np.min([int(np.ceil(len(ySkel)*(100-ratio)/100)), len(xSkel) - lenMin])

    xBeg = xSkel[:indBeg]
    yBeg = ySkel[:indBeg]

    xEnd = xSkel[indEnd:]
    yEnd = ySkel[indEnd:]

    fB = interpol(xBeg, yBeg)
    fE = interpol(xEnd, yEnd)

    return fB, fE

def intersect(C_cnt, f, P_skel, no=[]):
    """
    Calculate the intersection point of contours with a given line, while applying
    specific constraints.

    This function determines the intersection of a set of 2D contour points
    with a specified line equation, represented by its coefficients. It uses
    clique detection on a graph representation of nearby points to determine
    candidates and applies constraints to filter out certain points before
    selecting the one nearest to the given skeleton point.

    :param C_cnt: The contour data points, provided as an (N, 2) array-like object, where
                  each row represents an (x, y) position of a contour point.
    :param f: The coefficients of the line equation in the form (m, n, p), where
              `m*x + n*y + p = 0` represents the line.
    :param P_skel: The skeleton coordinates as a tuple or array-like object containing
                   (x_skel, y_skel), used to find the closest contour intersection
                   point to this skeleton position.
    :param no: Optional list of indices to be excluded from consideration for the
               intersection. These indices represent points that should not be part
               of the result, defaulting to an empty list.
    :return: Index of the contour point that represents the intersection closest
             to the specified skeleton location.
    :rtype: int
    """
    m,n,p = f

    xCnt,yCnt = C_cnt.T
    finter = lambda x,y: m*x+n*y+p
    res = [ finter(xCnt[i], yCnt[i]) for  i in range(len(xCnt))]
    ind = np.argwhere(np.abs(res)< np.sqrt(2)*4)

    test_cnt = np.c_[xCnt[ind], yCnt[ind]]

    G = radius_neighbors_graph(test_cnt, radius =10)
    T = nx.from_scipy_sparse_array(G)
    end_cliques =[i for i in list(nx.find_cliques(T))]

    I =[]
    for i, a in enumerate(end_cliques):
        A = [ind[a[j]][0] for j in range(len(a))]
        resA = [res[j] for j in A]
        if all(a not in no for a in A):
            I.append(A[np.argmin(np.abs(resA))])

    m = [np.sum((xCnt[a] - P_skel[0])**2 + (yCnt[a] - P_skel[1])**2) for a in I]
    intersectI = I[np.argmin(m)]

    return intersectI

def appendMedialAxis( C_skel, C_endpoint, order = 'after'):
    """
    Appends a medial axis to an existing skeleton based on a given endpoint, either before
    or after the current skeleton points, depending on the specified order.

    The function calculates a linear or near-linear interpolation between the endpoint
    and the closest skeleton point to extend the skeleton in the specified direction.
    The primary direction of extension is determined by assessing the largest absolute
    difference in coordinates (x or y).

    :param C_skel:
        A 2D array (numpy array) of the current skeleton points, where each point is
        represented as a row [x, y].

    :param C_endpoint:
        An array-like object representing the coordinates [x, y] of the endpoint from which
        the medial axis will be appended to the skeleton.

    :param order:
        A string specifying the direction of appending ('before' or 'after' the existing
        skeleton points). Default is 'after'.

    :return:
        A numpy 2D array representing the updated skeleton, including the appended medial axis.
        Each row is a point [x, y] in the extended skeleton.
    """
    xSkel, ySkel = C_skel.T
    if order == 'after':
        ind = -1
    elif order == 'before':
        ind = 0
    mainDir = np.argmax([np.abs(xSkel[ind]-C_endpoint[0]), np.abs(ySkel[ind]-C_endpoint[1])])

    if mainDir == 0:
        xa = np.arange(C_endpoint[0], xSkel[ind], 0.8*np.sign(xSkel[ind]-C_endpoint[0]))
        ya = np.linspace(C_endpoint[1], ySkel[ind], num=len(xa))
    elif mainDir == 1:
        ya = np.arange(C_endpoint[1], ySkel[ind], 0.8*np.sign(ySkel[ind]-C_endpoint[1]))
        xa = np.linspace(C_endpoint[0], xSkel[ind], num=len(ya))

    if order == 'after':
        xMA = np.concatenate((xSkel[:-1], np.flip(xa)), axis=0)
        yMA = np.concatenate((ySkel[:-1], np.flip(ya)), axis=0)
    elif order == 'before':
        xMA = np.concatenate((xa, xSkel[1:]), axis=0)
        yMA = np.concatenate((ya, ySkel[1:]), axis=0)

    C_axis = np.c_[xMA, yMA]

    return C_axis

def expand_skel_return(C_skel, C_cnt, ratio=3, lenMin=2):
    """
    Expand and adjust the skeleton representation by appending the medial axis and
    interpolating its endpoints based on the provided ratio and minimum length.
    The function is primarily intended for modifying skeletal geometry by extending
    connections to a given contour.

    :param C_skel: A list of points representing the skeleton.
    :param C_cnt: A list of points defining the contour.
    :param ratio: Ratio used for interpolating skeleton endpoints.
    :param lenMin: Minimum length threshold for extensions.
    :return: A tuple consisting of the expanded skeleton (`C_axis`) and the indices
        of intersection points (`Ib`, `Ie`) on the contour; `Ib` being the start
        intersection and `Ie` being the end intersection.
    """
    if len(C_skel)>2:
        C_skel, _ = orderPoints(C_skel)
    fb,fe = interpolateEnds(C_skel, ratio, lenMin)
    Ib = intersect(C_cnt, fb, C_skel[0])
    if fb==fe:
        Ie = intersect(C_cnt, fe, C_skel[-1], no=[Ib])
    else:
        Ie = intersect(C_cnt, fe, C_skel[-1])

    C_axis = appendMedialAxis(C_skel, C_cnt[Ib], order='before')
    C_axis = appendMedialAxis(C_axis, C_cnt[Ie], order='after')
    if len(C_axis)>3:
        C_axis, _ = orderPoints(C_axis)
    return C_axis, Ib, Ie

def path_cost(G, path):
    """
    Compute the total cost of a given path in a weighted graph. The cost is
    calculated as the sum of the edge weights between consecutive nodes in the
    path.

    :param G: A graph represented as a networkx graph-like object. The graph
        must have each edge annotated with a 'weight' attribute.
    :type G: networkx.Graph
    :param path: A list of nodes representing the path. The path must be valid
        with respect to the given graph `G`.
    :type path: list
    :return: The total cost of the path as the sum of edge weights.
    :rtype: float
    """
    return sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])

def get_length_return(C_axis):
    """
    Computes the length of the path and returns the corresponding path cost and adjacency matrix
    of the ordered points.

    The function first orders the input points `C_axis` using `orderPoints`. Once the points are
    ordered, an adjacency matrix `Tma` is computed. The shortest path between the start and
    end points in the ordered set is then determined using NetworkX, and its cost is calculated
    and returned along with the adjacency matrix.

    :param C_axis: A numpy array or similar structure representing the coordinates of
        points to be ordered and processed.
    :returns: A tuple containing:
        - The cost of the shortest path as computed on the adjacency matrix.
        - The adjacency matrix representing the ordered points.
    :rtype: Tuple[float, Any]
    """
    C_axis, Tma = orderPoints(C_axis)

    return path_cost(Tma, nx.shortest_path(Tma, source=0, target=np.size(C_axis, 0) - 1)), Tma

def get_axis_and_length(cnt, image):
    """
    Determines the medial axis and computes the length of the skeleton within a
    contour region on an input image.

    The function processes a binary skeleton of the input contour and calculates
    its medial axis. It also computes the length of the generated skeleton.

    :param cnt: The contour to be analyzed.
    :type cnt: numpy.ndarray
    :param image: The input image within which the contour lies.
    :type image: numpy.ndarray
    :return: A tuple containing the computed medial axis and the length of the
        skeleton.
    :rtype: tuple[numpy.ndarray, float]
    """
    skel = ski.morphology.medial_axis(snake_to_bw(cnt, image.shape))
    skel = np.argwhere(skel)
    MedialAxis, ie, ib = expand_skel_return(skel, cnt, ratio=5, lenMin=10)
    Lskel, Tma = get_length_return(MedialAxis)

    return MedialAxis, Lskel

def path_sublength(G, path, sublength):
    """
    Compute the node in the path where the cumulative weight of edges exceeds
    a given sublength. This function processes a path within a weighted graph
    and calculates where the cumulative weight surpasses the sublength
    threshold, returning the previous node in the path.

    :param G: A graph in which the path is contained, represented using
        a data structure such as a NetworkX graph. The graph must have
        weights associated with its edges.
    :type G: Any
    :param path: A list of nodes representing the sequential path in the
        graph whose cumulative edge weight is being analyzed.
    :type path: list
    :param sublength: A numeric value representing the threshold for
        cumulative weight along the path that must not be exceeded.
    :type sublength: int or float
    :return: The node in the path right before the cumulative weight exceeds
        the sublength threshold. Returns None if the sublength threshold
        is not exceeded along the given path.
    :rtype: int, float, or None
    """
    Ssubpath = 0
    for i in range(len(path)-1):
        Ssubpath = Ssubpath + G[path[i]][path[i+1]]['weight']
        if Ssubpath > sublength:
            return path[i-1]

def bin_volume(MedialAxis, cnt, ndiv=200):
    """
    Calculate the volumetric measurement of a structure by segmenting and analyzing its geometric properties along a medial axis.

    This function computes the volume of an object based on a set of contour and medial axis points.
    It first orders the medial axis points, connects contour points into a graph, and then processes
    paths to calculate sections, radii, heights, and ultimately the object volumes using geometric volume formulas.

    :param MedialAxis:
        numpy.ndarray with shape (N, 2), where N is the number of medial axis points. Represents
        the ordered set of points that forms the medial axis of the structure.
    :param cnt:
        numpy.ndarray with shape (M, 2), where M is the number of contour points. Represents the
        set of ordered points forming the contour of the structure in 2D space.
    :param ndiv:
        int, optional (default=200)
        Specifies the number of divisions to split along the contour paths to form subsections
        for volume calculations.
    :return:
        A tuple containing:
        - V (float): Total computed volume of the structure in pixels**3.
        - segments_bound (list): A list of boundary segments organized as pairs of x and y
          coordinates for each section of the structure.
        - volumes (list of float): A list of calculated volumes for each subsection computed
          based on the sections' geometries.
        - h (list of float): Heights of the bins/sections for volume computations.
        - radius (list of float): Radii corresponding to each section of the structure.
    """
    xCA, yCA = cnt.T
    MedialAxis, Tma = orderPoints(MedialAxis)
    xMA, yMA = MedialAxis.T
    mb, me = np.zeros(len(cnt)), np.zeros(len(cnt))
    Mb, Me = MedialAxis[0], MedialAxis[-1]

    for i, C in enumerate(cnt):
        mb[i] = np.sum((C[0] - Mb[0]) ** 2 + (C[1] - Mb[1]) ** 2)
        me[i] = np.sum((C[0] - Me[0]) ** 2 + (C[1] - Me[1]) ** 2)
        ib = np.argmin(mb)
        ie = np.argmin(me)

    ## create and clear the graph of the contour
    Gcnt = kneighbors_graph(cnt, 2, mode='distance')
    Tcnt = nx.from_scipy_sparse_array(Gcnt)
    end_cliques = [i for i in list(nx.find_cliques(Tcnt)) if len(i) == 3]
    edge_lengths = [find_longest_edge(i, Tcnt) for i in end_cliques]
    Tcnt.remove_edges_from(edge_lengths)

    ## Find the two paths along the contour
    # from one end to the other end of the gastruloid midline
    L = []
    P = []
    for path in nx.all_simple_paths(Tcnt, source=ib, target=ie):
        P.append(path)
        L.append(path_cost(Tcnt, path))

    ## Subdivide each side
    for i, l in enumerate(L):
        subl = 0
        I = [P[i][0]]
        for j in range(ndiv - 1):
            subl = subl + l / ndiv
            I.append(path_sublength(Tcnt, P[i], subl))
        I.append(P[i][-1])

        if i == 0:
            subX1 = [xCA[k] for k in I]
            subY1 = [yCA[k] for k in I]
        if i == 1:
            subX2 = [xCA[k] for k in I]
            subY2 = [yCA[k] for k in I]

    # Calculate the radius and heights
    path = list(nx.shortest_path(Tma, source=0, target=len(xMA) - 1))

    f = []
    M = [path[0]]
    h = []
    radius = []
    segments_bound = [list(zip([xCA[ib], xCA[ib]], [yCA[ib], yCA[ib]]))]
    fusionP = int(np.round(ndiv/50)) # (= 2%)
    print('removing each ', fusionP, ' first segments at each tip')

    INDEX_Seg = range(fusionP, len(subX1) - fusionP)
    for i in INDEX_Seg:

        xxx = [subX1[i], subX2[i]]
        yyy = [subY1[i], subY2[i]]
        segments_bound.append(list(zip(xxx, yyy)))

        #TODO: implement the cases dx =0 as in the IF profiles extraction pipeline
        m = (yyy[1] - yyy[0]) / (xxx[1] - xxx[0])
        p = yyy[0] - m * xxx[0]
        f.append(np.poly1d([m, p]))
        radius.append(np.power(np.power((yyy[1] - yyy[0]), 2) + np.power((xxx[1] - xxx[0]), 2), 0.5) / 2)

        finter = lambda x, y: y - f[i - fusionP](x)
        res = finter(xMA, yMA)
        M.append(np.argmin(np.abs(res)))
        path = list(nx.shortest_path(Tma, source=M[i-fusionP], target=M[i+1-fusionP]))
        h.append(path_cost(Tma, path))

    M.append(len(yMA) - 1)
    path = list(nx.shortest_path(Tma, source=M[-1-fusionP], target=M[-1]))
    h.append(path_cost(Tma, path))
    segments_bound.append(list(zip([xCA[ie], xCA[ie]], [yCA[ie], yCA[ie]])))

    #TODO : get areas of different bins directly from here !! s
    # this way it can be used in the BF analysis possibly

    ## Calculate the volume in pixels ^3
    vol_coneslice = lambda r, R, h: np.pi / 3 * h * (r ** 2 + R ** 2 + r * R)
    #vol_cone = lambda r, h: np.pi * h * r ** 2 / 3
    vol_cap = lambda r, h: np.pi * h * (3 * r ** 2 + h ** 2) / 6

    volumes=[]
    # first bin:
    volumes.append(vol_cap(radius[0], h[0]))
    # the middle:
    for i in range(1, len(h) - 1):
        volumes.append(vol_coneslice(radius[i], radius[i - 1], h[i]))
    # last bin:
    volumes.append(vol_cap(radius[-1], h[-1]))
    V = np.sum(volumes)

    return V, segments_bound, volumes, h, radius



# large image processing pipelines
def morphological_analysis_image(image, ndiv=200):
    """
    Performs morphological analysis of an input image, assessing its structure and extracting important
    shape-related data like contours, bounding boxes, medial axis, and volumes. The function operates
    by binarizing and processing the input image through a sequence of computational methods. Any
    produced errors during the intermediate steps are flagged and returned accordingly.

    :param image: Input image data to be analyzed.
    :type image: numpy.ndarray
    :param ndiv: The number of divisions for segmenting the medial axis during volume calculation. Defaults to 200.
    :type ndiv: int

    :return: A tuple containing the following outputs:
        - Flag (str): The status of the process indicating any errors or successful completion.
        - mask_cnt (numpy.ndarray): The contour or mask of the binary image, converted to a snake outline.
        - cnt (numpy.ndarray): The refined contour of the object in the binary image.
        - BBOX (list): The bounding box for the primary object in the mask [min_row, max_row, min_col, max_col].
        - MedialAxis (numpy.ndarray): The computed medial axis of the object for structural analysis.
        - length (float): The total computed length of the medial axis for the segmented structure.
        - segments_bound (numpy.ndarray): The bounds of individual segments divided along the medial axis.
        - volume (float): The total calculated volume of the object based on the medial axis and outlines.
        - volumes (numpy.ndarray): The corresponding volumes of each segment along the medial axis.
        - lengths (numpy.ndarray): The lengths of each segment along the medial axis.
        - ndiv (int): The division parameter used for segmentation, returned for completeness.
    :rtype: tuple
    """
    #TODO : save mask indexes = argwhere(mask) instead of the image itself!! save some memory
    #TODO : add the BKGD background value outside of mask

    Flag = 'Keep'
    mask_cnt = None
    BBOX = None

    try:
        lbl = binarize(image)
        mask = clean_binarized(lbl)
        mask_arg = np.argwhere(mask).T
        mask_cnt = bw_to_snake(mask)
        BBOX = [np.min(mask_arg[0]) - 100, np.max(mask_arg[0]) + 100, np.min(mask_arg[1]) - 100,
                np.max(mask_arg[1]) + 100]
    except:
        print('Error in the binarization of image')
        Flag = 'ErrorBin'

    cnt = None
    if Flag == 'Keep':
        try:
            cnt = get_contour(mask, image, alpha=0.075, beta=20, w_line=0, w_edge=0)
            cnt = np.asarray(cnt)
        except:
            print('Error in the contouring of the binary image')
            Flag = 'ErrorCnt'

    MedialAxis = None
    length = None
    if Flag == 'Keep':
        try:
            MedialAxis, length = get_axis_and_length(cnt, image)
        except:
            print('Error in length evaluation')
            Flag = 'ErrorLength'

    segments_bound = None
    volume = None
    volumes = None
    lengths = None

    if Flag == 'Keep':
        try:
            volume, segments_bound, volumes, lengths, thicknesses = bin_volume(MedialAxis, cnt, ndiv=ndiv)
        except:
            print('Error in volume evaluation')
            Flag = 'ErrorVolume'

    return Flag, mask_cnt, cnt, BBOX, MedialAxis, length, segments_bound,volume, volumes, lengths, ndiv

def profiles_analysis(serie_IF):
    """
    Analyzes profiles based on the provided image processing details, segment information,
    and calculated areas. Processes specified regions within an image, extracts relevant
    metrics, and assigns calculated values back to the input data structure.

    :param serie_IF: Dictionary containing analysis parameters and input data.
        Expected keys:
            - ``'absPath'`` (str): Absolute path to the input image file.
            - ``'channels'`` (dict): Mapping of channel labels to indices.
            - ``'Segments'`` (array): Coordinate segments defining analysis boundaries.
            - ``'Lengths'`` (list): Divisions or partitions relevant for processing.
            - ``'Contour'`` (array): Contour coordinates of the region of interest.

    :return: Updated input dictionary ``serie_IF`` with the following additional keys:
        - ``'Areas'``: Array of calculated areas for the specified segments.
        - For each channel in ``'channels'``, an additional key containing values
          of summed pixel intensities within the defined areas.
    :rtype: dict

    :raises ValueError: If the number of channels specified in the input dictionary
        does not match the loaded image's channel count or if other processing
        inconsistencies are encountered.

    """
    # TODO: calculate xbin and xproj and add in in return

    imagefile = serie_IF['absPath']
    dict_ch = serie_IF['channels']
    nchannels = len(dict_ch)
    if os.path.splitext(imagefile)[1] == '.tif':
        import tifffile
        print('tif!')
        image = tifffile.imread(imagefile)
    else:
        print('not tif!')
        image = get_middleslice(imagefile)
        if nchannels == 1:
            image_new = []
            image_new.append(image)
            image = np.array(image_new)
    if image.shape[0] != nchannels:
        print('number of specified channels is incorrect')
        pass

    ndiv = len(serie_IF['Lengths'])

    segments = np.array(serie_IF['Segments']).squeeze()
    subX1 = segments[:, 0, 0]
    subX2 = segments[:, 1, 0]
    subY1 = segments[:, 0, 1]
    subY2 = segments[:, 1, 1]

    cnt = np.array(serie_IF['Contour'])
    xCA, yCA = cnt.T

    Gcnt = kneighbors_graph(cnt, 2, mode='distance')
    Tcnt = nx.from_scipy_sparse_array(Gcnt)
    end_cliques = [i for i in list(nx.find_cliques(Tcnt)) if len(i) == 3]
    edge_lengths = [find_longest_edge(i, Tcnt) for i in end_cliques]
    Tcnt.remove_edges_from(edge_lengths)

    A = []
    B = []
    for i in range(len(subX1)):
        A.append(np.where((xCA == subX1[i]) & (yCA == subY1[i]))[0][0])
        B.append(np.where((xCA == subX2[i]) & (yCA == subY2[i]))[0][0])

    img_dim = image[0, :, :].shape
    imgtot = np.zeros(img_dim)

    f = []
    for i in range(ndiv):

        xxx = [subX1[i], subX2[i]]
        yyy = [subY1[i], subY2[i]]

        dx = np.diff(xxx)
        if not dx == 0:
            m = (yyy[1] - yyy[0]) / (xxx[1] - xxx[0])
            p = yyy[0] - m * xxx[0]
            f.append(np.poly1d([m, p]))
        else:
            f.append([])

    for ii in range(ndiv):

        Icnt1 = nx.shortest_path(Tcnt, source=A[ii], target=A[ii + 1])
        Icnt2 = nx.shortest_path(Tcnt, source=B[ii], target=B[ii + 1])
        xpnt = np.concatenate((xCA[Icnt1], xCA[Icnt2]), axis=None)
        ypnt = np.concatenate((yCA[Icnt1], yCA[Icnt2]), axis=None)

        dx = np.abs(subX2[ii] - subX1[ii])
        dy = np.abs(subY2[ii] - subY1[ii])

        if dx > 10 ** -2:
            xx = np.linspace(subX1[ii], subX2[ii], 2 * int(dx * np.max([1, np.abs(f[ii][1])])))
            yy = f[ii](xx)
        else:
            if dy < 10 ** -2:
                xx = []
                yy = []
            else:
                yy = np.linspace(subY1[ii], subY2[ii], 2 * int(dy))
                xx = subX1[ii] * np.ones(np.shape(yy))

        dxx = np.abs(subX2[ii + 1] - subX1[ii + 1])
        dyy = np.abs(subY2[ii + 1] - subY1[ii + 1])

        if dxx > 10 ** -2:
            xx2 = np.linspace(subX1[ii + 1], subX2[ii + 1], 2 * int(dxx * np.max([1, np.abs(f[ii + 1][1])])))
            yy2 = f[ii + 1](xx2)
        else:
            if dyy < 10 ** -2:
                xx2 = []
                yy2 = []
            else:
                yy2 = np.linspace(subY1[ii], subY2[ii], 2 * int(dyy))
                xx2 = subX1[ii + 1] * np.ones(np.shape(yy2))

        if ii == 0:
            xpnt = np.concatenate((xpnt, xx2), axis=None)
            ypnt = np.concatenate((ypnt, yy2), axis=None)
        if ii == ndiv - 1:
            xpnt = np.concatenate((xpnt, xx), axis=None)
            ypnt = np.concatenate((ypnt, yy), axis=None)
        else:
            xpnt = np.concatenate((xpnt, xx, xx2), axis=None)
            ypnt = np.concatenate((ypnt, yy, yy2), axis=None)

        sna = np.vstack((xpnt, ypnt)).T
        imgtest = snake_to_bw(sna, img_dim)
        ind = np.argwhere(imgtest).T

        imgtot[ind[0], ind[1]] = ii + 1

    area = np.zeros(ndiv)
    for i in range(ndiv):
        area[i] = np.argwhere(imgtot == i + 1).shape[0]

    serie_IF['Areas'] = area

    for ch in dict_ch:
        G = np.zeros(ndiv)
        img = image[dict_ch[ch], :, :]
        for i in range(ndiv):
            G[i] = np.sum(img[np.nonzero(imgtot == i + 1)])
        serie_IF[ch] = G

    return serie_IF

def calc_channel_area(df, channels=[]):
    """
    Calculates the area of specified channels in an image dataset and updates the results
    accordingly. The function processes a DataFrame where each row contains image metadata
    and contour information. It reads the corresponding image files, calculates areas within
    the contour for specified channels, and updates a Series object with the calculated areas.

    This function is designed for use in image analysis pipelines where channel-specific
    measurement needs to be performed within defined contours.

    :param df: A pandas DataFrame containing image metadata and analysis configurations. Each row
               should include fields like `absPath` for the image path, `Contour` for contour
               coordinates, and `channels`—a dictionary mapping channel names to their indices.
    :type df: pandas.DataFrame
    :param channels: A list of channel names whose areas need to be calculated. These channel names
                     are expected to match the keys in the `channels` dictionary from the `df` rows.
    :type channels: list[str]
    :return: A pandas Series indexed by the same indices as the input DataFrame. This Series contains
             dictionaries for each index, mapping channel names to their calculated areas and their
             relative ratios within the given contour.
    :rtype: pandas.Series
    """
    ch_areas = pd.Series(index=df.index)

    for counter, (i, row) in enumerate(df.iterrows()):
        analyze = True
        imagepath = row['absPath']
        print(imagepath)

        if not os.path.exists(imagepath):
            row['Flag'] = 'ImageNotFound'
            row = row.to_frame().transpose()
            row.index = [i]
            df.update(row)
            continue

        imtiff = tifffile.imread(imagepath)

        try:
            cnt = row['Contour']
            mask = snake_to_bw(cnt, imtiff[0, :, :].shape)
            gast_area = sum(sum(mask))
            ch_area_dict = {'DAPI': gast_area}
            analyze = True
        except:
            ch_areas[i] = 0
            analyze = False

        if analyze:
            for channel in channels:
                if channel in row['channels']:
                    ch = row['channels'][channel]
                    ch_image = np.int16(imtiff[ch, :, :])
                    masked_ch_img = ch_image * mask
                    try:
                        ch_image = np.int16(imtiff[ch, :, :])
                        binary, val = binarize_channel(ch_image)
                        masked_ch_img = binary * mask
                        ch_area = sum(sum(masked_ch_img))
                        ch_area_dict[channel] = ch_area
                        ch_area_dict[f'{channel} ratio'] = ch_area / gast_area
                        ch_area_dict[f'{channel} th'] = val
                    except:
                        ch_area_dict[channel] = 0
                    ch_areas[i] = ch_area_dict
                else:
                    print(channel, ' not specified in channel dictionary!')
            print('index:', i, 'Channel area calculation completed. :)')
        else:
            print('index:', i, 'No contour found so no ratio was calculated.')

        return ch_areas

def analysis_df(df, target, channels=['DAPI', 'BF', 'Sox2'], profile_analysis=True):
    """
    Perform analysis on a given DataFrame containing image data and update it with
    the analysis results, including computed features and flags.

    This function processes image data row by row in the provided DataFrame. It checks for
    image availability and runs a morphological analysis and optional profile analysis
    on each image. The results of the analysis are then added to the DataFrame.

    :param df: DataFrame containing image-related data and metadata for analysis.
    :type df: pandas.DataFrame
    :param target: Target directory path where the resulting JSON files will be saved.
    :type target: str
    :param channels: List of channel names to be processed during analysis. Defaults to
        ``['DAPI', 'BF', 'Sox2']``.
    :type channels: list[str], optional
    :param profile_analysis: Boolean flag indicating if profile analysis should be included.
        Defaults to ``True``.
    :type profile_analysis: bool, optional
    :return: Updated DataFrame containing the results of the analysis for each image,
        including computed features and flags.
    :rtype: pandas.DataFrame
    """
    print(df.index)

    df['Flag'] = 'Initial'
    df['Mask'] = None
    df['Contour'] = None
    df['BBOX'] = None
    df['MedialAxis'] = None
    df['Segments'] = None
    df['Length_MA'] = None
    df['Volume_MA'] = None
    df['Volumes'] = None
    df['Lengths'] = None
    df['Areas'] = None
    df['ndiv'] = None

    for ch in channels:
        df[ch] = None

    for counter, (i, row) in enumerate(df.iterrows()):

        ID = row['imageID']
        filename = target + ID + '.json'
        imagepath = row['absPath']

        if not os.path.exists(imagepath):
            row['Flag'] = 'ImageNotFound'
            row = row.to_frame().transpose()
            row.to_json(filename, index=False, orient='table')
            row.index = [i]
            df.update(row)
            continue

        imtiff = tifffile.imread(imagepath)
        dapich = row['channels']['DAPI']
        image = np.int16(imtiff[dapich,:,:])
        nbin = 200

        try:
            Flag, maskarg, cnt, BBOX, MedialAxis, length, segments_bound, volume, volumes, lengths, ndiv =  \
                morphological_analysis_image(image, ndiv=nbin)

            print('Analysis completed for row', i, ID)
            row['Flag'] = Flag
            row['Mask'] = maskarg
            row['Contour'] = cnt
            row['BBOX'] = BBOX
            row['MedialAxis'] = MedialAxis
            row['Segments'] = segments_bound
            row['Length_MA'] = length
            row['Volume_MA'] = volume
            row['Volumes'] = volumes
            row['Lengths'] = lengths
            row['ndiv'] = ndiv
        except:
            print('Analysis error')

        print(row['Flag'])

        if profile_analysis:
            if row['Flag'] == 'Keep':
                row_out = profiles_analysis(row)
                row = row_out

        row = row.to_frame().transpose()
        row.index = [i]
        df.update(row)
        print('df updated, exit \n')
        print(f'{counter +1}/{len(df)} completed')

    return df

def initialize_czi_df(config, save=True):
    """
    Initializes a dataframe for .czi image files and processes their metadata, generating max projections and saving
    the data and configurations. This function handles metadata extraction, file handling, and image processing
    to facilitate data organization and processing for .czi image files.

    :param config: Configuration dictionary containing the paths, file names, aliases, metadata-related keys,
                   and other settings needed for the initialization process.
    :type config: dict
    :param save: Boolean flag indicating whether to save the initialized dataframe to a JSON file.
    :type save: bool
    :return: A pandas DataFrame with metadata and file paths of the processed image files.
    :rtype: pd.DataFrame
    """
    raw_folder = config['data_path']
    files = config['files']
    aliases = config['aliases']
    target = config['results_folder']
    dict_channel = config['dict_channel']
    ncells = config['ncells']
    magn = config['magn']
    method = config['method']

    df_IF = pd.DataFrame(
        columns=['imageID', 'absPath', 'file', 'magn', 'method', 'um_per_pixel', 'Nx', 'Ny', 'channels', 'ncells'])
    for i, f in enumerate(files):
        print('f', f)

        df_IF = pd.DataFrame(
            columns=['imageID', 'absPath', 'file', 'magn', 'method', 'um_per_pixel', 'Nx', 'Ny', 'channels', 'ncells'])

        metadata = GetMetadata(raw_folder + f)
        um_per_pixel = metadata['dx']
        Nc = metadata['Nch']
        Nx = metadata['Nx']
        Ny = metadata['Ny']
        Ns = metadata['Nseries']
        print("# series = ", Ns)

        tiff_folder = os.path.join(target, aliases[i])

        if not os.path.exists(tiff_folder):
            os.mkdir(tiff_folder)

        for s in range(Ns):
            print('s', s)

            savefile = tiff_folder + '/' + aliases[i] + '_S' + str(s).zfill(2) + '_maxproj.tif'
            savefilez = tiff_folder + '/' + aliases[i] + '_S' + str(s).zfill(2) + '_maxprojz.tif'

            new_row = {'imageID': aliases[i] + '_S' + str(s).zfill(2), 'expID': aliases[i],
                       'absPath': savefile, 'file': raw_folder + f, 'magn': magn[0], 'method': method[0],
                       'um_per_pixel': um_per_pixel, 'Nx': Nx, 'Ny': Ny,
                       'channels': dict_channel[0], 'ncells': ncells[i]}
            df_IF = df_IF._append(new_row, ignore_index=True)

            if os.path.exists(savefile):
                print('File', savefile, 'already exists.')
            else:
                print('projecting...')
                # Create the max projection and save it
                img_tiff = np.zeros([Nc, Ny, Nx])
                indz_tiff = img_tiff.copy()

                for ch in range(Nc):
                    print(s, ch)
                    img_ch, indz = maxprojection(raw_folder + f, theS=s, theC=ch)
                    img_tiff[ch, :, :] = img_ch
                    indz_tiff[ch, :, :] = indz
                tifffile.imsave(savefile, img_tiff)
                tifffile.imsave(savefilez, indz_tiff)

            if save:
                df_IF.to_json(target + f'{aliases[i]}_data.json', orient='split')
                print('All files have been projected and a dataframe has each been initialized that is saved under ',
                      target, f'{aliases[i]}_data.json')

        config['initialization'] = {'tiffs': True, 'data_frames': True, 'saved': save}
        configs.update_configs(config)
        print('configs after initialization: ', config)

    return df_IF

def analyse_czi(config, profile_analysis=True, save=True, pickup_existing=False):
    """
    Analyzes .czi image data based on the provided configuration. Processes data for
    each alias in the configuration and saves the results if specified.

    :param config: A dictionary containing the configurations for analysis.
        Must include 'aliases' (list of aliases), 'results_folder' (directory
        for saving results), and 'dict_channel' (channel configuration dictionary).
    :type config: dict
    :param profile_analysis: A boolean indicating whether to perform profile analysis
        on the data. Defaults to True.
    :type profile_analysis: bool, optional
    :param save: A boolean indicating whether to save the processed data.
        If True, results will be saved in the specified results folder.
        Defaults to True.
    :type save: bool, optional
    :param pickup_existing: A boolean option to process only new data if existing
        processed files are found. Defaults to False. Currently not in use since
        the related functionality is commented out.
    :type pickup_existing: bool, optional
    :return: A DataFrame containing the processed data for the last alias in
        the configuration list.
    :rtype: pandas.DataFrame
    """
    aliases = config['aliases']
    target = config['results_folder']
    dict_channel = config['dict_channel']

    for ali in range(len(aliases)):
        filename = os.path.join(target + f'{aliases[ali]}_data.json')
        df_read = pd.read_json(filename, orient='split')

        #if pickup_existing:
        #    ex_filename = filename.replace('data.json', 'data_processed.json')
        #    if os.path.exists(ex_filename):
        #        config['image_analysis'] = {'morphological_analysis': True, 'profile_analysis': profile_analysis,
        #                                    'saved': save}
        #        print('configs after analyze czi: ', config)
        #        configs.update_configs(config)
        #        continue

        channels = dict_channel[0].keys()
        processed_df = analysis_df(df_read, target, channels=channels, profile_analysis=profile_analysis)

        if save:
            processed_df.to_json(target + f'{aliases[ali]}_data_processed.json', orient='split')
            print(f'A dataframe for {aliases[ali]} has been analyzed and is saved under {target} + {aliases[ali]}_data_processed.json.')

    config['image_analysis'] = {'morphological_analysis': True, 'profile_analysis': profile_analysis, 'saved': save}
    print('configs after analyze czi: ', config)
    configs.update_configs(config)

    return processed_df

def merge_dfs(config, save=True, savename='df_all'):
    """
    Merges multiple DataFrame objects from JSON files, appends an additional column to the
    resulting DataFrame, and optionally saves the merged DataFrame as a new JSON file. Updates
    the configuration dictionary to reflect changes.

    :param config: A dictionary containing the required configuration for the merge. Must include:
        - 'aliases': A list of aliases corresponding to the DataFrame files to be merged.
        - 'results_folder': The folder where the DataFrame files are located.
        - 'overview_plot': A dictionary with a 'saved' key indicating whether the resulting
          DataFrame should be saved.

    :param save: A boolean indicating whether to save the merged DataFrame as a new JSON file.
    :param savename: A string representing the name to use for saving the merged DataFrame file.

    :return: A pandas DataFrame representing the merged result.
    """
    aliases = config['aliases']
    target = config['results_folder']

    df_all = pd.read_json(target + f'{aliases[0]}_data_processed.json', orient='split')

    if len(aliases) > 1:
        for ali in aliases[1:]:
            if ali == aliases[0]:
                continue
            df = pd.read_json(target + f'{ali}_data_processed.json', orient='split')
            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all['expID'] = [i.rsplit('_', 1)[0] for i in df_all['imageID']]
    save_name = target + savename + '.json'

    if config['overview_plot']['saved']:
        df_all.to_json(save_name, orient='split')
        print(f'Data was successfully saved under {target + savename + ".json"}')

    config['merged'] = {'merged': True, 'save_name': save_name, 'saved': save}
    print(config)
    configs.update_configs(config)

    return df_all



def get_shape_descriptors(Mask, i, df):
    """
    Extracts shape descriptors from a labeled binary mask and updates the given
    dataframe with calculated measurements for a specific index. This function
    uses region properties from the `skimage.measure` library to calculate
    geometric and shape-related features, including solidity, perimeter, aspect
    ratio (AR), major and minor axis lengths, area, and circularity. The
    measurements are scaled using a pixel-to-micrometer conversion value from the
    dataframe.

    :param Mask: A binary mask from which shape descriptors will be extracted.
    :param i: int
        The index in the dataframe to update with the calculated measurements.
    :param df: pandas.DataFrame
        The dataframe containing a column 'um_per_pixel' for unit conversion,
        and target columns for updating the calculated properties.
    :return: pandas.DataFrame
        The updated dataframe containing calculated shape descriptors for the
        given index.
    """
    label, num_lab = ski.measure.label(Mask, return_num=True, connectivity=2)
    Prop = ski.measure.regionprops(label)
    #Moments_Hu = Prop[0].moments_hu

    pxsize = df.at[i, 'um_per_pixel']

    df.loc[i, 'Solidity'] = Prop[0].solidity
    df.loc[i, 'Perimeter'] = pxsize * Prop[0].perimeter
    df.loc[i, 'AR'] = Prop[0].axis_major_length / Prop[0].axis_minor_length
    df.loc[i, 'Major_Axis_um'] = [pxsize * Prop[0].axis_major_length]
    df.loc[i, 'Minor_Axis_um'] = [pxsize * Prop[0].axis_minor_length]
    df.loc[i, 'Area'] = pxsize * pxsize * Prop[0].area
    df.loc[i, 'Circularity'] = 4 * math.pi * Prop[0].area / (Prop[0].perimeter * Prop[0].perimeter)

    return df
