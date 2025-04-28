import os
import sys
import math
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)
import modules.configs_setup as configs
import modules.raw_IF_analysis as raw
import modules.javaThings as jv
jv.start_jvm()
jv.init_logger()

### for outlier detection of IF image analysis
def optimalsubplots(n, ratio=1):
    """
    Calculates the optimal number of rows and columns for subplots given the total
    number of plots and an optional aspect ratio. The function ensures that the layout
    of the subplots fits the desired aspect ratio as closely as possible.

    :param n: The total number of plots that require arrangement.
    :type n: int
    :param ratio: The desired aspect ratio (width-to-height) of the subplot grid. Defaults to 1.
    :type ratio: float, optional
    :return: A tuple containing the number of columns and rows for the subplot grid.
    :rtype: Tuple[int, int]
    """
    if n == 1:
        row = 1
        col = 1

        return col,row

    row = math.ceil(np.sqrt(n))
    if row == np.sqrt(n):
        col = row
    else:
        col = row+1

    if col/row > ratio:
        while col/row > ratio:
            col -= 1
            row = n // col
            row += np.sign(n % col)


    elif col/row < ratio:
        while col/row < ratio:
            col +=1
            row = n // col
            row+= np.sign(n % col)

    return col, row

def rotate(origin, point, angle):
    """
    Rotates a point around a specified origin by a given angle.

    This function computes the new coordinates for a point after
    applying a rotation about a specific origin. The rotation angle
    is measured in radians, and the mathematical conventions for
    2D Cartesian coordinate rotation are used.

    :param origin: The origin point used as the pivot for the rotation.
    :type origin: tuple[float, float]
    :param point: The point to rotate around the origin.
    :type point: tuple[float, float]
    :param angle: The angle of rotation in radians.
    :type angle: float
    :return: A tuple representing the new coordinates of the point after rotation.
    :rtype: tuple[float, float]
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def dispIm(ax,img, cmap='gray'):

    """
    Displays an image on a given axis with specific adjustments such
    as disabling the axis, enabling the frame, and scaling it.

    :param ax: The matplotlib axis object where the image will be displayed.
    :param img: The image to display, typically as a NumPy array or other
        supported data type by matplotlib for image rendering.
    :param cmap: The colormap used to display the image. Defaults to 'gray'.
    :return: The updated axis object after rendering the image.
    """

    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    ax.axis('scaled')
    ax.set_frame_on(True)

    return ax

def addContour(ax, cnt, color='r', **kwargs):

    """
    Adds a contour to a given matplotlib Axes object by plotting its points
    in reverse order (y vs x). The function supports optional customization
    of the plot using additional keyword arguments.

    If the contour (`cnt`) is not a numpy ndarray, it will be converted.
    An input contour must have a shape of (npts, 2). Otherwise, the function
    will print an error message describing the expected shape and return the
    unchanged Axes object without plotting the contour.

    :param ax: The matplotlib Axes object where the contour will be added.
    :type ax: matplotlib.axes.Axes
    :param cnt: The contour data as a 2D array of shape (npts, 2), where each
        row represents a point with its x and y coordinates.
    :type cnt: numpy.ndarray
    :param color: Optional; The color of the contour line. Default is 'r'.
    :type color: str
    :param kwargs: Keyword arguments to be passed to the matplotlib `plot`
        function for further customization (e.g., linestyle, linewidth).
    :return: The modified matplotlib Axes object with the contour added.
    :rtype: matplotlib.axes.Axes
    """

    if not isinstance(cnt, np.ndarray):
        cnt = np.asarray(cnt)
    if cnt.shape[1] !=2:
        print('Wrong dimensions of the contour array : ', cnt.shape, '. It should be npts x 2.')
        return ax
    xCnt, yCnt = cnt.T
    ax.plot(yCnt, xCnt, c=color, **kwargs)

def getbboxmax(data):
    """
    Determine the maximal dimensions of a set of bounding boxes and calculate new bounding boxes
    centered around the center of each original bounding box with the maximal dimensions.

    :param data: List of bounding boxes where each bounding box is represented as a list or tuple of
                 four elements [y_min, y_max, x_min, x_max].
    :type data: list[list[float]] or list[tuple[float, float, float, float]]
    :return: A tuple containing:
             - The maximal dimensions as a tuple (dimy, dimx).
             - A list of new bounding boxes along the x-dimension (bboxx).
             - A list of new bounding boxes along the y-dimension (bboxy).
    :rtype: tuple[tuple[int, int], list[list[float]], list[list[float]]]
    """

    # Determine the maximal dimension of the images
    dx = []
    dy = []
    bboxx = []
    bboxy = []
    for b in data:
        dx.append(b[3]-b[2])
        dy.append(b[1]-b[0])
    dimx = int(np.max(dx))
    dimy = int(np.max(dy))
    # Calculate the corresponding bbox centered around the center of each bbox for each image
    for b in data:
            bboxx.append( [(b[3]+b[2])/2 - dimx/2, (b[3]+b[2])/2 + dimx/2] )
            bboxy.append( [(b[1]+b[0])/2 - dimy/2, (b[1]+b[0])/2 + dimy/2] )

    return (dimy,dimx), bboxx, bboxy

def AddScale_well(ax, scale_pix, Verticalsize, s=2, unit='$\mu$m', c='white', ratio=50):
    """
    Add a scale bar to a matplotlib axis.

    This function inserts a scale bar into the plot to visually represent the
    scale of the image or data within the provided matplotlib axis. It provides
    customization for the size, color, and unit of measurement of the scale bar,
    as well as its vertical thickness. The scale bar is positioned in the 'lower
    right' of the axis by default.

    :param ax: The matplotlib axis to which the scale bar will be added.
    :type ax: matplotlib.axes._axes.Axes
    :param scale_pix: The pixel length of the scale bar.
    :type scale_pix: float or int
    :param Verticalsize: The total vertical size of the scale bar as
        a measurement.
    :type Verticalsize: float or int
    :param s: Font size for the scale bar label, defaults to 2.
    :type s: float or int, optional
    :param unit: The unit of measurement for the scale, defaults to '$\mu$m'.
    :type unit: str, optional
    :param c: The color of the scale bar, defaults to 'white'.
    :type c: str, optional
    :param ratio: The ratio to divide `Verticalsize` to establish the vertical
        thickness of the scale bar, defaults to 50.
    :type ratio: float or int, optional
    :return: None
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=s)
    scalebar = AnchoredSizeBar(ax.transData,
                               scale_pix, None, 'lower right',
                               pad=5,
                               color=c,
                               frameon=False,
                               size_vertical=Verticalsize/ratio,
                               fontproperties=fontprops)

    ax.add_artist(scalebar)


## paris style
def outlier_visualization_detection(df, explore=True, ratio=2, figheight = 5, channel=0, colors=[], mode = 'custom',
                            BBOX=False, MASK=True, CNT= True, MEDAX=True, SEG=False, ROT=False, BCK=False, ANNOT=None):
    """
    Visualizes and detects outliers in image datasets, providing a montage view with various customization modes.
    This function generates a visualization montage of images in a dataset, allowing the user to specify
    parameters for customization, including channel visualization, color overlays, masking, contouring,
    medial axis display, and rotation options.

    It works with various modes such as 'data', 'complete', 'custom', and 'plate', providing
    different levels of detail and configurations for the visualization. The resulting montage
    displays the images based on the provided parameters, aiding in the detection of outliers
    or analysis of image features.

    Supported formats include `.tif` and `.czi`, with partial support for `.vsi`. Additional
    parameters help manage rotation, bounding boxes, and annotations.

    :param df:
        DataFrame containing metadata and paths for the images, along with any selection criteria.
    :param explore:
        Flag to indicate whether to visually explore and display outliers (default: True).
    :param ratio:
        Aspect ratio for subplots (default: 2).
    :param figheight:
        Height of the generated figure montage (default: 5).
    :param channel:
        Channel number to display for multi-channel images (default: 0).
    :param colors:
        List of colors for overlays on the visualization, used to differentiate channels or regions.
    :param mode:
        Mode of visualizing images. Supported modes are 'data', 'complete', 'custom', and 'plate'.
        Each mode provides varying complexities of display.
    :param BBOX:
        Flag to determine if bounding boxes should be drawn around objects in the images.
    :param MASK:
        Flag to determine if masks should be overlaid on the images. Default: True.
    :param CNT:
        Flag to overlay contour regions on the images.
    :param MEDAX:
        Flag to overlay medial axes information on the images.
    :param SEG:
        Flag to indicate whether segmented elements should be shown.
    :param ROT:
        Flag enabling rotation of images according to calculated angles in the DataFrame.
    :param BCK:
        Flag to specify manipulation of background values in images during masking operations.
    :param ANNOT:
        Annotation content to be applied to the montage for labeling purposes.

    :return:
        Generates a montage of images as a visualization. Returns `0, 0, 0` for invalid configurations
        or empty DataFrame conditions in specific modes (e.g., 'plate' mode requiring well, row, and column
        metadata).
    """

    #TODO: implement the choice of the extension
    #TODO: improve using a switch type of statement?
    #TODO: add the option to plot the corresponding profiles ?
    #TODO: correct the error for the rotation ! and implement an adjustemnet of the stretching when resizing different images to the same size so that there is a common scale
    #TODO: correct the correspondance k for axes and i for df in the case when i am only enumerate the value where some df data exist !!
    #TODO: correct the case where no channels is specifice: then all channels for IF and 1 channel for BF**

    #TODO : adapt the function so it takes a dictionnary of what key to use
    # and plot and which kind of data it is (scatter? image? etc?)


    if mode == 'data':
        MASK = False
        CNT = False
        MEDAX = False
        SEG = False

    if mode == 'complete':
        MASK = True
        CNT = True
        MEDAX = True
        SEG = True

    if mode == 'plate':
        if not 'Well' in df.columns.values:
            print('Mode is plate, but no well info in dataframe')
            return 0, 0, 0
        if not 'Row' in df.columns.values:
            print('Mode is plate, but no row info in dataframe')
            return 0, 0, 0
        if not 'Col' in df.columns.values:
            print('Mode is plate, but no col info in dataframe')
            return 0, 0, 0
        ratio = 1.5
        nImages = 96

    else:
        nImages = len(df)

    if not any(m in mode for m in ['data', 'complete', 'custom', 'plate']):
        print("Possible values for mode are data, complete, plate and custom. \n Using the data mode, only the raw images will be displayed. "
              "Use the complete mode to see all informations. Use custom to specify what you want to see or not.\n")

    if ROT:
        if not 'rot_angle' in df.columns.values:
            print('Rotation angle not calculated. ')
        else:
            from scipy import ndimage

    df = df.reset_index()
    print('making a montage of ', nImages, ' images...')
    (col, row) = optimalsubplots(nImages, ratio=ratio)
    print(row, 'x', col, ' subplots')

    dim = None

    fig, axs = plt.subplots(row, col)
    if np.size(axs) == 1:
        axs = [axs]
    else:
        axs = axs.flat

    if mode == 'plate':
        K = []
        for i in df.index:
            K.append(int(12 * (df.at[i, 'Row']-1) + df.at[i, 'Col']-1))
    else:
        K = range(len(df.index))

    for (k, i) in zip(K, df.index):

        filepath = df.at[i, 'absPath']
        if not filepath:
            axs[k].axis('off')
            continue
        if not os.path.exists(filepath):
            axs[k].axis('off')
            continue

        if os.path.splitext(filepath)[1] == '.vsi':
            image = raw.get_middleslice(filepath)
        elif os.path.splitext(filepath)[1] == '.czi':
            pass
        elif os.path.splitext(filepath)[1] == '.tif':
            tif = tifffile.imread(filepath)
            tif = tif / np.amax(tif)
            tif = np.clip(tif, 0, 1)

            dimtif = np.shape(tif)
            ich = np.argmin(dimtif)
            nch = dimtif[ich]
            if channel > nch:
                raise IndexError("Requested channel is ", channel, " but tiff image only contains ", nch, 'channels')
            image = np.swapaxes(tif, ich, 0)

            image = np.moveaxis(image[[channel], :, :], [0,1,2], [2,0,1])
            final = image

        if ROT:
            if not 'rot_angle' in df.columns.values:
                print('No rotation angle as been calculated!')
            else:
                rotated = ndimage.rotate(image, df.at[i, 'rot_angle'], reshape=True)
                nx, ny = image.shape[:2]
                diag = np.sqrt(nx**2+ny**2)
                x,y = rotated.shape[:2]
                rotated = rotated[:,:,np.newaxis]
                newx = int(x / y * diag)
                newy = int(y / x * diag)
                final = np.zeros((newx, newy, 1))
                final[int(newx/2-x/2):int(newx/2+x/2), int(newy/2-y/2):int(newy/2+y/2)] = rotated
                if i in df[df['Mask'].notnull()].index.values:
                    mask_raw = np.array(df.at[i, 'Mask']).astype(np.int32)
                    mask = [rotate((ny / 2, nx / 2), m, math.radians(df.at[i, 'rot_angle'])) for m in mask_raw]
                    mask = np.asarray([(m[0] + df.at[i, 'dy_rot'], m[1] + df.at[i, 'dx_rot']) for m in mask])
                    mask = raw.snake_to_bw(mask, final.shape[:2])
                    if BCK:
                        mask_inv = np.invert(mask)
                        print('BCK',  np.nanmean(image[np.nonzero(mask_inv)]) )
                        final[np.nonzero(mask_inv)] = np.nanmean(image[np.nonzero(mask_inv)])
                    else:
                        mask_inv = 1-mask

                        final[np.nonzero(mask_inv)] = 0

                else:
                    df.at[i, 'BBOX_rot'] = [0, final.shape[0], 0, final.shape[1]]

            if np.shape(final)[-1] == 2:
                final = np.dstack((final, np.zeros(final.shape[:2])))

        if colors:
            final_new = np.zeros(np.shape(final))
            for i, c in enumerate(colors):
                final_new[:, :, c] = final[:, :, i]
            final = final_new

            out = (final/np.max(np.max(final)).astype(float))
            dispIm(axs[k], out)

        else:
            final = image
            dispIm(axs[k], image)

        if not dim:
            dim = np.shape(final)
            print('Images of size ', dim)


    if MASK:
        for (k, i) in zip(K, df[df['Mask'].notna()].index):
            mask = np.array(df.at[i, 'Mask'])
            if ROT:
                #Case where this is a Immunofluorescence image
                nx, ny = df.at[i, 'Nx'], df.at[i, 'Ny']
                mask = [rotate((ny / 2, nx / 2), m, math.radians(df.at[i, 'rot_angle'])) for m in mask]
                mask = np.asarray([(m[0] + df.at[i, 'dy_rot'], m[1] + df.at[i, 'dx_rot']) for m in mask])

            addContour(axs[k], mask, color='y')

    if CNT:
        for (k, i) in zip(K, df[df['Contour'].notna()].index):
            cnt = np.array(df.at[i, 'Contour'])
            if ROT:
                cnt = [rotate((ny / 2, nx / 2), m, math.radians(df.at[i, 'rot_angle'])) for m in cnt]
                cnt = np.asarray([(m[0] + df.at[i, 'dy_rot'], m[1] + df.at[i, 'dx_rot']) for m in cnt])
            addContour(axs[k], cnt, color='b')

    if MEDAX:
        for (k, i) in zip(K, df[df['MedialAxis'].notna()].index):
            med = np.array(df.at[i, 'MedialAxis'])
            if ROT:
                med = [rotate((ny/2, nx/2), m, math.radians(df.at[i, 'rot_angle'])) for m in med]
                med = np.asarray([(m[0] + df.at[i, 'dy_rot'], m[1] + df.at[i, 'dx_rot']) for m in med])

            addContour(axs[k], med, color='r', linewidth=2)

    if SEG:
        for (k, i) in zip(K, df[df['Segments'].notna()].index):
            S = df.at[i, 'Segments']
            S = np.array(S).squeeze()
            n = int(np.ceil(np.shape(S)[0]/20))
            for seg in range(np.shape(S)[0]):
                if seg%(n) == 0:
                    axs[k].plot([S[seg, 0, 1], S[seg, 1, 1]], [S[seg, 0, 0], S[seg, 1, 0]], c='g')

    if BBOX:
        try:
            if ROT:
                dim, bbx, bby = getbboxmax(df['BBOX_rot'])
                for (k, bx, by) in zip(K, bbx, bby):
                    axs[k].set_xlim(bx[0], bx[1])
                    axs[k].set_ylim(by[0], by[1])
            else:
                dim, bbx, bby = getbboxmax(df['BBOX'])
                for (k, bx, by) in zip(K, bbx, bby):
                    axs[k].set_xlim(bx[0], bx[1])
                    axs[k].set_ylim(by[0], by[1])
        except:
            print('no bbox')

    for k in list(set(range(col*row)) - set(K)):
        axs[k].axis('off')

    if not dim:
        dim = [100, 100]
    print('Final display of size ', dim)
    figwidth = figheight*col/row*dim[1]/dim[0]
    fig.set_size_inches(figwidth, figheight)

    alpha, delta = 0, 0
    if explore:
        alpha, delta = 0.05, 0.2

    fig.subplots_adjust(left=alpha, right=1-alpha, top=1-alpha, bottom=alpha)
    fig.subplots_adjust(wspace=delta, hspace=delta)

    if ANNOT:
        if BBOX:
            for (k, i, bx, by) in zip(K, df[df[ANNOT].notna()].index, bbx, bby):
                axs[k].text(bx[0] + 0.1 * dim[1], by[0] + 0.9 * dim[0], df.at[i, ANNOT], color='r',
                            fontsize=figwidth/col*9)
        else:
            for (k, i) in zip(K, df[df[ANNOT].notna()].index):
                axs[k].text(0.1, 0.9, df.at[i, ANNOT], color='r', transform=axs[k].transAxes, fontsize=figwidth/col*9)

    return fig, axs, dim

# princeton style
'''
def czi_plot_overview(df, folder, param='imageID', morph_analysis=False, mark_filtered=False, filtered_gastrus=[],
                      channel=1, ncol=7, save=True, ending='.pdf', cmap='binary'):

    plot and overview of all gastruloids from an czi file, with the chosen channel in a well plate like style

    if len(filtered_gastrus) > 0:
        filtered = [oid for oid in df['imageID'] if oid.rsplit('_', 1)[1] in filtered_gastrus]

    ncol = ncol
    nrow = math.ceil(len(df) / ncol)

    if len(df) < ncol:
        fig, axes = plt.subplots(1, len(df), figsize=(len(df), 1))

    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

        for i in range(nrow * ncol - len(df)):
            axes[nrow - 1, ncol - i - 1].set_visible(False)
            # axes = axes.flatten()

    for ix, ax in zip(df.index, axes.flatten()):

        imagepath = df.at[ix, 'absPath']
        if os.path.exists(imagepath):
            imtiff = tifffile.imread(imagepath)
            image = np.int16(imtiff[channel, :, :])

        um_per_pixel = df.at[ix, 'um_per_pixel']
        ID = df.at[ix, 'imageID'].rsplit('_', 1)[1]

        ax.imshow(image, cmap=cmap)

        if morph_analysis:
            try:
                medax = np.array(df.at[ix, 'MedialAxis'])
                xAx, yAx = medax.T
                cnt = np.array(df.at[ix, 'Contour'])
                xCnt, yCnt = cnt.T
                #S = np.array(df.at[ix, 'Segments'])

                ax.plot(yAx, xAx, c='r', lw=1)
                ax.plot(yCnt, xCnt, c='k', lw=1)
                # for seg in range(len(S[0])):
                #    if seg%2 ==0:
                #        ax.plot([S[0][seg][0][1], S[0][seg][1][1]], [S[0][seg][0][0], S[0][seg][1][0]], c='white', alpha=0.5)

            except:
                ax.scatter(0.1 * image.shape[0], 0.1 * image.shape[1], c='k', marker='x', s=20)

        if mark_filtered:
            if ID in filtered_gastrus:
                df.at[ix, 'Flag'] != 'Keep'
                ax.scatter(0.2 * image.shape[0], 0.1 * image.shape[1], c='r', marker='x', s=20)

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_title(f'i:{ix}\n{df.at[ix, param]}', fontsize=5, pad=0)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        if ix == 0:
            AddScale_well(ax, 100 / um_per_pixel, 100, 100, c='k')

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)

    if save:
        target = folder + r'plots/'
        if not os.path.exists(target):
            os.mkdir(target)

        if (param == 'imageID') and (len(df.expID.unique()) == 1):
            savename = target + f'{df.at[ix, "expID"]}_overview{ending}'
        else:
            savename = target + f'{df.at[ix, param]}_overview{ending}'
        plt.savefig(savename, dpi=200, transparent=True)
        print('Successfully saved the overview plot under ', savename)

    else:
        plt.show()
        plt.close()
'''

def czi_plot_overview(df, config, param='imageID', morph_analysis=True, mark_filtered=True, filtered_gastrus=[], ncol=7,
                      save=True, cmap='binary'):
    """
    Generates an overview plot visualizing data from a given DataFrame, where images and
    their annotations, such as medial axes or contours, are displayed in a grid layout.
    The function allows for marking filtered images, custom channel selection, and analysis
    of morphological data.

    :param df:
        Input DataFrame containing image data and metadata such as image paths,
        scaling factors, and ID information.

    :param config:
        Dictionary containing configuration parameters, including channel information,
        plot settings, and save paths.

    :param param:
        The column name in the DataFrame to use as a label for the images in the
        plot grid. Defaults to 'imageID'.

    :param morph_analysis:
        Boolean flag indicating whether to display morphological analysis, such as the
        medial axis or contours, on the plot. Defaults to True.

    :param mark_filtered:
        Boolean flag indicating whether to mark filtered items in the plot using a
        specific marker. Defaults to True.

    :param filtered_gastrus:
        List of IDs corresponding to filtered images that should be highlighted in
        the plot. Defaults to an empty list.

    :param ncol:
        Number of columns in the plot grid layout. Determines how images are
        arranged in terms of rows and columns. Defaults to 7.

    :param save:
        Boolean flag indicating whether to save the generated plot to a file. If False,
        displays the plot instead of saving it. Defaults to True.

    :param cmap:
        Colormap used for displaying the images in the plot. Defaults to 'binary'.

    :return:
        None. Performs plotting and optionally saves the generated plot to disk.
    """
    ch_dict = config['dict_channel'][0]
    channel = ch_dict[config['overview_plot']['ch_to_use']]

    if len(filtered_gastrus) > 0:
        filtered = [oid for oid in df['imageID'] if oid.rsplit('_', 1)[1] in filtered_gastrus]

    ncol = ncol
    nrow = math.ceil(len(df) / ncol)

    if len(df) < ncol:
        fig, axes = plt.subplots(1, len(df), figsize=(len(df), 1))

    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

        for i in range(nrow * ncol - len(df)):
            axes[nrow - 1, ncol - i - 1].set_visible(False)

    for ix, ax in zip(df.index, axes.flatten()):

        imagepath = df.at[ix, 'absPath']
        if os.path.exists(imagepath):
            imtiff = tifffile.imread(imagepath)
            image = np.int16(imtiff[channel, :, :])

        um_per_pixel = df.at[ix, 'um_per_pixel']
        ID = df.at[ix, 'imageID'].rsplit('_', 1)[1]

        ax.imshow(image, cmap=cmap)

        if morph_analysis:
            try:
                medax = np.array(df.at[ix, 'MedialAxis'])
                xAx, yAx = medax.T
                cnt = np.array(df.at[ix, 'Contour'])
                xCnt, yCnt = cnt.T
                #S = np.array(df.at[ix, 'Segments'])

                ax.plot(yAx, xAx, c='r', lw=1)
                ax.plot(yCnt, xCnt, c='k', lw=1)
                # for seg in range(len(S[0])):
                #    if seg%2 ==0:
                #        ax.plot([S[0][seg][0][1], S[0][seg][1][1]], [S[0][seg][0][0], S[0][seg][1][0]], c='white', alpha=0.5)

            except:
                ax.scatter(0.1 * image.shape[0], 0.1 * image.shape[1], c='k', marker='x', s=20)

        if mark_filtered:
            if ID in filtered_gastrus:
                df.at[ix, 'Flag'] != 'Keep'
                ax.scatter(0.2 * image.shape[0], 0.1 * image.shape[1], c='r', marker='x', s=20)

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_title(f'i:{ix}\n{df.at[ix, param]}', fontsize=5, pad=0)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        if ix == 0:
            AddScale_well(ax, 100 / um_per_pixel, 100, c='k')

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)

    if save:
        folder = config['plots_folder']
        ending = config['overview_plot']['format']
        if (param == 'imageID') and (len(df.expID.unique()) == 1):
            savename = folder + f'{df.at[ix, "expID"]}_overview{ending}'
        else:
            savename = folder + f'{df.at[ix, param]}_overview{ending}'
        plt.savefig(savename, dpi=200, transparent=True)
        print('Successfully saved the overview plot under ', savename)

    else:
        plt.show()
        plt.close()

    config['overview_plot'] = {'ch_to_use': config['overview_plot']['ch_to_use'], 'format': ending, 'param': param,
                               'morph_analysis': morph_analysis, 'mark_filtered': mark_filtered,
                               'filtered_gastrus': filtered_gastrus, 'ncol': ncol, 'saved': save, 'cmap': cmap}

    outlier_dict = {key: list() for key in config['aliases']}
    config['outliers'] = outlier_dict
    configs.update_configs(config)
    
    
    
    
def get_image_ax_multi_pt(df, ix, ax, CNT=True, MA=True, SEG=False, mark=False):
    """
    Displays a processed image with optional overlays of the medial axis, contour, and segmentation for visualization.

    This function displays an image specified by a row in a DataFrame on a given Matplotlib
    axis, along with optional overlays such as contours, medial axes, segmented lines,
    and marking. The specific overlays visualized depend on the flags provided as parameters.

    :param df: Pandas DataFrame containing image metadata and optional data for overlays.
    :param ix: Integer index specifying the row in the DataFrame.
    :param ax: Matplotlib axis on which the image and overlays are rendered.
    :param CNT: Boolean value to indicate whether to overlay the contour of the image.
    :param MA: Boolean value to indicate whether to overlay the medial axis.
    :param SEG: Boolean value to indicate whether to overlay the segmented lines.
    :param mark: Boolean value to indicate whether to add a marker if the image is flagged.
    :return: Returns a Matplotlib axis (ax) containing the processed visualization.
    """
    imagepath = df.at[ix, 'Path_data']
    image = imread(imagepath)[:, :, 2]

    ax.imshow(image, cmap='gray')
    ax.axis('scaled')
    ax.axis('off')
    # pipl.AddScale_well(ax, 100/um_per_pixel, 100, 100)

    if CNT:
        try:
            cnt = np.array(df.at[ix, 'Mask'])
            xCnt, yCnt = cnt.T
            ax.plot(yCnt, xCnt, c='pink', alpha=0.75, lw=1)
        except:
            print('No contour could be calculated')

    if MA:
        try:
            medax = np.array(df.at[ix, 'MedialAxis'])
            xAx, yAx = medax.T
            ax.plot(yAx, xAx, c='white', alpha=0.75, lw=1)
        except:
            print(f'No medial axis could be calculated for index {ix}')

    if SEG:
        try:
            S = np.array(df.at[ix, 'Segments'])
            for seg in range(len(S)):
                if seg % 2 == 0:
                    ax.plot([S[seg][0][1], S[seg][1][1]], [S[seg][0][0], S[seg][1][0]], c='white', alpha=0.5, lw=1)
        except:
            print(f'No segments could be calculated for index {ix}')

    if mark:
        if df.at[ix, 'Flag'] != 'Keep':
            ax.scatter(image.shape[0] * 0.1, image.shape[1] * 0.1, marker='x', color='k', s=50)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.set_title(f"{ix} {df.at[ix, 'imageID'].split('-1')[0]}", fontsize=5)

    return ax



def plot_overview_multi_pt(path, df, ncol=5, save=False, CNT=True, MA=True, SEG=False, mark=False, name=''):
    """
    Generate an overview plot with multiple subplots for data visualization. The function organizes
    a set of images, processes, and optionally saves them in a structured layout. It allows for
    customization such as applying markers, setting various data analysis options, and saving output
    files with designated naming conventions.

    :param path: The file path where the output file should be saved if the save option is enabled.
    :type path: str
    :param df: A pandas DataFrame containing the data used for generating the plots. It must include
        the necessary indices for processing.
    :type df: pandas.DataFrame
    :param ncol: The number of columns to layout in the subplot grid. Default value is 5.
    :type ncol: int, optional
    :param save: A flag that determines whether to save the generated plot as a file. Defaults to False.
    :type save: bool, optional
    :param CNT: Determines whether to include CNT data in the plot generation process.
    :type CNT: bool, optional
    :param MA: Determines whether to include moving average (MA) in the plot.
    :type MA: bool, optional
    :param SEG: A flag indicating whether to include segmentation (SEG) data in the plotting process.
    :type SEG: bool, optional
    :param mark: A flag that specifies whether markers should be added to the plot.
    :type mark: bool, optional
    :param name: An optional string to add custom naming while saving the file. If not passed,
        default naming conventions are applied.
    :type name: str, optional
    :return: The generated matplotlib plot object containing the multi-subplot visualization.
    :rtype: matplotlib.pyplot.Figure
    """
    ncol = ncol
    nrow = math.ceil(len(df) / ncol)

    if len(df) < ncol:
        fig, axes = plt.subplots(1, len(df), figsize=(len(df), 1))
    
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

        for i in range(nrow * ncol - len(df)):
            axes[nrow - 1, ncol - i - 1].set_visible(False)
            # axes = axes.flatten()

    for ax, ix in zip(axes.flatten()[:len(df)], df.index):
        get_image_ax_multi_pt(df, ix, ax, CNT, MA, SEG, mark)

    invis = len(axes.flatten()) - len(df)

    for j in range(invis):
        axes[nrow - 1, ncol - j - 1].set_visible(False)

    fig.suptitle(path.rsplit('/', 2)[1])
    plt.tight_layout(pad=.5)

    if save:
        file_name = f'image_overview_{name}.pdf'
        if mark:
            file_name = f'image_overview_morph_analysis_{name}.pdf'
        plt.savefig(os.path.join(path, file_name), dpi=200, transparent=True)

    return plt


def crop_image(img, dfOI, ix):
    try:
        mask_cnt = np.array(dfOI.loc[ix, 'Mask'])

        x_min, x_max = np.min(np.array(mask_cnt)[:, 1]), np.max(np.array(mask_cnt)[:, 1])
        center = x_min + (x_max - x_min) / 2
        dist = x_max - x_min

        if dist < 1024:
            xl = center - 1024 / 2
            xr = center + 1024 / 2

            if xl < 0:
                plus_r = xl * -1
                xl = 0
                xr += plus_r

            if xr > 1280:
                minus_l = xr - 1280
                xr = 1280
                xl -= minus_l

        else:
            print('unfortunately too large.')
    except:
        xl = int((1280 - 1024) / 2)
        xr = int(1280 - xl)

    img_crop = img[:, int(xl):int(xr)]

    return img_crop


###########################################################
# profile analysis plotting

#def get_xbin():
# instead of calc x bins with some percentage cut off, only calculate before plotting and provide option to specify
# cutoff amount


#simple all profiles with raw intensity plot
# all profile with mean
##
