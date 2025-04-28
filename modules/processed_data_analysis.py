import os
import json
import scipy
import shapely
import numpy as np
import pandas as pd
from shapely import Polygon
from itertools import combinations
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline


def get_file_list(folder_path, ending=''):
    """
    Retrieve a list of files with a specific ending from a given folder. The function
    provides both a relative and an absolute file path list of the files that match
    the specified ending. The files are sorted alphabetically.

    :param folder_path: Path to the directory containing files.
        The path should end with a delimiter (e.g., slash for Linux/macOS
        or backslash for Windows).
    :type folder_path: str
    :param ending: File ending to filter by. Default is an empty string,
        which retrieves all files in the folder.
    :type ending: str
    :return: A tuple containing two lists:
        - A list of file names (relative paths) that match the ending.
        - A list of absolute paths to these files in the specified folder.
    :rtype: tuple[list[str], list[str]]
    """
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(ending)])
    abs_file_list = [folder_path + f for f in file_list]
    return file_list, abs_file_list

def load_processed_df(path, all=True):
    """
    Loads and processes a JSON file into a pandas DataFrame. The function reads the JSON file
    from the specified file path and attempts to create a DataFrame from the data. If the
    `all` parameter is set to False, the DataFrame is filtered to include only rows where the
    'Flag' column has the value 'Keep'. The total number of rows in the resulting DataFrame is
    printed.

    :param path: The file path to the JSON file to be loaded.
    :type path: str
    :param all: Boolean flag indicating whether to keep all rows in the DataFrame or filter
        rows where the 'Flag' column has the value 'Keep'. Default is True.
    :type all: bool
    :return: A pandas DataFrame created from the contents of the JSON file. If `all` is False,
        only rows with the 'Keep' flag are included.
    :rtype: pd.DataFrame
    """
    with open(path) as project_file:
        df = json.load(project_file)
        try:
            df = pd.DataFrame(df["data"], columns=df["columns"])
        except:
            try:
                df = pd.DataFrame(df["data"])
            except:
                df = pd.DataFrame.from_dict(df)
        if not all:
            df = df[df['Flag'] == 'Keep']
        print('# rows: ', len(df))

    return df

def mark_outliers(df, outliers, cnfgs):
    for ix in df.index:
        if df.at[ix, 'imageID'] in outliers:
            df.at[ix, 'Flag'] = 'outlier'
            
    cnfgs['outliers_flagged'] = {'flagged': True, 'label': 'outlier'}
    
    return df


'''
def preprocess_dfs(cnfgs, colsOI):
    # read in and load all data frames
    data_dict = {}
    for k, v in cnfgs['data_paths'].items():
        df_path = v + 'results_IF/df_all.json'
        df = load_processed_df(df_path)
        exp = v.rsplit('/', 2)[1]
        
        # indicate previously defined outliers
        df = mark_outliers(df, cnfgs['outliers'][exp], cnfgs)
        # convert units from pixel to metric units
        df = convert_units_to_um(df, args=['Length_MA', 'Volume_MA'])
        # div intensities by area
        channelsOI = cnfgs['dict_channel'][[*cnfgs['dict_channel'].keys()][0]].keys()
        df = div_by_area(df, channels=channelsOI)

        
        df_clean = df[df['Flag'] == 'Keep']
        df_clean.loc[:, 'expID'] = exp
        df_clean.loc[:, 'group'] = [i.split('_S')[0] for i in df_clean['imageID']]
        df_clean.loc[:, 'final_size'] = [cnfgs['final_size'][group] for group in df_clean['group']]
    
        df_profiles = df_clean[colsOI]
        data_dict[v] = df_profiles
         
    return data_dict
'''

def preprocess_dfs(cnfgs, colsOI):
    # read in and load all data frames
    data_dict = {}

    if cnfgs.get('data_paths', None):

        for k, v in cnfgs['data_paths'].items():
            df_path = v + 'results_IF/df_all.json'
            df = load_processed_df(df_path)
            exp = v.rsplit('/', 2)[1]

            # indicate previously defined outliers
            df = mark_outliers(df, cnfgs['outliers'][exp], cnfgs)
            # convert units from pixel to metric units
            df = convert_units_to_um(df, args=['Length_MA', 'Volume_MA'])
            # div intensities by area
            channelsOI = cnfgs['dict_channel'][[*cnfgs['dict_channel'].keys()][0]].keys()
            df = div_by_area(df, channels=channelsOI)

            df_clean = df[df['Flag'] == 'Keep']
            df_clean.loc[:, 'expID'] = exp
            df_clean.loc[:, 'group'] = [i.split('_S')[0] for i in df_clean['imageID']]
            df_clean.loc[:, 'final_size'] = [cnfgs['final_size'][group] for group in df_clean['group']]

            df_profiles = df_clean[colsOI]
            data_dict[v] = df_profiles

    else:
        df_path = cnfgs['merged']['save_name']
        df = load_processed_df(df_path)
        exp = cnfgs['data_path'].rsplit('/', 2)[1]

        # indicate previously defined outliers
        df = mark_outliers(df, sum(cnfgs['outliers'].values(), []), cnfgs)
        # convert units from pixel to metric units
        df = convert_units_to_um(df, args=['Length_MA', 'Volume_MA'])
        # div intensities by area
        channelsOI = [*cnfgs['dict_channel'][0].keys()]
        df = div_by_area(df, channels=channelsOI)

        df_clean = df[df['Flag'] == 'Keep']
        df_clean.loc[:, 'group'] = [i.split('_S')[0] for i in df_clean['imageID']]
        df_clean.loc[:, 'final_size'] = [cnfgs['final_size'][group] for group in df_clean['group']]

        df_profiles = df_clean[colsOI]
        data_dict[exp] = df_profiles


    return data_dict

def merge_clean_dfs_from_dict(data_dict):
    # merge separate data frames to a single one
    joint_df = pd.DataFrame()
    for k, v in data_dict.items():
        joint_df = pd.concat([joint_df, v], ignore_index=True)
        
    return joint_df

def filter_outliers(outlier_dict, df, label='outlier'):
    """
    Filters outliers from a DataFrame based on the provided dictionary of outlier keys and
    updates a specified flag column for matching records.

    """
    for k, v in outlier_dict.items():
        outliers = [k + '_S' + o for o in v]
        index = df[df['imageID'].isin(outliers)].index
        df.loc[index, 'Flag'] = label

    return df

def clean_df(df):
    """
    Filters a DataFrame to include only rows where the 'Flag' column has the value 'Keep'.

    This function takes a pandas DataFrame and applies a filtering condition, retaining
    only rows that have the value 'Keep' in the 'Flag' column. The filtered DataFrame
    is then returned. It is assumed that the input DataFrame contains a 'Flag' column.

    :param df: The pandas DataFrame to be filtered
    :type df: pandas.DataFrame
    :return: A pandas DataFrame containing only the rows where the 'Flag' column
        equals 'Keep'
    :rtype: pandas.DataFrame
    """
    return df[df['Flag'] == 'Keep']

def convert_units_to_um(df, args=[]):
    """
    Convert specific measurement values to micrometer (μm) or other units by applying conversion
    factors based on the 'um_per_pixel' value present in the dataframe.
    This function creates new columns in the dataframe for the converted values.

    :param df: Input dataframe containing measurement data and the conversion factor
        'um_per_pixel'.
    :type df: pandas.DataFrame
    :param args: List of measurement types to convert. Possible values include:
        'Length_MA', 'Volume_MA', 'Areas', 'cnt_Area', 'cnt_Perimeter', and
        'centroid_dist'. Each value adds respective converted columns.
    :type args: list[str], optional
    :return: The dataframe with additional columns containing converted values.
    :rtype: pandas.DataFrame
    """
    if ('Length_MA' in args) and not ('Length_MA_um' in df.columns):
        df['Length_MA_um'] = df['Length_MA'] * df['um_per_pixel']
    if ('Volume_MA' in args) and not ('Volume_MA_mm3' in df.columns):
        df['Volume_MA_mm3'] = df['Volume_MA'] * df['um_per_pixel'] ** 3 / (10 ** 9)
        df['Volume_eq_mm3'] = df['Length_MA'] ** 3 * np.pi / 6 * df['um_per_pixel'] ** 3 / (10 ** 9)
    if ('Areas' in args) and not ('Area_um2' in df.columns):
        df['Area_um2'] = df['Areas'].apply(np.sum)
        df['Area_um2'] = df['Area_um2'] * df['um_per_pixel'] ** 2
    if ('cnt_Area' in args) and not ('cnt_Area_um2' in df.columns):
        df['cnt_Area_um2'] = df['cnt_Area'] * df['um_per_pixel'] ** 2
    if ('cnt_Perimeter' in args) and not ('cnt_Perimeter_um' in df.columns):
        df['cnt_Perimeter_um'] = df['cnt_Perimeter'] * df['um_per_pixel']
    if ('centroid_dist' in args) and not ('centroid_dist_um' in df.columns):
        df['centroid_dist_um'] = df['centroid_dist'] * df['um_per_pixel']

    return df

def div_by_area(df, channels=['DAPI']):
    """
    Normalizes the intensity values in the specified channels by dividing them
    by the 'Areas' column values. This operation is performed for each provided
    channel, and new columns are created with the names formatted as
    "<channel>_area_norm". The function modifies the DataFrame in-place and
    returns the updated DataFrame.

    :param df:
        The pandas DataFrame containing the data to be processed. Must include
        the specified channels and an 'Areas' column.
    :param channels:
        A list of string channel names for which normalization by area
        will be applied. Defaults to ['DAPI'].
    :return:
        The updated DataFrame with new columns added for the normalized values.
    """
    for ch in channels:
        newcol = ch + '_' + 'area_norm'
        print(newcol)

        if newcol in df.columns:
            continue
        else:
            df[newcol] = None
            dff = df[(df[ch].notnull()) & (df['Areas'].notnull())]
            dff[newcol] = dff.apply(lambda x: np.divide(x[ch], x['Areas']), axis=1)
            df.update(dff)
    return df


###########################################################
# operations to compute profiles IF experiment

def get_profile_orientation(df, orientation_channel=None, method='median'):
    """
    Determines the orientation of profiles in the provided DataFrame based on the selected
    method. The function evaluates a specified orientation channel, or determines one
    automatically if not provided, and processes the profiles to assess whether they should
    be flipped or not.

    :param df: The input pandas DataFrame containing the profile data and channel information.
    :param orientation_channel: Optional; The specific channel to evaluate orientation. If not provided, the channel with index 0
        in the last row's `channels` dictionary will be used.
    :param method: Optional; The calculation method for determining orientation. Can be 'median' or 'mean'.
        Default is 'median'.
    :return: A list indicating the orientation for each profile. Values are 1 if flipping is required,
        0 otherwise, and None for profiles where orientation could not be determined.
    :rtype: list
    """
    channel_dict = df.iloc[-1]['channels']

    if not orientation_channel:
        orientation_channel = list(channel_dict.keys())[list(channel_dict.values()).index(0)]

    flip = list()
    for i, profile in enumerate(df[f'{orientation_channel}_area_norm']):
        try:
            profile = profile[round(len(profile) * 0.1):round(len(profile) * 0.9)]
            profile = np.array(profile, dtype=float)

            if method == 'median':
                half1 = np.median(profile[: round(len(profile) / 3)])
                half2 = np.median(profile[round(len(profile) / 3) * 2:])

            if method == 'mean':
                half1 = np.mean(profile[: round(len(profile) / 3)])
                half2 = np.mean(profile[round(len(profile) / 3) * 2:])

            if half1 < half2:
                flip.append(1)
            else:
                flip.append(0)
        except:
            flip.append(None)
    return flip

def preprocess_df_for_profile_plotting(df, cnfgs, orientation_channel='Bra'):
    # specifiy extra parameters that make plotting profiles easier
    final_size = cnfgs['final_size']
    cutfuse = cnfgs['cut-fuse-groups']
    df['Area'], df['final_size'], df['cut-fused'], df['time'] = 0, 0, 0, 0
    for i in df.index:
        t, boolean = 0, 0

        if '144' in df.loc[i, 'group']:
            t = 144
        if '96' in df.loc[i, 'group']:
            t = 96
        else:
            t = 120
        if df.loc[i, 'group'] in cutfuse:
            boolean = 'Yes'
        else:
            boolean = 'No'

        df.loc[i, 'Area'] = np.nansum(df.loc[i, 'Areas'])
        df.loc[i, 'time'] = t
        df.loc[i, 'cut-fused'] = boolean
        df.loc[i, 'final_size'] = final_size[df.loc[i, 'group']]

    df['group'] = [cnfgs['groups_dict'][g] for g in df['group']]

    df_specific = df[df['final_size'] == 120050]
    for ix in df_specific.index:
        if df.loc[ix, 'Volume_MA_mm3'] > 0.02:
            df.loc[ix, 'final_size'] = 1200
            df.loc[ix, 'group'] = '1200'
        else:
            df.loc[ix, 'final_size'] = 50
            df.loc[ix, 'group'] = '50'

    # orientate the profiles aling AP axis
    df['correct_orientation'] = get_profile_orientation(df, orientation_channel=orientation_channel, method='mean')

    return df

def convert_16_to_8_bit(df, channels):
    # set up new data frames to match intensity value scale
    new_df_8bit = df.copy()

    matrix = np.ndarray((len(df), 194))
    for i, ix in enumerate(df.index):
        if not '2402' in df.at[ix, 'expID']:
            for chan in channels:
                data_ch = np.divide(np.array(df.at[ix, f'{chan}'], dtype=float), 256)
                data_dapi = np.divide(np.array(df.at[ix, f'DAPI'], dtype=float), 256)
                data_ch_norm = np.divide(np.array(df.at[ix, f'{chan}_area_norm'], dtype=float), 256)
                data_dapi_norm = np.divide(np.array(df.at[ix, f'DAPI_area_norm'], dtype=float), 256)
    
                new_df_8bit.at[ix, f'{chan}'] = data_ch
                new_df_8bit.at[ix, f'DAPI'] = data_dapi
                new_df_8bit.at[ix, f'{chan}_area_norm'] = data_ch_norm
                new_df_8bit.at[ix, f'DAPI_area_norm'] = data_dapi_norm

    return new_df_8bit

def flip_profiles(df, channels=[], bool_col_name='correct_orientation'):
    """
    Flip profiles in specified channels of a DataFrame based on the value of a boolean column.

    This function reverses the profiles in the specified channels of the DataFrame rows
    where the value in the `bool_col_name` column is 0. The `bool_col_name` column must
    exist in the DataFrame, and at least two channels must be provided for flipping.

    :param df: The DataFrame containing the profile data and the boolean orientation
        column. The DataFrame must include the `bool_col_name` column and the specified
        `channels` columns.
    :type df: pandas.DataFrame
    :param channels: A list of column names representing the channels whose profiles
        need to be flipped. The list must contain at least two channel names.
    :type channels: List[str]
    :param bool_col_name: The column name in the DataFrame representing the boolean
        flag for orientation. Defaults to 'correct_orientation'.
    :type bool_col_name: str
    :return: A modified DataFrame where profiles in the specified `channels` are flipped
        based on the value of the `bool_col_name` column.
    :rtype: pandas.DataFrame
    """
    if (bool_col_name in df.columns) and (len(channels) > 1):
        df['correct_orientation'] = [int(i) for i in df['correct_orientation']]
        for ix in df.index:
            if df.at[ix, bool_col_name] == 0:
                for ch in channels:
                    if df.at[ix, ch]:
                        try:
                            profile = df.at[ix, ch][::-1]
                            df.at[ix, ch] = profile
                        except:
                            print('No flipping for ', df.at[ix, 'imageID'], 'channel:', ch)
    else:
        print('Could not flip profiles, some information missing')

    return df

def normalize_data_by_exp_mean(df, chan, exp, filter_cond=['3000'], sorted_groups=['50', '50x6', '300', '1200/4', '300x4', '1200', '3000']):

    df_e = df[df['expID']==exp]
    for c in filter_cond:
        df_e = df_e[df_e['group'] != c]
        
    ### get dictionary of min max values per condition (0.1 offset)
    min_max_dict = {}
    mean_profile_dict = {}
    
    for t in df_e.time.unique():
        df_t = df_e[df_e['time']==t]
        
        for i, s in enumerate(sorted_groups):
            df_s = df_t[df_t['group'] == s]
            
            if len(df_s) < 3:
                continue
            
            mean_profile_dict[f'{t}-{s}'] = get_meanprofile(df_s, [f'{chan}_area_norm'])
            l = len(mean_profile_dict[f'{t}-{s}'][f'{chan}_area_norm'])
            profile = mean_profile_dict[f'{t}-{s}'][f'{chan}_area_norm'][round(l*0.1):-round(l*0.1)]
    
            minI, maxI = np.nanmin(profile), np.nanmax(profile)
            min_max_dict[f'{chan}-{t}-{s}'] = minI, maxI
            
            
    ## normalize each gastruloids profile by exp min max and save data in data frame (0-1)
    df_e[f'{chan}_mean_exp_norm'] = df_e[f'{chan}']
    for t in df_e.time.unique():
        df_t = df_e[df_e['time']==t]
        
        for i, s in enumerate(sorted_groups):
            df_s = df_t[df_t['group'] == s]
            
            if len(df_s) < 3:
                continue
                    
            xs = np.linspace(0, 1, len(df_s.at[df_s.index[0], f'{chan}_area_norm']))
            lm = len(xs)
            minI, maxI = min_max_dict[f'{chan}-{t}-{s}']
            
            
            for ix in df_s.index:
                p = df_s.at[ix, f'{chan}_area_norm']    
                p = np.array(p, dtype=float)
                norm_p = (p - minI) / (maxI - minI)
                
                df_e.at[ix, f'{chan}_mean_exp_norm'] = norm_p

    
    return df_e, mean_profile_dict

def normalize_data_by_cond_mean(df, chan, filter_cond=[],sorted_ids=[]):
    for c in filter_cond:
        df = df[df['expID'] != c]

    ### get dictionary of min max values per condition (0.1 offset)
    min_max_dict = {}
    mean_profile_dict = {}

    for t in df.time.unique():
        df_t = df[df['time'] == t]

        for i, s in enumerate(sorted_ids):
            df_s = df_t[df_t['expID'] == s]

            if len(df_s) < 3:
                continue

            mean_profile_dict[f'{t}-{s}'] = get_meanprofile(df_s, [f'{chan}_area_norm'])
            l = len(mean_profile_dict[f'{t}-{s}'][f'{chan}_area_norm'])
            profile = mean_profile_dict[f'{t}-{s}'][f'{chan}_area_norm'][round(l * 0.1):-round(l * 0.1)]

            minI, maxI = np.nanmin(profile), np.nanmax(profile)
            min_max_dict[f'{chan}-{t}-{s}'] = minI, maxI

    ## normalize each gastruloids profile by exp min max and save data in data frame (0-1)
    df[f'{chan}_mean_exp_norm'] = df[f'{chan}']
    for t in df.time.unique():
        df_t = df[df['time'] == t]

        for i, s in enumerate(sorted_ids):
            df_s = df_t[df_t['expID'] == s]

            if len(df_s) < 3:
                continue

            xs = np.linspace(0, 1, len(df_s.at[df_s.index[0], f'{chan}_area_norm']))
            lm = len(xs)
            minI, maxI = min_max_dict[f'{chan}-{t}-{s}']

            for ix in df_s.index:
                p = df_s.at[ix, f'{chan}_area_norm']
                p = np.array(p, dtype=float)
                norm_p = (p - minI) / (maxI - minI)

                df.at[ix, f'{chan}_mean_exp_norm'] = norm_p

    return df, mean_profile_dict

def get_meanprofile(df, keys, **kwargs):
    """
    Compute the mean profile for given keys from a DataFrame.

    This function iterates over the provided keys and, for each key, computes
    the mean profile across all non-null indices in the DataFrame. The mean
    profile is calculated as the column-wise average of the numeric `profile`
    values provided under the given key, after replacing `None` values with
    NaN for computation compatibility. If no valid profiles are found for
    a key, the mean profile is defaulted to [NaN, NaN, NaN]. The results
    are stored in a dictionary, where the keys are the input keys and the
    values are the computed mean profiles.

    :param df: The DataFrame containing the data with profiles to calculate
        mean values for.
    :type df: pandas.DataFrame
    :param keys: A list of keys indicating the columns in the DataFrame
        for which mean profiles should be computed.
    :type keys: list
    :param kwargs: Additional keyword arguments (not actively used in the function).
    :type kwargs: dict
    :return: A dictionary where the keys are the input keys from `keys`
        and the values are the computed mean profiles corresponding to those keys.
    :rtype: dict
    """
    dic = {}
    for key in keys:
        profiles = []
        for i in df[df[key].notnull()].index:
            profile = df.at[i, key]
            profile = [np.nan if v is None else v for v in profile]
            profiles.append(profile)

        if np.shape(profiles)[0] == 0:
            profiles_mean = [np.nan, np.nan, np.nan]
        else:
            profiles_mean = np.nanmean(profiles, axis=0)
        dic[key] = profiles_mean

    return dic

def get_stdprofile(df, keys, **kwargs):
    """
    Calculate the standard deviation profiles for specified keys in a DataFrame.

    This function computes the standard deviation across profiles for specified
    keys in the given DataFrame. A profile for each key is extracted, and its
    NaN values are handled where applicable. The standard deviation is computed
    only for non-empty profiles.

    :param df: The DataFrame containing the data with profile values.
    :type df: pandas.DataFrame
    :param keys: List of column names (keys) in the DataFrame for which the
        standard deviation profiles should be computed.
    :type keys: list
    :param kwargs: Additional keyword arguments (unused in the function, allows
        for flexibility in interface).
    :return: A dictionary where each specified key maps to its respective
        standard deviation profile, or NaNs if no profiles exist for the key.
    :rtype: dict
    """
    dic = {}
    for key in keys:
        profiles = []
        for i in df[df[key].notnull()].index:
            profile = df.at[i, key]
            profile = [np.nan if v is None else v for v in profile]
            profiles.append(profile)

        if np.shape(profiles)[0] == 0:
            profiles_std = [np.nan, np.nan, np.nan]
        else:
            profiles_std = np.nanstd(profiles, axis=0)
        dic[key] = profiles_std

    return dic

def get_varprofile(df, keys, **kwargs):
    """
    Computes the variance profile for specified keys in a DataFrame. The function calculates
    the variance of the values for each key in the input list from a DataFrame where the values
    are not null. It uses numpy's nanvar method to compute variance, ignoring NaN values,
    and allows additional parameters to be specified.

    :param df: Pandas DataFrame containing data for computations.
    :type df: pandas.DataFrame
    :param keys: List of column names in the DataFrame for which to compute variance profiles.
    :type keys: list
    :param kwargs: Additional keyword arguments passed directly to numpy's nanvar function.
    :return: Dictionary mapping each key to its corresponding variance profile.
    :rtype: dict
    """
    dic = {}
    for key in keys:
        profiles = []
        for i in df[df[key].notnull()].index:
            profiles.append(df.at[i, key])
        profiles_std = np.nanvar(profiles, **kwargs)
        dic[key] = profiles_std

    return dic

def get_cvprofile(df, keys, **kwargs):
    """
    Calculate the coefficient of variation profile for the given dataframe.

    This function computes the coefficient of variation (CV) profile for
    the provided dataframe. The CV is calculated for each key by dividing
    the standard deviation by the mean for the corresponding data. The
    profile is returned as a dictionary with keys corresponding to the
    provided keys and values representing the coefficient of variation.

    :param df:
        The input dataframe containing the data.
    :param keys: list
        A list of column names from the dataframe for which the
        coefficient of variation profile should be calculated.
    :param kwargs:
        Additional keyword arguments that can be passed to the underlying
        `get_meanprofile` and `get_stdprofile` functions, if required.
    :return: dict
        A dictionary where keys are the column names from the `keys` list,
        and values are the calculated coefficient of variation for each
        respective column.
    """
    dic = {}
    means = get_meanprofile(df, keys, **kwargs)
    stds = get_stdprofile(df, keys, **kwargs)

    for key in keys:
        dic[key] = np.divide(stds[key], means[key])

    return dic

def get_normalized_mean_profile(m, offset=0.1, cutoff=True):
    """
    Compute the normalized mean profile of the given data array.

    This function normalizes the input data `m` by scaling it to fit within the range [0, 1].
    If specified, a portion of the data at both ends can be truncated based on the `offset`
    parameter. The option `cutoff` determines whether this truncation is applied. A linearly
    spaced array `x` is also computed based on the length of the (possibly truncated) array.

    :param m: Input data array to be normalized.
    :type m: numpy.ndarray
    :param offset: Fraction of the data to be removed from both ends, based on its length.
        This should be a value between 0 and 1.
    :type offset: float
    :param cutoff: Indicates whether to apply the offset truncation to the input data array.
    :type cutoff: bool
    :return: A tuple containing two elements: the normalized data array `normed_m` and
        a linearly spaced array `x` corresponding to the length of the processed data array.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    lm = int(len(m)*offset)
    x = np.linspace(0, 1, len(m))
    if cutoff:
        x = x[lm:-lm]
        m = m[lm:-lm]
    normed_m = (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m))

    return normed_m, x

def findextrema10(profile, k=10):
    ''' Find the extrema of a given 1D profile using the k% minimal and maximal points. '''
    #TODO: automatize the application of the function to a list of key OR
    # change all function here so that pipelines are a serie of df.apply functions?

    ordered = np.argsort(profile)
    Imin = np.mean(profile[ordered[:int(len(ordered)/k)]])
    Imax = np.mean(profile[ordered[-int(len(ordered)/k):]])

    return Imin, Imax

def findboundary_extrema(profile, k=10, xmin=0.0, xmax=1.0, gradient=2):
    '''Determine the position of a boundary defined as the place
    where the intensity is going from under to above (Imax+Imin)/2 '''


    Imin, Imax= findextrema10(profile, k)
    imin = int(len(profile)*xmin)
    imax = int(len(profile)*xmax)

    if len( np.argwhere(np.diff(np.sign(profile[imin:imax]-(Imax+Imin)/2)) == gradient)) > 0:
        iboundary = imin + np.argwhere(np.diff(np.sign(profile[imin:imax]-(Imax+Imin)/2)) == gradient)
    Iboundary = [profile[x] for x in iboundary]

    return iboundary, Iboundary



###########################################################
# operations to perform BF multi-timepoint analysis 

def prepare_multi_tp_df(df):
    
    df_small = df.drop(['Mask', 'Areas', 'marker', 'ncells', 'Saved', 'BBOX', 'MedialAxis', 'Segments', 'Length_MA', 'Volume_MA', 'Volumes', 'Lengths', 'Thicknesses'], axis=1)
    
    df_small['Exp_Date'] = [exp.split('_')[0] for exp in df_small['exp']]
    df_small['Microscope'] = 'OlympusCKX41'
    
    df_small = df_small.rename(columns={'absPath': 'Path_data', 'expID': 'Name', 'time': 'TimePt', 'um_per_pixel': 'Pixel_size',
                        'Rep':'Replicate', 'magn':'Objective', 'size': 'Condition', 'Contour':'Mask'})
    
    new_cols = ['Point_in', 'Segments', 'MedialAxis', 'MA_length_um','Solidity', 'AR', 'Circularity', 
            'Area', 'Perimeter', 'Major_Axis_um']
    
    for new_col in new_cols:
        df_small[new_col] = np.nan
        
    return df_small

def prepare_manual_count_df(df):
    
    df_clean = df.copy()
    df_clean = df_clean[df_clean['Flag'] != 'All Bad']
    df_clean = df_clean[df_clean['Flag'] != 'Bad']
    
    df_exp_stats = pd.DataFrame(columns=['exp', 'Replicate', 'TimePt', 'group', 'Condition', 
                                     'N', 'uniaxial', 'multipolar', 'collapsed', 'other'], 
                                index=np.arange(len(df_clean.exp.unique())))
    
    df_exp_stats['exp'] = df_clean.exp.unique()
    
    for i, exp in enumerate(df_clean.exp.unique()):
        df_e = df_clean[df_clean['exp'] == exp]
        df_exp_stats.at[i, 'Replicate'] = df_e.Replicate.unique()[0]
        df_exp_stats.at[i, 'TimePt'] = df_e.TimePt.unique()[0]
        df_exp_stats.at[i, 'group'] = df_e.group.unique()[0]
        df_exp_stats.at[i, 'Condition'] = df_e.Condition.unique()[0]
        df_exp_stats.at[i, 'N'] = len(df[df['exp'] == exp])
        
    return df_exp_stats






###########################################################
# operations to perform multipolarity analysis

def prepare_multipolar_df(df):
    """
    Prepare a modified DataFrame by adding fluorescence-related columns.

    The function processes the given DataFrame, applies a transformation
    to specified columns, and generates new columns with a suffix '_fluo',
    where each entry is represented as a dictionary with a 'bf' key pointing
    to the original value from the column.

    :param df: pandas.DataFrame
        The input DataFrame with columns: 'Mask', 'Solidity', 'Perimeter',
        'AR', 'Major_Axis_um', 'Area', and 'Circularity'. These columns will
        be transformed, and new columns will be added to the DataFrame.
    :return: pandas.DataFrame
        Returns a modified DataFrame that includes all original columns and
        new columns with '_fluo' suffixed to their names. Each '_fluo' column
        contains dictionaries where the 'bf' key maps to the original column
        values.
    """
    cols = ['Mask', 'Solidity', 'Perimeter', 'AR', 'Major_Axis_um', 'Area', 'Circularity']
    for col in cols:
        new_col = col + '_fluo'
        df[new_col] = df[col].apply(lambda x: {'bf': x})

    return df

def filter_multipolarity_outliers(f, df, outliers):
    
    if '_150_' in f:
        outs = outliers['150']
    if '_300_' in f:
        outs = outliers['300']
    if '_600_' in f:
        outs = outliers['600']
    if '_1200_' in f:
        outs = outliers['1200']
    
    well = f.split('[', 1)[1].split('_1_merge', 1)[0]
    if well in outs.keys():
        
        os = outs[well]
        for i in os:
            if i < 32:
                df.at[i, 'n_fluo_poles'] = 0
                df.at[i, 'sum_fluo_mask'] = 0
                df.at[i, 'avg_fluo_mask'] = np.nan
            elif 32 < i < 47:
                df.at[i, 'n_fluo_poles'] = df.at[i, 'n_fluo_poles'] - 1
            elif i > 46:
                df.at[i, 'n_fluo_poles'] = 1
    
    return df



## extract parameters of the dynamic Mesp2 poles data
def get_multipolar_dynamics_stats(pole_data):
    
    mean = np.nanmean(pole_data, axis=0)
    var = np.nanvar(pole_data, axis=0)
    std = np.nanstd(pole_data, axis=0)
    cv = np.divide(std, mean)

    return [mean, var, std, cv]

def get_max_n_poles(pole_data):
    
    data_list=list()
    
    for i, row in enumerate(pole_data):
        data_list.append(np.nanmax(row))
        
    return data_list

def get_n_multipol(pole_data):
    
    multi_counter = 0
    for i, row in enumerate(pole_data):
        if any(row > 1):
            multi_counter += 1

    return multi_counter

def get_polarization_t(pole_data,  time_steps=np.arange(73, 144)):
    
    data_list = list()
    
    for i, row in enumerate(pole_data):
        if len(row) == len(time_steps):
            for x, t in zip(row, time_steps):
                if x > 0:
                    data_list.append(t)
                    break
                    
        else:
            print('Number of specified time steps and time steps in data are not matching for index ', i)
            
    return data_list

def get_uniaxial_t(pole_data, time_steps=np.arange(73, 144)):
    
    data_list = list()
    
    for i, row in enumerate(pole_data):
        if len(row) == len(time_steps):
            for z, (x, t) in enumerate(zip(row, time_steps)):
                d_late = row[z:]
                if len(np.unique(d_late)) == 1:
                    if np.unique(d_late)[0] == 1:
                        data_list.append(t)
                        break
        else:
            print('Number of specified time steps and time steps in data are not matching for index ', i)
            
    return data_list

def get_merging_t(pole_data, time_steps=np.arange(73, 144)):
    
    data_list = list()
    
    for i, row in enumerate(pole_data):
        if len(row) == len(time_steps):
            if any(row > 1):
                for z, (x, t) in enumerate(zip(row, time_steps)):
                    d_late = row[z:]
                    if len(np.unique(d_late)) == 1:
                        if np.unique(d_late)[0] == 1:
                            data_list.append(t)
                            break
        
        else:
            print('Number of specified time steps and time steps in data are not matching for index ', i)
    
    return data_list

def get_distance_between_poles(pole_coords):
    
    distances = list()
    for i in pole_coords:
        dists = list()
        
        if len(i) == 2:
            a = np.array(i[0])
            b = np.array(i[1])
            dist = np.linalg.norm(a-b)
            dists.append(dist)
        
        elif len(i) > 2:
            for a, b in combinations(i, 2):
                dist = np.linalg.norm(np.array(a)-np.array(b))
                dists.append(dist)
                
        else:
            dist = np.nan
            dists.append(dist)
        
        distances.append(dists)
    
    return distances








###########################################################
# operations to perform statistical live movie analysis

def col_diff_Major_Minor(df):
    """
    Compute the difference between major and minor axis using:
       diff_Major_Minor = Major - (Major / AR)
    """
    df_analysis = df.copy()
    df_analysis.insert(0, "diff_Major_Minor", " ")

    for i in range(len(df_analysis)):
        Major = df_analysis.at[i, 'Major_Axis_um']
        AR = df_analysis.at[i, 'AR']
        diff_Major_Minor = []
        for j in range(len(Major)):
            diff_Major_Minor.append(Major[j] - (Major[j] / AR[j]))
        df_analysis.at[i, 'diff_Major_Minor'] = diff_Major_Minor

    return df_analysis

def get_condition_as_num(cond_str):
    """
    Convert condition string, like '100shape', to an integer (e.g., 100).
    """
    cond_sub = cond_str[0:-5]
    return int(cond_sub)

def get_cell_number(name_file):
    """
    Extract the number of cells from a filename string.
    E.g., 'WT129sv_600c_shape.json' -> 600
    """
    ncells = name_file.split('WT129sv_')[1].split('c')[0]
    return int(ncells)

def min_max_normalize(df, metrics):
    """
    Normalize the data (0 to 1) for each condition and metric.
    Returns the normalized DataFrame and normalization parameters.
    """
    normalized_df = df.copy()
    normalization_params = {}
    for metric in metrics:
        normalization_params[metric] = {}
        for condition in df['Condition_int'].unique():
            condition_df = df[df['Condition_int'] == condition]
            time_mean = condition_df.groupby('Time (h)')[metric].mean()
            global_min = time_mean.min()
            global_max = time_mean.max()
            normalized_df.loc[df['Condition_int'] == condition, metric] = (
                                                                                  condition_df[metric] - global_min
                                                                          ) / (global_max - global_min)
            normalization_params[metric][condition] = (global_min, global_max)
    return normalized_df, normalization_params

def filter_peaks_troughs(df, metrics, threshold=0.5, prominence=0.5, distance=1):
    """
    Identify peaks and troughs and replace them with NaN using scipy.signal.find_peaks.
    """
    peaks_df = df.copy()
    for metric in metrics:
        for condition in df['Condition_int'].unique():
            condition_df = df[df['Condition_int'] == condition]
            peaks, _ = find_peaks(condition_df[metric], threshold=threshold,
                                  prominence=prominence, distance=distance)
            troughs, _ = find_peaks(-condition_df[metric], threshold=threshold,
                                    prominence=prominence, distance=distance)
            peaks_df.loc[condition_df.index[peaks], metric] = np.nan
            peaks_df.loc[condition_df.index[troughs], metric] = np.nan
    return peaks_df

def filter_peaks_troughs_array(array, threshold=0.5, prominence=0.5, distance=1):
    ar = np.array(array)
    peaks, _ = find_peaks(ar, threshold=threshold, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-ar, threshold=threshold, prominence=prominence, distance=distance)
    out = list(peaks) + list(troughs)
    arr = ar.copy()

    for o in out:
        arr[o] = np.nan

    return arr

def replace_negative_with_nan(df, metrics):
    """
    Replace negative values with NaN (except for Circularity).
    """
    filtered_df = df.copy()
    for condition in df['Condition_int'].unique():
        for metric in metrics:
            if metric != 'Circularity':
                condition_df = filtered_df[filtered_df['Condition_int'] == condition].copy()
                outlier_condition = (condition_df[metric] < 0)
                filtered_df.loc[condition_df[outlier_condition].index, metric] = np.nan
    return filtered_df

def filter_outliers(df, metrics, window=6, sigma=3):
    """
    Filter outliers using a local 3-sigma rule within a 'window' range of hours.
    """
    filtered_df = df.copy()
    for condition in df['Condition_int'].unique():
        for metric in metrics:
            condition_df = filtered_df[filtered_df['Condition_int'] == condition].copy()
            for time in condition_df['Time (h)'].unique():
                time_window = condition_df[
                    (condition_df['Time (h)'] >= time - window) &
                    (condition_df['Time (h)'] <= time + window)
                    ]
                mean_val = time_window[metric].mean()
                std_val = time_window[metric].std()
                lower_bound = mean_val - sigma * std_val
                upper_bound = mean_val + sigma * std_val
                outliers = (
                        (condition_df['Time (h)'] == time) &
                        ((condition_df[metric] < lower_bound) | (condition_df[metric] > upper_bound))
                )
                filtered_df.loc[condition_df[outliers].index, metric] = np.nan
    return filtered_df

def filter_outliers_array(array, window=6, sigma=3):
    filtered_arr = array.copy()

    for t in range(len(array))[6:]:
        time_window = array[t - window:t + window]
        mean_val = np.nanmean(time_window)
        std_val = np.nanstd(time_window)
        lower_bound = mean_val - sigma * std_val
        upper_bound = mean_val + sigma * std_val

        if lower_bound >= array[t] or array[t] >= upper_bound:
            filtered_arr[t] = np.nan

    return filtered_arr

def restore_original_data(original_df, filtered_normalized_df, normalization_params, metrics):
    """
    Restore normalized data back to original scale using stored min/max parameters.
    """
    restored_df = original_df.copy()
    for metric in metrics:
        for condition in original_df['Condition_int'].unique():
            condition_indices = original_df[original_df['Condition_int'] == condition].index
            global_min, global_max = normalization_params[metric][condition]
            normalized_values = filtered_normalized_df.loc[condition_indices, metric]
            restored_values = normalized_values * (global_max - global_min) + global_min
            restored_df.loc[condition_indices, metric] = restored_values
    return restored_df

def find_onset_time_midpoint(time_values, data, weights, s_factor=1):
    """
    Use a UnivariateSpline to find onset time where data crosses midpoint
    between min and max.
    """
    spline = UnivariateSpline(time_values, data, w=weights, s=s_factor)
    spline_values = spline(time_values)
    min_val = np.min(spline_values)
    max_val = np.max(spline_values)
    midpoint_value = (min_val + max_val) / 2
    onset_idx = np.where(spline_values >= midpoint_value)[0][0]
    onset_time = time_values[onset_idx]
    return onset_time, spline, spline_values, midpoint_value

def find_onset_time_cost_function(time_values, data):
    """
    Find onset time by minimizing cost function C = sigma1^2 + sigma2^2,
    where the data is split into two segments at each possible boundary.
    """
    cost_values = []

    # We skip the first two and last two points to avoid tiny segments
    for t_boundary in range(2, len(time_values) - 2):
        data1 = data[:t_boundary]
        data2 = data[t_boundary:]
        sigma1_squared = np.var(data1)
        sigma2_squared = np.var(data2)
        cost = sigma1_squared + sigma2_squared
        cost_values.append(cost)

    min_cost_index = np.argmin(cost_values)
    onset_time = time_values[min_cost_index + 2]  # offset by 2
    return onset_time, cost_values










# statistical analysis of profiles











###########################################################
# operations to compute morphological parameters from contour
def get_shape(cnt):
    """
    Determines the geometric shape and type of a given set of contours
    and returns a structured dictionary representation of the shape
    along with a status flag.

    This function attempts to map a list of contours to its polygonal
    representation. If the mapping is unsuccessful, it catches any
    exception and assigns an error flag.

    :param cnt: A list of contour points that represent a geometric shape
    :type cnt: list[tuple[int, int]]
    :return:
        A tuple where the first element is a dictionary containing the
        type and coordinates of the shape, and the second element is a
        string denoting the status ('Keep' for success, 'ErrorShape'
        for failure).
    :rtype: tuple[dict, str]
    """
    cnt_shape = None
    Flag = "Keep"
    try:
        cnt_list = list(map(tuple, cnt))
        cnt_shape = {'type': 'Polygon', 'coordinates': [cnt_list]}
    except:
        print('no shape could be calculated')
        Flag = "ErrorShape"

    return cnt_shape, Flag


def get_area(shape):
    """
    Calculates the area of a given shape using its coordinates.

    This function takes a shape object in GeoJSON format and computes its area
    using the Shapely library. It extracts the coordinates of the shape, assumes
    a polygon geometry, and calculates the polygon's area.

    :param shape: The shape object in GeoJSON format containing coordinates.
    :type shape: dict

    :return: The calculated area.
    :rtype: float
    """
    return shapely.area(Polygon(shape['coordinates'][0]))


def get_perim(shape):
    """
    Calculates the perimeter of a given geometric shape represented by its
    coordinates.

    This function takes a dictionary containing structured information about the
    geometry of a shape and returns its perimeter. The function expects the input
    to include 'coordinates' in a specific GeoJSON-like format compatible with
    constructing a shapely Polygon.

    :param shape: A dictionary containing the geometry data of the shape.
        The 'coordinates' attribute should represent the boundaries of the
        polygon in a GeoJSON-compliant format.
    :type shape: dict
    :return: The perimeter of the shape based on its computed boundary length.
    :rtype: float
    """
    return shapely.length(Polygon(shape['coordinates'][0]))


def get_centroid(shape):
    """
    Computes the centroid of a given polygonal shape defined by its coordinates.

    This function accepts a shape object whose 'coordinates' key contains
    the points defining the polygon. It uses these points to calculate
    the central x and y coordinates (centroid) of the polygon.

    :param shape: Dictionary containing the geometry of the shape.
        The dictionary must include a 'coordinates' key, which holds
        a list of coordinate points defining the edges of the polygon.
    :type shape: dict
    :return: A tuple containing the x and y coordinates of the centroid.
    :rtype: tuple[float, float]
    """
    cent = Polygon(shape['coordinates'][0]).centroid

    return cent.x, cent.y


def calc_centroid_dist(xc, yc, cnt):
    """
    Calculate the Euclidean distances between the centroid and the points in the contour.

    This function computes the distances of each point in the provided
    contour `cnt` from a given centroid defined by its coordinates `xc`
    and `yc`. The result is an array of distances, where each element
    corresponds to the respective distance of a contour point from the
    centroid.

    :param xc: The x-coordinate of the centroid.
    :type xc: float
    :param yc: The y-coordinate of the centroid.
    :type yc: float
    :param cnt: An iterable of (x, y) coordinate pairs representing the
        contour.
    :type cnt: list[tuple[float, float]]
    :return: A 1-dimensional numpy array containing distances of each
        point in the contour from the centroid.
    :rtype: numpy.ndarray
    """
    d = np.zeros(len(cnt))
    for i, (x, y) in enumerate(cnt):
        d[i] = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5

    return d

#TODO: complete functions
#def calc_angle_between_poles(N_poles, centers, gast_center):
#    return angles

def calc_elongation(x):
    """
    Calculates the elongation of an object based on the given parameter.

    This function takes the input value and directly returns it. It is assumed to
    be a placeholder or a simple function for the mapping of input to output.

    :param x: Input value used for the calculation of elongation.
    :type x: Any

    :return: Returns the provided input value without modifications.
    :rtype: Any
    """
    return x

def calc_circularity(x):
    """
    Calculates the circularity of a given value. Circularity is a measure often used
    in shape analysis in mathematics and computer vision. It helps assess how
    circular a shape or value is based on the given input.

    :param x: Input value to be analyzed for circularity
    :type x: float
    :return: The calculated circularity value
    :rtype: float
    """
    return x


###########################################################
# operations on the data frame

# functions used in df.apply(function)

def get_num_fluo_segs(d):
    """
    Calculates the number of fluorescence segments in the given dictionary.

    This function takes a dictionary and computes the total number of keys present.
    If the input is not valid or any exception occurs during the computation,
    the function returns `np.nan`.

    :param d: Dictionary containing data for which the number of keys
              representing fluorescence segments is to be calculated.
    :type d: dict
    :return: The number of keys in the dictionary or `np.nan` if an
             exception occurs.
    :rtype: int or float
    """
    try:
        l = len(d.keys())
    except:
        l = np.nan
    return l

def get_ratio(x, ch='Sox2'):
    """
    Calculate and return the specified ratio value from a dictionary or default
    to 0 if the input is not a dictionary.

    :param x: Input data which might be a dictionary containing ratio values.
    :type x: dict or any
    :param ch: Specific key identifier used to extract the ratio information
        from the dictionary. Defaults to 'Sox2'.
    :type ch: str
    :return: The requested ratio value if `x` is a dictionary and contains
        the specified key, otherwise 0.
    :rtype: int or float
    """
    if type(x) == dict:
        ratio = x[f'{ch} ratio']
    else:
        ratio = 0
    return ratio