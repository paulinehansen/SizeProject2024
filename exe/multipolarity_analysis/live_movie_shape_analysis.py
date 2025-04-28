import os
import time
from sys import path
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/multipolarity_analysis', '')
path.insert(1, modules)

import modules.configs_setup as configs
import modules.raw_live_analysis as raw
import modules.processed_data_analysis as pda

# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with czi files')
parser.add_argument('condition', type=str, help='Specify condition to analyze: "150", "300", "600", "1200", or "all".')
args = parser.parse_args()

data_folder = args.data_path
cond = args.condition

# load configs
cnfgs = configs.load_configs(data_folder)
print(cnfgs.keys())


if cond == 'all':
    folder_OIs = [c for c in cnfgs['data_paths'].keys() if not c == 'ending']
else:
    folder_OIs = [cond]
print('Analyzing conditions ', folder_OIs)


# iterate over all folders
for folder_OI in folder_OIs:

    # load lists of paths to dfs and movies
    _, df_list = pda.get_file_list(cnfgs['data_paths'][folder_OI], ending=cnfgs['data_paths']['ending'])

    print('Starting analysis for condition ', folder_OI)
    start_group_time = time.time()

    # compute analysis for all dfs in folder
    raw.compute_cumulated_live_movie_mask_analysis(df_list, save=True)

    group_time = time.time()
    print(f" analysis time for group {folder_OI} --- %s seconds ---" % (group_time - start_group_time))
    # TODO: add configs things here as well

os._exit(1)
