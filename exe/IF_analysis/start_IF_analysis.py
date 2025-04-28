# reads data input folder
# creates folder structure for analysis / results
# initializes cnfgs.yml file that will be the only file we are using
import os
from sys import path
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
path.insert(1, modules)
import modules.configs_setup as configs

parser = ArgumentParser()

# Required str argument providing the path to the data
parser.add_argument('data_path', type=str,
                    help='A required str argument providing the path to a folder with .czi files.')

try:
    args = parser.parse_args()
    raw_data_folder = args.data_path
    print(raw_data_folder)

    # create folder structure
    configs_folder, results_folder, plots_folder = configs.get_folder_structure(raw_data_folder)

    # read default cnfgs
    configs_init = configs.load_default_configs()

    # update with raw data folder information
    configs_init['data_path'] = raw_data_folder
    configs_init['configs_folder'] = configs_folder
    configs_init['results_folder'] = results_folder
    configs_init['plots_folder'] = plots_folder
    print(configs_init)

    # save in cnfgs folder
    configs.update_configs(configs_init)

except:
    print('You need to pass the path to a folder with .czi files as string argument.')

os._exit(1)
