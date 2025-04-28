import os
from sys import path
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
path.insert(1, modules)
import modules.raw_IF_analysis as raw
import modules.processed_data_plotter as pdp
import modules.javaThings as jv
import modules.configs_setup as configs
jv.start_jvm()
jv.init_logger()

# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with czi files')
args = parser.parse_args()
raw_data_folder = args.data_path

# load configs
cnfgs = configs.load_configs(raw_data_folder)


###################################################
# load raw data, save individual tiff maxproj of czi files, initialize a data frame that notes down the metadata
init_df = raw.initialize_czi_df(cnfgs, save=True)

# load data frames, load each tiff maxproj, perform morphological and profile analysis and save new dataframe
processed_df = raw.analyse_czi(cnfgs, profile_analysis=True, save=True)

# merge all created data frames and save as 'savename.json' in target folder
df_all = raw.merge_dfs(cnfgs, save=True)

###################################################
# visualize morphological analysis to filter out outliers
for group in df_all.expID.unique():
  df_group = df_all[df_all['expID'] == group]
  pdp.czi_plot_overview(df_group, cnfgs, param='imageID', morph_analysis=True, ncol=7, save=True, cmap='binary')

jv.kill_jvm()
os._exit(1)
