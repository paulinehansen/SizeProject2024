import os
from sys import path
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/BF_analysis', '')
path.insert(1, modules)

import modules.configs_setup as configs
import modules.raw_live_analysis as raw
import modules.processed_data_analysis as pda
import modules.processed_data_plotter as pdp


# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with image files and configs folder.')
args = parser.parse_args()
raw_data_folder = args.data_path

# load cnfgs
cnfgs = configs.load_configs(raw_data_folder)
save_path = cnfgs['plots_folder']

# read data
df_path = cnfgs['merged']['save_name']

if '_shape.json' in df_path:
    before_shape_analysis_df_path = df_path.replace('_shape.json', '.json')
    if os.path.exists(before_shape_analysis_df_path):
        df = pda.load_processed_df(before_shape_analysis_df_path)
    else:
        print('No "raw" file available in the specified path, but shape analysis has already been completed.')
else:
    df = pda.load_processed_df(df_path)



# prepare data for gastruloid shape analysis
multi_tp_df = pda.prepare_multi_tp_df(df)

# additionally run shape analysis so its run on same code as live movies
save_name = df_path.replace('.json', '_shape.json')
shape_df = raw.compute_multi_tp_mask_analysis(multi_tp_df, save_path=save_name, save=True)

# plot overview plots, to check analysis
for exp in shape_df.exp.unique():
    df_small = shape_df[shape_df['exp']==exp]
    pdp.plot_overview_multi_pt(save_path, df_small,
                               ncol=5, save=True, CNT=True, MA=True, SEG=False, mark=True, name=exp)

print(multi_tp_df.head())
os._exit(1)