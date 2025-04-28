import os
import sys
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)

import modules.configs_setup as configs
import modules.processed_data_analysis as pda
import modules.javaThings as jv
jv.start_jvm()
jv.init_logger()


# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with image files and configs folder.')
args = parser.parse_args()
raw_data_folder = args.data_path

# load cnfgs
cnfgs = configs.load_configs(raw_data_folder)
print(cnfgs.keys())

colsOI = ['imageID', 'expID', 'group', 'ncells', 'final_size', 'um_per_pixel',  'channels', 'Flag', 'ndiv', 'Length_MA_um', 'Volume_MA_mm3', 'Areas', 'DAPI', 'DAPI_area_norm', 'Sox2', 'Sox2_area_norm', 'Bra', 'Bra_area_norm', 'Foxc1', 'Foxc1_area_norm', 'correct_orientation']
colsOI = ['imageID', 'expID', 'group', 'ncells', 'final_size', 'um_per_pixel',  'channels', 'Flag', 'ndiv', 'Length_MA_um', 'Volume_MA_mm3', 'Areas', 'DAPI', 'DAPI_area_norm', 'Bra', 'Bra_area_norm', 'Cer1', 'Cer1_area_norm', 'Meox1', 'Meox1_area_norm' ]


if ('cut-fuse-groups' in cnfgs.keys()) and ('final_size' in cnfgs.keys()):
    cutfuse = cnfgs['cut-fuse-groups']
    final_size = cnfgs['final_size']
else:
    if 'final_size' not in cnfgs.keys():
        print('Please specifiy the final size per group in the configs file. An argument has been created where you can input those group-sizes (as dictionary) in the file. For each group please add an intiger size value like: 300.')
        #cnfgs['final_size'] = dict.fromkeys(joint_df.group.unique())

    if 'cut-fuse-groups' not in cnfgs.keys():
        print('Please specifiy cut and fused samples in the configs file. An argument has been created where you can input those groups (as list) that are considered cut or fused. Here are all of the groups you can choose from.')
        cnfgs['cut-fuse-groups'] = []

    #configs.update_configs(cnfgs)
print(cnfgs.keys())



# add somne extra identifiers and remove outliers
data_dict = pda.preprocess_dfs(cnfgs, colsOI)

# joint all data frames that should be analyszed together to one 
joint_df = pda.merge_clean_dfs_from_dict(data_dict)

# prepare data for profile plotting, determine the AP axis orientation of the profiles given an orientation channel
oc = 'Bra'
profiles_joint_df = pda.preprocess_df_for_profile_plotting(joint_df, cnfgs, orientation_channel=oc)

# save data frame
if cnfgs.get('merged').get('savename', None):
    df_path = cnfgs['merged']['savename']
else:
    df_path = cnfgs['merged']['save_name']
profiles_joint_df.to_json(df_path, orient='split')
print('Preprocessed df has been saved under ', df_path)


# save analysis detailes to configs
cnfgs['outliers_flagged'] = {'flagged': True, 'label': 'outlier'}
cnfgs['unit_conversion'] = True
cnfgs['profile_orientation'] = {'orientation_bool': True, 'orientation_channel': oc}
configs.update_configs(cnfgs)


jv.kill_jvm()
os._exit(1)
