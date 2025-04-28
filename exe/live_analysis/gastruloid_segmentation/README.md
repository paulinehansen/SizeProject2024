# Analysis Live BrightField
Judith Pineau 2023, Physics of Biological Functions Unit, Institut Pasteur, France 

This project aims at following gastruloid growth when imaged in a 96-well plate over several days.
The code is adapted for images taken with a 4X objective on an Olympus videomicroscope or 10X incusyte. However, it can be adapted for different magnifications (first by changing pixel size, but also certainly by changing the method used to obtain the mask, as the contrast also changes). Given the variability in image contrast between experiments, the main thing to play with is the function used to obtain the mask.

The workflow contains several steps:
[1. If necessary convert to tiff file format](#1-reconstruct-vsi-files-to-tiff) (from the Olympus) into TIFF (imageJ macro ``Convert_vsi_to_TIFF_for96wp.ijm``)\
[2. If more than one z, select optimal z slice](#2-select-optimal-z-slice): If the movies were taken as a z-stack (which is what is usually done because the focus can move), reduce it to just the optimal z (imageJ macro ``TL_Olympus_Select_ZVariance.ijm``).\
[3. If necessary, stitch the movies](#3-stitch-timelapse-movies) using the ``StitchingTimeLapse_Olympus_96wp.ijm`` imageJ macro (see below for instructions)\
[4. Generate the masks](#4-generate-masks) using the python script. We runrun it on a computing cluster (needs GPU).\
[5. Do a selection of the movies](#5-select-movies) where segmentation has globally worked using the imageJ macro ``2023_Validate_Masks_TL.ijm``. \
[6. Run shape analysis](#6-run-shape-analysis) of the selected movies using a python script on the cluster. \
[7. Do the plots](#7-generate-plots) on a local instance of python, in a jupyter notebook.

### Step 1-3 for Olympus movies only

### 1. Reconstruct vsi files to TIFF

Use the imageJ macro ``Convert_vsi_to_TIFF_for96wp.ijm``: run and select your acquisition folder containing the .vsi files (you will have to run it in every timelapse folder). This creates a TIFF folder containing the TIFF files for each position.

### 2. Select optimal z slice

Use the imageJ macro ```TL_Olympus_Select_ZVariance.ijm```: run and select your acquisition folder containing the TIFF files This creates a new folder containing the TIFF of the timelapse, but with only 1 z. The z is selected depending on the maximum variance in the slice, for each time point, to try and keep only the slice with the best contrast.


### 3.Stitch timelapse movies

In the folder where you want to do the stitching, create folders named _1_TLfrom_48h, 2_TLfrom_72h..._, containing the TIFF files of the corresponding timelapse. **It is important that the folders start by the number indicating their order, and finish with 'h'**. Use the imageJ macro 
```StitchingTimeLapse_Olympus_96wp.ijm``` to stitch the movies and save the new TIFF. This will also generate a .csv file that contains the time associated to each image.

### 4. Generate masks

This step in performed on the High Performance Computing cluster. If you do not have access to it yet, do the corresponding Pasteur MOOC (https://moocs.pasteur.fr/dashboard).

**When doing this analysis for the first time**, you must first create a virtual environment containing the necessary packages:

```shell
cd /pasteur/zeus/projets/p01/Pbf1_Zeus/Judith # Go to your personal folder in Zeus
cd Bash # I go to the folder in which I put all my bash odes, and my environments 
module load Python/3.11.0 #Load Python -  I used Python/3.11.0
python3 -m venv 202310_JP_SegmentAnything_testenv # Create your virtual environment for this analysis (I went into a folder that I called Bash in my Zeus folder before doing this, so that I have all of this a bit sorted
source 202310_JP_SegmentAnything_testenv/bin/activate  # Access it


```

Install the packages (not sure there are no extra packages here…)

```python
# Here, you should still be in your virtual environment

pip install numpy==1.24.2
pip install matplotlib==3.7.1
pip install opencv-python==4.7.0.72
pip install pandas==2.0.0
pip install scipy # I had the version 1.11.3
pip install func_timeout
pip install scikit-image==0.20.0
pip install torch==2.0.0
pip install torchvision==0.15.1
pip install git+https://github.com/facebookresearch/segment-anything.git

pip install scikit-learn==1.2.2
pip install networkx==3.1

deactivate # deactivate your virtual environment
cd /pasteur/zeus/projets/p01/Pbf1_Zeus/Judith # I go back to my personal folder in Zeus
```

**To do for each analysis:**

To do analysis on the cluster, your data must be on the Zeus server.
My code is organised so that the data are analysed by condition: create one folder per condition, containing the stitched movies, the .csv files containing the time points, a .txt file with the list of files you want to analyze. For now, I don’t know how to do it automatically, so I just do a .txt file (on Notepad ++, UNIX formatting) with all the files like this for example:\
./G129_BG06_48h_20231002_001_WIB5-1_1stitch.tif_z.tif\
./G129_BG06_48h_20231002_001_WIC5-1_1stitch.tif_z.tif
…..\
In the folder, you must also have the SAM model file ```sam_vit_h_4b8939.pth```.

Open the ```example_mask.bash``` file with a text editor that allows UTF-8 LF
Remarks:
For the input variables, do not put space around the = signs, or anywhere.
Check the paths that have to be changed (the first one that leads to my personnal folder, the ones leading to the FileList and to the files, the one to activate the python environment, the one that leads to the python script to launch)

code to write in terminal to launch on cluster (you can adapt the number of GPU and the memory if needed):\
```shell
sbatch -p gpu -q gpu --gres=gpu:2,gmem:30G './Bash/example_mask.bash'

#####Outputs:
This python script will generate a Mask folder with TIFF stacks containing the masks of each movie, but also .json files for each movie, where each row is a time point and contains important information such as the condition, pixel size, coordinates of the outline of the mask...

### 5. Select movies

After having generated the masks using python, it is good to check whether the segmentation worked. To do so, use the imageJ macro ```2023_Validate_Masks_TL.ijm```.
This macro will generate a stack with the bright field image + the contour of the mask frawn on it, and for each movie ask if you want to keep it or not.
Movies that are kept are saved into a new folder called 'Select', together with the corresponding masks, and a folder called 'Draw' is also created where the overlay is saved.

After this step, you need to copy the .json files of the selected movies into the Select folder.


### 6. Run shape analysis

This step is run on the cluster, using a python script.
You need to create (or reuse) the FileList.txt file with the list of movies to analyse.

The python script will generate a new .json file with some shape characteristics: aspect ratio, major axis length,....

Open the ```example_shape.bash``` file , change the parts that are specific to you or the analysis and save.\
Remarks:

For the input variables, do not put space around the = signs, or anywhere.
Check the paths that have to be changed (the first one that leads to my personnal folder, the ones leading to the FileList and to the files, the one to activate the python environment, the one that leads to the python script to launch)

code to write in terminal to launch on cluster : ```sbatch './Bash/example_shape.bash'``` 


### 7. Generate plots

Put all the shape.json files together in one folder. Then, use the 202311_Plot_Analysis_TLBF_sizeProject_Exp1_clean_fin jupyter notebook example to generate plots

