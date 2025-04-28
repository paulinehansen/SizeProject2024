/* 20230525 Judith Pineau
 *  
 *  Stitching timelapse movies from 96 well plates on Olympus
 *  First: Run the Convert_vsi_to_TIFF2.ijm macro,
 *  and make a folder with inside the TIFF folders:
 *  
 *  Folder names: 1_TLfrom_48h ; 2_TLfrom_72h ; 3_TLfrom_96h..... What's important is the number at the beginning and the h at the end
 */


// Create dialog ; Fill in with info about the folders to stitch
//Dialog.create("Title");
Dialog.create("Information about the folders to stitch");
Dialog.addDirectory("Directory with all the folders containing the TIFF files", "");
Dialog.addNumber("Time step (h)", 1);
Dialog.addNumber("Number of acquisition folders", 3);
Dialog.show()

mainDir = Dialog.getString();
dt = Dialog.getNumber();
n_folders = Dialog.getNumber();

Dialog.createNonBlocking("Give Starting times of each acquisition (in hours)");
for (i = 0; i < n_folders; i++) {
	ID = "Folder"+i;
	Dialog.addNumber(ID, 3);
}
Dialog.show()
Time_folders=newArray(n_folders);
for (i = 0; i < n_folders; i++) {
	Time_folders[i]=Dialog.getNumber();
}



print(mainDir)
mainList = getFileList(mainDir);
List_direct=newArray(n_folders);
n=0;
for (k=0; k<mainList.length; k++) {
     	//print(mainList[k]);
     if ((endsWith(mainList[k], "h/")) & !(endsWith(mainList[k], ".tif"))){
        print(mainList[k]);
        List_direct[n] = mainDir+mainList[k];
        n=n+1;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//First, do it for the first folder to initialize
N_sub = newArray(n_folders);
WellID = newArray(0);

subList =getFileList(List_direct[0]);
N_sub[0] = subList.length;
//Do a list with the Well ID corresponding to each file	for this folder
Well_ID1=newArray(subList.length);
	
for (k=0; k<subList.length; k++) {
    print(subList[k]);
    if (!(endsWith(subList[k], "/")) & (endsWith(subList[k], "-1_1.tif_z.tif"))){// changed
		name=subList[k];
		path=substring(name,0,lengthOf(name)-4);
		
			
		Well_ID_1 = substring(path,0,lengthOf(path)-10); // changed
		Well_ID_2 = split(Well_ID_1, "_WI");
		Well_ID_3 = Well_ID_2[lengthOf(Well_ID_2)-1];
		Well_ID1[k] = Well_ID_3;
		print("Well ID is "+Well_ID1[k]);	
        }
}



//Start by opening the first timelapse
N_sub = newArray(n_folders); // to put the number of time points per imaging session

for (k=0; k<subList.length; k++) {
     print(subList[k]);
     if (!(endsWith(subList[k], "/")) & (endsWith(subList[k], "-1_1.tif_z.tif"))){// changed
		name1=subList[k];
		path=substring(name1,0,lengthOf(name1)-4);
		
		open(List_direct[0]+subList[k]);
		rename(name1);
		Well_ID = Well_ID1[k];
		print("Trying to stitch Well"+Well_ID);
		selectWindow(name1);
		getDimensions(width, height, channels, slices, frames);
		N_sub[0] = frames;
		saveAs("Tiff",mainDir+path+"stitch.tif");
		rename(name1);
		
		// Then look for the follow up movies in the next folders
		for (n=1; n<n_folders; n++) {
				
			// Get list of files in this folder
			subList_n =getFileList(List_direct[n]);

			for (k_n = 0; k_n < subList_n.length; k_n++) {
				tag_well = Well_ID + "-1_1.tif_z.tif";   // changed
				if (!(endsWith(subList_n[k_n], "/")) & (endsWith(subList_n[k_n], tag_well))){
					name2=subList_n[k_n];
					open(List_direct[n]+name2);
					rename(name2);
					getDimensions(width, height, channels, slices, frames);
					N_sub[n] = frames;
					run("Concatenate...", "title="+name1+" open image1="+name1+" image2="+name2);
					saveAs("Tiff",mainDir+path+"stitch.tif");
					rename(name1);
				}
			}

					
		}
		
		// Make a file with the time points
		
		// Get the number of time points
		getDimensions(width, height, channels, slices, frames);
		Times = newArray(0);
		for (n = 0; n < n_folders; n++) {
			T_n = Array.getSequence(N_sub[n]);
			for (y = 0; y < N_sub[n]; y++ ){
    			T_n[y] = dt*T_n[y] + Time_folders[n];
			 }
			 Times = Array.concat(Times,T_n);
		}
		for(z=0;z<(frames-1);z++){
			setResult("Time", z, Times[z]);
		}
		updateResults();
		selectWindow("Results");
		saveAs("Results", mainDir+path+"_time.csv");
		close("Results");   	
			
				
		run("Close All");
		call("java.lang.System.gc");   

			
			
     }
}

