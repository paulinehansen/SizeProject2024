/*
 * It needs to have a folder with the bright field movies, containing a Mask folder in which are saved TIFF files of the masks, named as name+end_mask
 * 
 * It will save the movies and masks you want to keep to a new folder called Select,
 * and also sve the overlay of the mask contour and the bright field image in a folder called Draw
*/

setLineWidth(3);

Dialog.createNonBlocking("Information for the analysis");
Dialog.addDirectory("Choose a Directory", "");
Dialog.addString("End of mask name", "Mask.tiff");
Dialog.show()

myDir = Dialog.getString();
end_mask = Dialog.getString();

myDirM= myDir + "Mask/"; 
newdir= myDir + "Select/"; 
newdirM= newdir + "MaskSelect/";
newdirD= newdir + "Draw/"; 
print(newdir); 
File.makeDirectory(newdir);
File.makeDirectory(newdirM);
File.makeDirectory(newdirD);

list = getFileList(myDir);
listMask =  getFileList(myDirM);
j = 0;

len_end_mask = lengthOf(end_mask);

for (i=0; i<listMask.length; i++) {
    if (!(endsWith(listMask[i], "/")) & (endsWith(listMask[i], end_mask))){
       Maskname=listMask[i];
           
       print(Maskname);

       name = substring(Maskname,0,lengthOf(Maskname)-len_end_mask);
	   open(myDir+name);
	   rename("BF");
	   selectWindow("BF");	
	   open(myDirM+Maskname);
	   rename("Mask");
       getDimensions(width, height, channels, slices, frames);         
            
       //Draw ROIs on each slice      
       selectWindow("Mask");
       run("Analyze Particles...", "size=1000-Infinity clear include add stack");
            
       setForegroundColor(255, 0, 0);
       N_ROI = roiManager("count");
       selectWindow("BF");
       run("Duplicate...", "title=BF_draw duplicate");
       selectWindow("BF_draw");
       Stack.getStatistics(voxelCount, mean, min, max, stdDev);
       setMinAndMax(min, max);
       run("RGB Color");
            
            
       for (l = 0; l < N_ROI; l++) {         	
            selectWindow("BF_draw");
            roiManager("select", l);
            roiManager("draw");        
       }
            
            
		Dialog.createNonBlocking("Keep or Trash?")
		Dialog.addCheckbox("Keep ", true);
		Dialog.show();
		keep = Dialog.getCheckbox();

		
		if (keep == true) {	
			selectWindow("BF");
			saveAs("Tiff",newdir+"/"+name);
			selectWindow(name);
			run("Close");
				
			selectWindow("Mask");
			saveAs("Tiff",newdirM+"/"+Maskname);
			selectWindow(Maskname);
			run("Close");
				
			selectWindow("BF_draw");
			saveAs("Tiff",newdirD+"/"+name+"Draw.tif");
			selectWindow(name+"Draw.tif");
			run("Close");
				
				
		}
		j=j+1;
		roiManager("deselect");
		roiManager("delete");
		run("Close All");
		call("java.lang.System.gc");

	}

    print(listMask[i]+" done");
    call("java.lang.System.gc");
    
}
        
  

