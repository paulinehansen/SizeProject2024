/*20221213 Judith Pineau
	Analysis of timelapse in BF at the Olympus videomicroscope
	Selecting the best z position according to the value of variance
*/

myDir = getDirectory("Choose a Directory ");

list = getFileList(myDir);

Projdir= myDir + "/Z_select/"; 
print(Projdir); 
File.makeDirectory(Projdir);

setForegroundColor(255, 255, 255);
setBackgroundColor(0, 0, 0);

setBatchMode(true);

for (k=0; k<list.length; k++) {
     print(list[k]);
     if (!(endsWith(list[k], "/")) & (endsWith(list[k], ".tif"))){
		name=list[k];
		path=substring(name,0,lengthOf(name)-4);
		print(path);
		   
		open(myDir+list[k]);
		getDimensions(width, height, channels, slices, frames);
		
		imageTitle=getTitle();
		selectWindow(name);
		rename("Image");
		getDimensions(width, height, channels, slices, frames);			
		nbTime=frames;
		
		
		//Try to find best focus slice for each time point

		selectWindow("Image");
		run("Duplicate...", "title=Image_SD duplicate");
		run("Variance...", "radius=2 stack");

		getDimensions(width, height, channels, slices, frames);	
		newImage("Black", "16-bit black", width, height, 1);


		for (i = 1; i <= frames; i++) {
	
			SD = newArray(slices);
			selectWindow("Image_SD");
			Stack.setFrame(i);
			for (j = 1; j <= slices; j++) {
				selectWindow("Image_SD");
				Stack.setSlice(j);
				getRawStatistics(nPixels, mean, min, max, std, histogram);
				SD[j] = std;
			}
			SD_rank = Array.rankPositions(SD);
			n_max = SD_rank[slices];
			print(n_max);
			selectWindow("Image");
			run("Duplicate...", "title=I duplicate slices="+n_max+"-"+n_max+" frames="+i+"-"+i+"");
			run("Concatenate...", "  title=Image_z_s open image1=Black image2=I image3=[-- None --]");

			selectWindow("Image_z_s");
			print("Frame"+i+" z slice "+n_max+"");
			rename("Black");
		}

		selectWindow("Black");
		Nf = frames+1;
		run("Duplicate...", "title=Stack duplicate range=2-"+Nf+"");
		close("Black");
		selectWindow("Stack");
		saveAs("Tiff",Projdir+name+"_z.tif");
			
		run("Close All");
		call("java.lang.System.gc");
		print(list[k]+" done");
   }
}

setBatchMode(false);


