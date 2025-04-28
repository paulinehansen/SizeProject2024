/*
 * Judith Pineau/ 
 * Adapted from macro by Olivier Burri @ EPFL - SV - PTECH - BIOP
 * Takes a folder as an input, and converts all the .vsi files in it to TIFF, and saves them in a new folder called 'TIFF' 
 * 
 */

extension = "vsi";

dir = getDirectory("Select a directory containing one or several .vsi files");

files = getFileList(dir);
File.makeDirectory(dir+File.separator+"TIFF"+File.separator);

setBatchMode(true);
k=0;
n=0;

run("Bio-Formats Macro Extensions");
for(f=0; f<files.length; f++) {
    if(endsWith(files[f], "."+extension)) {
        k++;
        id = dir+files[f];
        Ext.setId(id);
        Ext.getSeriesCount(seriesCount);
        print(seriesCount+" series in "+id);
        n+=seriesCount;

        for (i=0; i<1; i++) {
            run("Viewer", "open=["+id+"]");
            print(id);
            fullName    = getTitle();
            dirName     = substring(fullName, 0,lastIndexOf(fullName, "."+extension));
            fileName    = substring(fullName, lastIndexOf(fullName, " - ")+3, lengthOf(fullName));
            print("fullName: "+fullName);
            print("dirName: "+dirName);
            print("fileName: "+fileName);
            rename("Image");
            selectWindow("Image");
            if(endsWith(dirName, "-1")){
                print("Saving "+fileName+" under "+dir+File.separator+dirName);
                getDimensions(x,y,c,z,t);
                saveAs("tiff", dir+File.separator+"TIFF"+File.separator+dirName+"_"+(i+1)+".tif");
            } else {
                print("Not Saving");  
            }
            run("Close All");
            call("java.lang.System.gc");
        }
    }
}
Ext.close();
setBatchMode(false);
showMessage("Converted "+k+" files and "+n+" series!");
call("java.lang.System.gc");
