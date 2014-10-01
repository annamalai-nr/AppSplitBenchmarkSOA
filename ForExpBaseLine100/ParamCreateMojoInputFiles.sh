CurFolder=`pwd` #outside ApkDecompOp folder where we have ApkFiles, GT folder etc.
AppDecompFolder=ApkToolDecompOp #contains rev engg code of all apps
GTFolder=GT #contains GT extracted for every app

cd $AppDecompFolder #going in ApkToolDecompOp folder
Folders=`ls -l | grep -v ^$| grep ^d| awk '{print $9}'`
echo $Folders

for Folder in $Folders
do
	echo "Entering Folder $Folder"
	cd $Folder  #going in Apk's rev engg folder
	python ../../CsvToMojoInputConv.py smali/BaselineSingletonClusters.csv smali/BaselineSingletonClusters.rsf
	#python ../../CsvToMojoInputConv.py smali/WSumClustDetails.csv smali/AppSplitClusterDetails.rsf


	#ApkName=`echo $Folder| sed -e 's/.Decomp//g'` #stripping .Decomp from the folder name and getting apk name
	#ApkGTFolder=$CurFolder/$GTFolder/$ApkName
	#echo $ApkGTFolder
	#python ../../CsvToMojoInputConv.py $ApkGTFolder/GT.csv $ApkGTFolder/GTClusterDetails.rsf	
	cd - #coming back to ApkDecompOp folder
done
