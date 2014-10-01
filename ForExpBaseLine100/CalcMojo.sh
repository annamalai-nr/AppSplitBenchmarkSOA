CurFolder=`pwd` #outside ApkDecompOp folder where we have ApkFiles, GT folder etc.
AppDecompFolder=./ApkToolDecompOp #contains rev engg code of all apps
GTFolder=./GT #contains GT extracted for every app
MojoFolder=/home/annamalai/mojo #contains mojo and mojo fm impl

cd $AppDecompFolder
Folders=`ls -l | grep -v ^$| grep ^d| awk '{print $9}'`
echo $Folders


for Folder in $Folders
do
	#echo "Entering Folder $PWD"
	OperatingDir=`readlink -f .`
	cd $Folder
	wc -l smali/Labels.txt|awk '{print $1}'
	cp smali/BaselineSingletonClusters.rsf $MojoFolder/mojo


	ApkName=`echo $Folder| sed -e 's/.Decomp//g'` #stripping .Decomp from the folder name and getting apk name
	ApkGTFolder=$CurFolder/$GTFolder/$ApkName
	cp $ApkGTFolder/GTClusterDetails.rsf $MojoFolder/mojo


	cd $MojoFolder
	echo "Calculating mojo for $Folder"
	java mojo.MoJo mojo/BaselineSingletonClusters.rsf mojo/GTClusterDetails.rsf $1
	rm mojo/*.rsf
	cd $OperatingDir
done
