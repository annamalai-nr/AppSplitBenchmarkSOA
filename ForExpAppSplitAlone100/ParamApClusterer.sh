ApkDecompFolder=./ApkToolDecompOp

echo $1 $2 $3 $4 #inputs WILL be used

cd $ApkDecompFolder
ApkFolders=`ls -l|grep ^d|awk '{print $9}'`
for Folder in $ApkFolders
do
	echo "Processing $Folder"
	cd $Folder
	python ../../ApClusterer.py ./smali/Labels.txt ./smali/DependencyMatrix.txt ./smali/SemanticDistMat.txt ./smali/TreeDistMatrix.txt $1 $2 $3 ./smali/ClusterDetails28.csv ./smali/WSumClustDetails.csv $4
	cd -	
done
