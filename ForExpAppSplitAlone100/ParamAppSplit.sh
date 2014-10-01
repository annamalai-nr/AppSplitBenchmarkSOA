#./DecompileApksApktool.sh #Creates the app folder inside ApkToolDecompOp
#./ExtraxtClusterFeatues.sh #Files created:
#./CreateNeccessaryMatrices.sh #Files created: Labels.txt, DependencyMatrix.txt, SemanticDistMat.txt, SemanticVectors.txt, TreeDistMatrix.txt, DistMat.txt, SimMat.txt
#./SemanticClusterer.sh #Files created: SemanticDistMat.txt SemanticVectors.txt inside every smali folder
#./ApClusterer.sh $1 $2 $3 $4 #File Created: ClusterDetails28.csv
#./MatrixToGraphViz #call this if visulaization of is required (for eg. GT extraction)

#./CreateClusterFolders.sh #Folder Created: Clusters28


		for SimType in $4
		do
			echo "Calling ApCluster with $1 $2 $3 $SimType"
		
			./ParamApClusterer.sh $1 $2 $3 $SimType #File Created: ClusterDetails28.csv
			./ParamCreateMojoInputFiles.sh #Files created: AppSplitClusterDetails.rsf (inside smali folder)
			FNameForMojoBoxPlot=MojoFMOP\_$1\_$2\_$3\_$SimType.txt
			echo $FNameForMojoBoxPlot
			./CalcMojo.sh -fm | grep -v "apk.Decomp" |grep -v mojo >$FNameForMojoBoxPlot 
			echo "$FNameForMojoBoxPlot" >>FilesToConsiderForBP.txt

			#Delete all files inside ApkToolsDecompOp that are copied temply
			find ./ApkToolDecompOp -maxdepth 1 -type f -delete

			#if req - delete the clusterdetails28.csv
			#may remove later if not req!!!!
			#for FF in `find . -name ClusterDetails28.csv`;do rm $FF;done 
			#for FF in `find . -name AppSplitClusterDetails.rsf`;do rm $FF;done 
		done

