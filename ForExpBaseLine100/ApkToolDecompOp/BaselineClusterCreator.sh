for f in `find . -name Labels.txt`
do
	pardir=`dirname $f`
	echo "Creating baseline clusters for $f"
	touch $pardir/BaselineSingletonClusters.csv
	ln=1
	for line in `cat $f`
	do 
		echo $ln,$line >>$pardir/BaselineSingletonClusters.csv
		ln=`expr $ln + 1`
	done
done

