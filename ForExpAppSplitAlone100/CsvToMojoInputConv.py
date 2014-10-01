#13 Sep 2014
#2 37 PM


import os
import sys
from pprint import pprint

#DESC: creates mojo input .rsf file eg format: contain <clustername> <elemname> and one line per elem to cluster mapping
#      see example from http://www.cse.yorku.ca/~bil/downloads/

#input(s) 
#argv1 - name of the ClusterDetails28.csv file (contained format: <clustername>,<elem> <elem2> ...)
#argv2 - name of the output rsf file



def GetClusterDictFromClusterDetailsLines (ClusterDetailsLines):

	ClusterDetailsDict = {}

	for Line in ClusterDetailsLines:
		ClusterName = Line.split(",")[0].strip()
		CulsterElems = Line.split(",")[1].strip().split(" ")

		ClusterDetailsDict[ClusterName] = CulsterElems

	return ClusterDetailsDict


def AdjustKeysTO2Digits(ClusterDetailsDict):
	NewClusterDetailsDict = {}

	for Key in ClusterDetailsDict:
		NewKey = Key
		if len(Key) < 2:
			NewKey = str(Key).zfill(2)

		NewClusterDetailsDict[NewKey] = ClusterDetailsDict[Key]

	return NewClusterDetailsDict


def main():
	ClusterDetailsLines = [Line.strip() for Line in open(str(sys.argv[1])).readlines()]

	ClusterDetailsDict = GetClusterDictFromClusterDetailsLines (ClusterDetailsLines)

	ClusterDetailsDict = AdjustKeysTO2Digits(ClusterDetailsDict) #helps maintaining the oder in the 
	                                                             #ClusterDetails28.csv intact in the ClusterDetails28Dict
	#pprint (ClusterDetailsDict)

	FH = open (str(sys.argv[2]), "w") #.rsf file (output)

	LinesWritten = 0
	for ClustNameAsKey in sorted(ClusterDetailsDict): #has to be read after sorting by key (see comment in line 49)
		for Elem in ClusterDetailsDict[ClustNameAsKey]:
			StrToWrite = "contain " + str(int(ClustNameAsKey)) #using int and then str to remove zfill (see line 37)
			StrToWrite += " " + str(Elem)
			print>>FH, StrToWrite
			LinesWritten = LinesWritten + 1

	if 1 == LinesWritten:
		print>>FH,"contain 2 dummy"
		#print StrToWrite

	FH.close()


if __name__ == '__main__':
	main()
