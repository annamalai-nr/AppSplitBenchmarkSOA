#4 pm 
#24 Sep 2014


#from matplotlib import pyplot as plt
import sys
import numpy as np
from pprint import pprint
#import simplejson
#import json
from matplotlib import pyplot as plt



plt.figure(figsize=(10,10))
plt.ylim(0,100)
plt.xlim(None)

FileNameAsLines = [Line.strip() for Line in open(sys.argv[1]).readlines()]

CombinedBPData = []

for Index, File in enumerate(FileNameAsLines):

    MojoFMNPArray = np.genfromtxt(File, dtype=float) 

    try:
	    	print "0th percentile: ", np.percentile (MojoFMNPArray, 0)
	    	print "25th percentile: ", np.percentile (MojoFMNPArray, 25)
	    	print "50th percentile: ", np.percentile (MojoFMNPArray, 50)
	    	print "75th percentile: ", np.percentile (MojoFMNPArray, 75)
	    	print "100th percentile: ", np.percentile (MojoFMNPArray, 100)
    except:
			print "err"


    print "Average: ", np.average(MojoFMNPArray)
    print "Std dev: ", np.std (MojoFMNPArray)

    CombinedBPData.append(MojoFMNPArray)

    plt.scatter ([Index] * len(MojoFMNPArray), MojoFMNPArray, c='yellow') #to print the dataitems


#CombinedBPData = [MojoFMNPArray, MojoFMNPArray] #to print more than one plot

#plt.scatter ([0] * len(MojoFMNPArray), MojoFMNPArray, c='yellow') #to print the dataitems with color

#bp_dict = plt.boxplot(MojoFMNPArray, 0, '', positions=[0]) #to print the box and dataitems

bp_dict = plt.boxplot(CombinedBPData, 0, '', whis=np.inf, widths = 0.5, positions=list(xrange(len(CombinedBPData))))



#plt.show()
#exit (0)

for BoxId, Box in enumerate(bp_dict['medians']):
    #BoxId = 0

    MojoFMNPArray = CombinedBPData[BoxId]
    XForText = (bp_dict['medians'][BoxId].get_xydata()[0][0] + bp_dict['medians'][BoxId].get_xydata()[1][0])/2
    YForText = 80

    #StrToPrint = FileNameAsLines[BoxId] + "\n" + "Average: " + str(np.average (MojoFMNPArray).round(2)) + "\n" + "Std dev: " + str(np.std (MojoFMNPArray).round(2))
    FileNameParts = [Part.strip() for Part in FileNameAsLines[BoxId].split("_")]
    FileNameParts = FileNameParts[1:]
    FilNameStr = "\n".join(FileNameParts)
    StrToPrint = FilNameStr + "\n" + str(np.average (MojoFMNPArray).round(2)) + "\n" + str(np.std (MojoFMNPArray).round(2))
    plt.text(XForText,YForText, StrToPrint,
             horizontalalignment='center')# below


for line in bp_dict['medians']:
    	x, y = line.get_xydata()[1]
    	plt.text(x, y, '%.2f' % y,
             horizontalalignment='center')

for line in bp_dict['boxes']:
        x, y = line.get_xydata()[1] # bottom of left line
        plt.text(x,y, '%.2f' % y,
             horizontalalignment='center', # centered
             verticalalignment='top')      # below
        x, y = line.get_xydata()[2] # bottom of right line
        plt.text(x,y, '%.2f' % y,
             horizontalalignment='center', # centered
             verticalalignment='bottom')# below	

#does not loop through all caps

for line in bp_dict['caps']:
        x1, y1 = line.get_xydata()[0]
        x2, y2 = line.get_xydata()[1]

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        plt.text(x,y, '%.2f' % y,
             horizontalalignment='center', # centered
             verticalalignment='top') #below
        

plt.savefig('foo.pdf',figsize=(40,40),dpi=600)
plt.show()


