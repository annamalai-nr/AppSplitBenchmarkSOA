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
plt.plot (markersize=200)
plt.legend(['A simple line'])


FileNameAsLines = [Line.strip() for Line in open(sys.argv[1]).readlines()]
SmallThreshold = int(sys.argv[2])
MediumThreshold = int(sys.argv[3])
DataYesCmdIp = str(sys.argv[4])

CombinedBPData = []

for Index, File in enumerate(FileNameAsLines):

    MojoFMNoOfPkgsNPArray = np.genfromtxt(File, dtype=float) 

    
    NoOfLabelsArray = MojoFMNoOfPkgsNPArray[::2]    # Elements from MojoFMNPArray starting from 0 iterating by 2
    AccuracyArray = MojoFMNoOfPkgsNPArray[1::2] # Elements from MojoFMNPArray starting from 1 iterating by 2  

    pprint (NoOfLabelsArray)
    pprint (AccuracyArray)
    
    
    print "0th percentile: ", np.percentile (AccuracyArray, 0)
    print "25th percentile: ", np.percentile (AccuracyArray, 25)
    print "50th percentile: ", np.percentile (AccuracyArray, 50)
    print "75th percentile: ", np.percentile (AccuracyArray, 75)
    print "100th percentile: ", np.percentile (AccuracyArray, 100)
    print "Average: ", np.average (AccuracyArray)
    print "Std dev: ", np.std (AccuracyArray)

    CombinedBPData.append(AccuracyArray)

    
    AppSizeCategory = []
    ColorMap = []
    MarkerShape = []
    

    for AppIndex in range(len(NoOfLabelsArray)):
        if NoOfLabelsArray[AppIndex] <= SmallThreshold:
            #small app
            AppSizeCategory.append('small')
            ColorMap.append('blue')
            MarkerShape.append(">")

        elif NoOfLabelsArray[AppIndex] <= MediumThreshold:
            #medium app
            AppSizeCategory.append('medium')
            ColorMap.append('yellow')
            MarkerShape.append("<")

        else:
            #largeapp
            AppSizeCategory.append('large')
            ColorMap.append('red')
            MarkerShape.append("o")

    pprint (AppSizeCategory)
    print AppSizeCategory.count('small')
    print AppSizeCategory.count('medium')
    print AppSizeCategory.count('large')


    #plt.scatter ([Index] * len(AccuracyArray), AccuracyArray, c=ColorMap, marker="o")   #to print the small dataitems

    if "data" == DataYesCmdIp:
        for DataElemIndex,DataElem in enumerate(AppSizeCategory):
            if 'small' == DataElem:
                plt.scatter(Index,AccuracyArray[DataElemIndex],c=ColorMap[DataElemIndex],marker="o",s=75,alpha=.5)
            elif 'medium' == DataElem:
                plt.scatter(Index,AccuracyArray[DataElemIndex],c=ColorMap[DataElemIndex],marker="^",s=100,alpha=.5)
            else:
                plt.scatter(Index,AccuracyArray[DataElemIndex],c=ColorMap[DataElemIndex],marker="*",s=100,alpha=.5)
    
    
    



#CombinedBPData = [MojoFMNPArray, MojoFMNPArray] #to print more than one plot

#plt.scatter ([0] * len(MojoFMNPArray), MojoFMNPArray, c='yellow') #to print the dataitems with color

#bp_dict = plt.boxplot(MojoFMNPArray, 0, '', positions=[0]) #to print the box and dataitems

bp_dict = plt.boxplot(CombinedBPData, 0, '', whis=np.inf, widths = 0.5, positions=list(xrange(len(CombinedBPData))))


XTicksList = []
for BoxId, Box in enumerate(bp_dict['medians']):
    #BoxId = 0

    MojoFMNPArray = CombinedBPData[BoxId]
    XForText = (bp_dict['medians'][BoxId].get_xydata()[0][0] + bp_dict['medians'][BoxId].get_xydata()[1][0])/2
    YForText = 80

    #StrToPrint = FileNameAsLines[BoxId] + "\n" + "Average: " + str(np.average (MojoFMNPArray).round(2)) + "\n" + "Std dev: " + str(np.std (MojoFMNPArray).round(2))
    FileNameParts = [Part.strip() for Part in FileNameAsLines[BoxId].split("_")]
    FileNameParts = FileNameParts[1:]
    FilNameStr = "\n".join(FileNameParts)
    StrToPrint = FilNameStr + "\n" + "Avg:" + str(np.average (MojoFMNPArray).round(2)) + "\n" + "Std:"+ str(np.std (MojoFMNPArray).round(2))
    #plt.text(XForText,YForText, StrToPrint, horizontalalignment='center')# below
    XTicksList.append(StrToPrint)

plt.xticks(list(xrange(0,len(FileNameAsLines)+1)), XTicksList)


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


