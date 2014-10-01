#12:00 noon
#19 sep 2014

#desc: Performs AP clustering using struct, semantic and tax sim

#arg1 - Labels.txt file
#arg2 - prog dep (sim) matrix file
#arg3 - semantic distance matrix file
#arg4 - tree distance matrix file
#arg5 - weight for struc sim (range [0,1])
#arg6 - weight for semantic sim (range [0,1])
#arg7 - weight for tax sim (range [0,1])
#arg8 - sum cobination o/p clusters file
#arg9 - prod cobination o/p clusters file


import numpy
import sys
from pprint import pprint
import sklearn
from sklearn.cluster import AffinityPropagation
import pylab as plt
import os
from matplotlib import cm
import math


def IdentifySelfDepPkgFromDepMat(DepMat):
    LocalDepMat = DepMat.copy()
    numpy.fill_diagonal(LocalDepMat, 0)
            
    LinesToDel = []  
    for RowIndex, Row in enumerate (LocalDepMat):
        if 0 == Row.sum():
            ColSum = LocalDepMat[:,RowIndex].sum()
            if 0 == ColSum:
                LinesToDel.append(RowIndex)
        
    return LinesToDel      
    
def PruneMat(Mat, SelfDepPkgIndices):
    SelfDepPkgIndices.sort()
    AdjustedIndex = []
    for IndexOfIndex, IndexOfPkgToBeDeleted in enumerate(SelfDepPkgIndices):
        AdjustedIndex.append(IndexOfPkgToBeDeleted - IndexOfIndex)
    
    LocalMat = Mat.copy()
    for Index in AdjustedIndex:
        LocalMat = numpy.delete(LocalMat, Index, 0)
        LocalMat = numpy.delete(LocalMat, Index, 1)
        
    return LocalMat
    
def PruneLabels (Labels, SelfDepPkgIndices):
    PrunedLabels = []
    for Index, Pkg in enumerate(Labels):
        if Index not in SelfDepPkgIndices:
            PrunedLabels.append(Pkg)
    return PrunedLabels
      


def GetTreeDepth(Label):
    return len(Label.split("/")) if "/" in Label else 1
        
  
    
def GetLCADepth (RowLabel, ColLabel):
    RowLabelParts = [Component.strip() for Component in RowLabel.split("/")]
    ColLabelParts = [Component.strip() for Component in ColLabel.split("/")]
    
    MinLabels = min(len(RowLabelParts), len(ColLabelParts))
    LCADepth = 0
    for Index in xrange(0, MinLabels):
        if RowLabelParts[Index] == ColLabelParts[Index]:
            LCADepth += 1
            continue
        else:
            break
    return LCADepth


def GetSemanticSimValue (Type, RowDepth, ColDepth, LCADepth, Hops, MaxOntoDepth):
    if "spl" == Type:
        SPL = (1+1)/float(Hops+1)
        return SPL

    elif "swp" == Type:   
        DistN1 = RowDepth - LCADepth
        DistN2 = ColDepth - LCADepth
        SWP = float(2*LCADepth)/(DistN1 + DistN2 + 2*LCADepth) #will be in range [0-1]
        return SWP
    elif "sli" == Type:
        Alpha = 0.2 #suggested in paper
        Beta = 0.6 #suggested in paper
        Nr = math.exp(Beta*LCADepth) - math.exp(-(Beta*LCADepth))
        Dr = math.exp(Beta*LCADepth) + math.exp(-(Beta*LCADepth))
        SLI = math.exp (-(Alpha*Hops))*(Nr/Dr)
        return SLI
    elif "slc" == Type:
        Nr = float(Hops+1)
        Dr = 2*MaxOntoDepth+1
        SLC = Nr/Dr
        SLC = -(math.log(SLC))
        return SLC
    elif "smy" == Type:
        Nr = (LCADepth*LCADepth*pow(RowDepth,0)*pow(ColDepth,0)) /(pow((RowDepth-LCADepth),1) * pow((ColDepth-LCADepth),1))
        #SMy = float(Nr)/float(pow(Hops, 2))
        SMY = Nr
        return SMY
    else:
        print "please enter valid type of similarity: swp or sli or slc or smy"
        exit (-1)

def NormalizeInRange01(Mat):
    Range = (Mat.max() - Mat.min())+0.0001#smoothing
    NormedMat = Mat - Mat.min()
    NormedMat = NormedMat/Range
    return NormedMat

def GetOntologySimMat (Type, PrunedTreeDistMat, PrunedLabels, PrunedSemanticSimMat, NormalizedPrunedDepMat, MaxTreeDepth):
    OntologySimilarityMat = numpy.zeros(shape = (len(PrunedLabels), len(PrunedLabels)))
     

    for RowIndex in xrange(0, len(PrunedTreeDistMat)):
        for ColIndex in xrange(0, len(PrunedTreeDistMat)):
            if RowIndex == ColIndex:
                #have to be Max similarity!
                continue
                
            Hops = (PrunedTreeDistMat[RowIndex][ColIndex]) #dist b/w row elem and col elem
            RowLabel = PrunedLabels[RowIndex]
            ColLabel = PrunedLabels[ColIndex]
            RowDepth = GetTreeDepth(RowLabel)
            ColDepth = GetTreeDepth(ColLabel)
            LCADepth = GetLCADepth(RowLabel, ColLabel)

            #similar to cannot link constraint (if the LCA is root, then don't link these to nodes)
            if 0 == LCADepth and "smy" == Type: 
                OntologySimilarityMat[RowIndex][ColIndex] = 0
                PrunedSemanticSimMat[RowIndex][ColIndex] = 0
                NormalizedPrunedDepMat[RowIndex][ColIndex] = 0
                continue
            
            OntologySimilarityMat[RowIndex][ColIndex] = GetSemanticSimValue (Type, 
                RowDepth, ColDepth, LCADepth, Hops, MaxTreeDepth)
      
    #Normalizing to range [0,1]
    OntologySimilarityMat = NormalizeInRange01(OntologySimilarityMat)

    #*** After normalizing ***
    #Setting maximum ontological similarity to the diag elements
    numpy.fill_diagonal(OntologySimilarityMat, 1)
       
    return OntologySimilarityMat, PrunedSemanticSimMat, NormalizedPrunedDepMat
        

def GetMaxLabelDepth(Labels):
    MaxDepth = 1
    for Label in Labels:
        Depth = GetTreeDepth(Label)
        if MaxDepth < Depth:
            MaxDepth = Depth
    return MaxDepth


def FormPredictedLabels (ClusterDict, Labels):
    PredictedLabels = [None] * len(Labels)
    for ClusterIdAsKey in ClusterDict:
        Elems = ClusterDict[ClusterIdAsKey]
        for Elem in Elems:
            IndexOfEleminLabels = Labels.index(Elem)
            PredictedLabels[IndexOfEleminLabels] = ClusterIdAsKey
    return PredictedLabels

def PrintClusterResultsAndExit(Labels, ProdClustResultSaveFName, SumClustResultSaveFName):
    ClusterDict = {}
    for Index, Label in enumerate(Labels):
        ClusterDict[Index] = []
        ClusterDict[Index].append(Label)
        
    PredictedLabels = FormPredictedLabels(ClusterDict, Labels)
    SaveClustResultToFile (ClusterDict, ProdClustResultSaveFName)
    SaveClustResultToFile (ClusterDict, SumClustResultSaveFName)
    #GT = [Line.split( )[1].strip() for Line in open(sys.argv[5], 'r').readlines()]
    #print 'Adj Rand Index = ', sklearn.metrics.adjusted_rand_score(GT, PredictedLabels)
    exit(0)



def PerformApCluster(CombinedMat, PrunedLabels):
    Preference = numpy.diag(CombinedMat)/len(CombinedMat)  
    #so, combined mat controls the preferences. Hence when we use only one sim,
    #this step should not be a problem
    Preference = Preference * Preference #this scaling might not be needed always?


    #for making self dependent packages as singleton clusters 
    #if this is not done, AP will put this pkg in some other cluster 
    ColSums = numpy.sum(CombinedMat, axis=0)
    RowSums = numpy.sum(CombinedMat,axis=1)
    for Index in xrange(0, len(CombinedMat)):
        if 0 == RowSums[Index]  and 0 == ColSums[Index]: # this should not happen, coz pruned mat has no self dep pkgs - check with debuging
            CombinedMat[Index][Index] = 1
            Preference[Index] = 1

    af = AffinityPropagation(affinity = 'precomputed', verbose=True, damping=.5, preference=Preference).fit(CombinedMat)


    ClusterCenters = af.cluster_centers_indices_
    ClusteredLabels = af.labels_
    print('Estimated number of clusters: %d' % len(ClusterCenters))
    print ClusteredLabels


    '''
    pprint (PrunedLabels)
    numpy.set_printoptions(threshold='nan')
    numpy.set_printoptions(precision=2)
    CombinedMatForPrint = map(list, CombinedMat)
    pprint (CombinedMatForPrint)
    '''

    ClusterDict = {}
    for Index, Elem in enumerate(ClusteredLabels):
        if Elem not in ClusterDict.keys():
            ClusterDict[Elem] = []
            ClusterDict[Elem].append(PrunedLabels[Index])
        else:
            ClusterDict[Elem].append(PrunedLabels[Index])

    return ClusterCenters, ClusterDict


def MergeSelfDepPkgsAsSingletonCLusters (ClusterDict, SelfDepPkgIndices, Labels):

    NextKey = len(ClusterDict)
    for Index in SelfDepPkgIndices:
        ClusterDict[NextKey] = []
        ClusterDict[NextKey].append(Labels[Index])
        NextKey += 1
    print "clusters"
   
    return ClusterDict

def NormalizeDepMat (DepMat):
    RowWiseSum = DepMat.sum(axis=1)[:,None]
    RowWiseSum [RowWiseSum == 0] = 1 #if rowwise sum is zero to avoid divide by zero error
    NormalizedDepMat = DepMat/RowWiseSum
    return NormalizedDepMat


def VisualizeColorMaps(CombinedMat, NormalizedPrunedDepMat, 
    PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels):

    '''   
    #plt.grid(True)
    #plt.subplots_adjust(bottom=0.50)
    plt.pcolor(NormalizedPrunedDepMat)
    plt.colorbar(use_gridspec=True) #to resize to the tight layout format
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat)+0.5),PrunedLabels, rotation=30,ha='right') 
    #in prev line: ha = horizontal alignment - right is used to make label terminate the the center of the grid
    plt.title("NormalizedPrunedDepMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout() #to resize so that all labels are visible
    #plt.savefig('foo.pdf',figsize=(4,4),dpi=600) # to save image as pdf, fig size may or maynot be used
    plt.show()
    

    plt.pcolor(PrunedSemanticSimMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels, rotation=45,ha='right')
    plt.title("PrunedSemanticSimMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout()
    plt.show()
    '''
    
    plt.pcolor(OntologySimilarityMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels, rotation=45,ha='right')
    plt.title("OntologySimilarityMat",fontsize=20,verticalalignment='bottom')
    plt.xlabel('Packages')
    plt.ylabel('Packages')
    plt.tight_layout()
    plt.show()

'''
    plt.pcolor(CombinedMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat)), PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat)),PrunedLabels, rotation=45,ha='right')
    plt.title("CombinedMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout()
    plt.savefig('foo.pdf',dpi=600)   
    plt.show()
'''
    
def InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels):
    ClusterCenters,ClusterDict = PerformApCluster(CombinedMat, PrunedLabels)

    ClusterDict = MergeSelfDepPkgsAsSingletonCLusters (ClusterDict, 
        SelfDepPkgIndices, Labels)

    pprint (ClusterDict)

    PredictedLabels = FormPredictedLabels(ClusterDict, Labels)
    print "Predcited Labels ", PredictedLabels
        

    for Index, Center in enumerate(ClusterCenters):
        print "Center Index, Center Label: ",Center,  PrunedLabels[Center]

    return ClusterDict


def SaveClustResultToFile (ClusterDict, FileName):

    FH = open(FileName, 'w')
    for ClusterIdAsKey in ClusterDict:
        LineToPrint = str(int(ClusterIdAsKey)+1) + ","
        for Elem in ClusterDict[ClusterIdAsKey]:
            LineToPrint = LineToPrint + Elem + " "
        LineToPrint = LineToPrint[:-1]
        print>>FH, LineToPrint
    FH.close()


    #GT = [Line.split( )[1].strip() for Line in open(sys.argv[5], 'r').readlines()]
    #print 'Adj Rand Index = ', sklearn.metrics.adjusted_rand_score(GT, PredictedLabels)


def GetCombinedProMat(PDGMat,SemSimMat, TaxSimMat, WStruct, WSem, WTax):
    if (0 == WStruct and 0 == WSem and 0 == WTax):
        return numpy.zeros (PDGMat.shape)

    CombinedMat = numpy.ones (PDGMat.shape)
    if (0 != WStruct):
        CombinedMat = CombinedMat * PDGMat
    if (0 != WSem):
        CombinedMat = CombinedMat * SemSimMat
    if (0 != WTax):
        CombinedMat = CombinedMat * TaxSimMat
        
    return CombinedMat    
    

'''
main
'''
def main():

    Labels = [Label.strip() for Label in open (str (sys.argv [1]), 'r').readlines()]

    StructDependencyMat = numpy.genfromtxt(sys.argv[2], dtype=float) #similarity mat
    
    TreeDistInfoMat = numpy.genfromtxt(sys.argv[4], dtype=float) #Distance mat

    WStruct = float(sys.argv[5]) #weight for struc sim (range [0,1])
    WSem = float(sys.argv[6]) #weight for struc sim (range [0,1])
    WTax = float(sys.argv[7]) #weight for struc sim (range [0,1])

    ProdClustResultSaveFName = str(sys.argv[8])

    SumClustResultSaveFName = str(sys.argv[9])

    SimType = str(sys.argv[10])

    if 1 >= len(Labels):
        #no elements to cluster
        PrintClusterResultsAndExit(Labels, ProdClustResultSaveFName, SumClustResultSaveFName)  

    #Identify self dep elements in the mat
    SelfDepPkgIndices = IdentifySelfDepPkgFromDepMat(StructDependencyMat)

    #pruning based on self dependency
    PrunedDepMat = PruneMat(StructDependencyMat, SelfDepPkgIndices)
    PrunedTreeDistMat = PruneMat (TreeDistInfoMat, SelfDepPkgIndices)
    PrunedLabels = PruneLabels (Labels, SelfDepPkgIndices)

    #semantic dist mat may or may not be present ;)
    if (os.path.getsize(sys.argv[3]) == 0): #sematic dist mat has content
        PrunedSemanticSimMat = numpy.identity(len(PrunedLabels))
    else:
        SemDistMat = numpy.genfromtxt(sys.argv[3], dtype=float)
        PrunedSemanticDistMat = PruneMat (SemDistMat, SelfDepPkgIndices)
        #converting dist to sim matrix
        PrunedSemanticSimMat = 1 - PrunedSemanticDistMat 

    #after pruning, all them turn out to be self dependent!
    if 0 == len(PrunedLabels):
        #no elements to cluster
        PrintClusterResultsAndExit(Labels, ProdClustResultSaveFName, SumClustResultSaveFName)

    #This is a point where clustering must be performed!
    

    print "Weights WStruct, WSem, WTax: ", WStruct, WSem, WTax


    NormalizedPrunedDepMat = NormalizeDepMat (PrunedDepMat) 

    MaxTreeDepth = GetMaxLabelDepth(Labels)
    
    OntologySimilarityMat, PrunedSemanticSimMat, NormalizedPrunedDepMat  = \
        GetOntologySimMat (SimType, PrunedTreeDistMat, PrunedLabels, PrunedSemanticSimMat,
            NormalizedPrunedDepMat, MaxTreeDepth)

    #scaling sim values by weight
    ScaledNormalizedPrunedDepMat = WStruct * NormalizedPrunedDepMat
    ScaledPrunedSemanticSimMat = WSem * PrunedSemanticSimMat
    ScaledOntologySimilarityMat = WTax * OntologySimilarityMat
    
    
    #combining using linear (sum) combination
    WeigtedSumMat = (ScaledPrunedSemanticSimMat + \
        ScaledNormalizedPrunedDepMat + ScaledOntologySimilarityMat)/3

    #combining using non linear (prod) combination
    WeigthedProdMat = GetCombinedProMat(ScaledNormalizedPrunedDepMat, 
        ScaledPrunedSemanticSimMat,ScaledOntologySimilarityMat, WStruct, WSem,
        WTax)


    CombinedMat = WeigthedProdMat #currently considering only product
    #CombinedMat = CombinedMat/CombinedMat.max() #is this required?
    CombinedMat = NormalizeInRange01(CombinedMat)

    ClusterDict = InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels)

    #VisualizeColorMaps (CombinedMat, NormalizedPrunedDepMat, PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels)

    SaveClustResultToFile (ClusterDict, ProdClustResultSaveFName)

    CombinedMat = WeigtedSumMat #currently considering only product
    #CombinedMat = CombinedMat/CombinedMat.max() #is this required?
    CombinedMat = NormalizeInRange01(CombinedMat)

    ClusterDict = InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels)

    #VisualizeColorMaps (CombinedMat, NormalizedPrunedDepMat, PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels)

    SaveClustResultToFile (ClusterDict, SumClustResultSaveFName)


if __name__ == '__main__':
    main()
#12:00 noon
#19 sep 2014

#desc: Performs AP clustering using struct, semantic and tax sim

#arg1 - Labels.txt file
#arg2 - prog dep (sim) matrix file
#arg3 - semantic distance matrix file
#arg4 - tree distance matrix file
#arg5 - weight for struc sim (range [0,1])
#arg6 - weight for semantic sim (range [0,1])
#arg7 - weight for tax sim (range [0,1])
#arg8 - sum cobination o/p clusters file
#arg9 - prod cobination o/p clusters file


import numpy
import sys
from pprint import pprint
import sklearn
from sklearn.cluster import AffinityPropagation
import pylab as plt
import os
from matplotlib import cm
import math


def IdentifySelfDepPkgFromDepMat(DepMat):
    LocalDepMat = DepMat.copy()
    numpy.fill_diagonal(LocalDepMat, 0)
            
    LinesToDel = []  
    for RowIndex, Row in enumerate (LocalDepMat):
        if 0 == Row.sum():
            ColSum = LocalDepMat[:,RowIndex].sum()
            if 0 == ColSum:
                LinesToDel.append(RowIndex)
        
    return LinesToDel      
    
def PruneMat(Mat, SelfDepPkgIndices):
    SelfDepPkgIndices.sort()
    AdjustedIndex = []
    for IndexOfIndex, IndexOfPkgToBeDeleted in enumerate(SelfDepPkgIndices):
        AdjustedIndex.append(IndexOfPkgToBeDeleted - IndexOfIndex)
    
    LocalMat = Mat.copy()
    for Index in AdjustedIndex:
        LocalMat = numpy.delete(LocalMat, Index, 0)
        LocalMat = numpy.delete(LocalMat, Index, 1)
        
    return LocalMat
    
def PruneLabels (Labels, SelfDepPkgIndices):
    PrunedLabels = []
    for Index, Pkg in enumerate(Labels):
        if Index not in SelfDepPkgIndices:
            PrunedLabels.append(Pkg)
    return PrunedLabels
      


def GetTreeDepth(Label):
    return len(Label.split("/")) if "/" in Label else 1
        
  
    
def GetLCADepth (RowLabel, ColLabel):
    RowLabelParts = [Component.strip() for Component in RowLabel.split("/")]
    ColLabelParts = [Component.strip() for Component in ColLabel.split("/")]
    
    MinLabels = min(len(RowLabelParts), len(ColLabelParts))
    LCADepth = 0
    for Index in xrange(0, MinLabels):
        if RowLabelParts[Index] == ColLabelParts[Index]:
            LCADepth += 1
            continue
        else:
            break
    return LCADepth


def GetSemanticSimValue (Type, RowDepth, ColDepth, LCADepth, Hops, MaxOntoDepth):
    if "spl" == Type:
        SPL = (1+1)/float(Hops+1)
        return SPL

    elif "swp" == Type:   
        DistN1 = RowDepth - LCADepth
        DistN2 = ColDepth - LCADepth
        SWP = float(2*LCADepth)/(DistN1 + DistN2 + 2*LCADepth) #will be in range [0-1]
        return SWP
    elif "sli" == Type:
        Alpha = 0.2 #suggested in paper
        Beta = 0.6 #suggested in paper
        Nr = math.exp(Beta*LCADepth) - math.exp(-(Beta*LCADepth))
        Dr = math.exp(Beta*LCADepth) + math.exp(-(Beta*LCADepth))
        SLI = math.exp (-(Alpha*Hops))*(Nr/Dr)
        return SLI
    elif "slc" == Type:
        Nr = float(Hops+1)
        Dr = 2*MaxOntoDepth+1
        SLC = Nr/Dr
        SLC = -(math.log(SLC))
        return SLC
    elif "smy" == Type:
        Nr = (LCADepth*LCADepth*pow(RowDepth,0)*pow(ColDepth,0)) /(pow((RowDepth-LCADepth),1) * pow((ColDepth-LCADepth),1))
        #SMy = float(Nr)/float(pow(Hops, 2))
        SMY = Nr
        return SMY
    else:
        print "please enter valid type of similarity: swp or sli or slc or smy"
        exit (-1)

def NormalizeInRange01(Mat):
    Range = (Mat.max() - Mat.min())+0.0001#smoothing
    NormedMat = Mat - Mat.min()
    NormedMat = NormedMat/Range
    return NormedMat

def GetOntologySimMat (Type, PrunedTreeDistMat, PrunedLabels, PrunedSemanticSimMat, NormalizedPrunedDepMat, MaxTreeDepth):
    OntologySimilarityMat = numpy.zeros(shape = (len(PrunedLabels), len(PrunedLabels)))
     

    for RowIndex in xrange(0, len(PrunedTreeDistMat)):
        for ColIndex in xrange(0, len(PrunedTreeDistMat)):
            if RowIndex == ColIndex:
                #have to be Max similarity!
                continue
                
            Hops = (PrunedTreeDistMat[RowIndex][ColIndex]) #dist b/w row elem and col elem
            RowLabel = PrunedLabels[RowIndex]
            ColLabel = PrunedLabels[ColIndex]
            RowDepth = GetTreeDepth(RowLabel)
            ColDepth = GetTreeDepth(ColLabel)
            LCADepth = GetLCADepth(RowLabel, ColLabel)

            #similar to cannot link constraint (if the LCA is root, then don't link these to nodes)
            if 0 == LCADepth and "smy" == Type: 
                OntologySimilarityMat[RowIndex][ColIndex] = 0
                PrunedSemanticSimMat[RowIndex][ColIndex] = 0
                NormalizedPrunedDepMat[RowIndex][ColIndex] = 0
                continue
            
            OntologySimilarityMat[RowIndex][ColIndex] = GetSemanticSimValue (Type, 
                RowDepth, ColDepth, LCADepth, Hops, MaxTreeDepth)
      
    #Normalizing to range [0,1]
    OntologySimilarityMat = NormalizeInRange01(OntologySimilarityMat)

    #*** After normalizing ***
    #Setting maximum ontological similarity to the diag elements
    numpy.fill_diagonal(OntologySimilarityMat, 1)
       
    return OntologySimilarityMat, PrunedSemanticSimMat, NormalizedPrunedDepMat
        

def GetMaxLabelDepth(Labels):
    MaxDepth = 1
    for Label in Labels:
        Depth = GetTreeDepth(Label)
        if MaxDepth < Depth:
            MaxDepth = Depth
    return MaxDepth


def FormPredictedLabels (ClusterDict, Labels):
    PredictedLabels = [None] * len(Labels)
    for ClusterIdAsKey in ClusterDict:
        Elems = ClusterDict[ClusterIdAsKey]
        for Elem in Elems:
            IndexOfEleminLabels = Labels.index(Elem)
            PredictedLabels[IndexOfEleminLabels] = ClusterIdAsKey
    return PredictedLabels

def PrintClusterResultsAndExit(Labels):
    ClusterDict = {}
    for Index, Label in enumerate(Labels):
        ClusterDict[Index] = []
        ClusterDict[Index].append(Label)
        
    PredictedLabels = FormPredictedLabels(ClusterDict, Labels)
    #GT = [Line.split( )[1].strip() for Line in open(sys.argv[5], 'r').readlines()]
    #print 'Adj Rand Index = ', sklearn.metrics.adjusted_rand_score(GT, PredictedLabels)
    exit(0)



def PerformApCluster(CombinedMat, PrunedLabels):
    Preference = numpy.diag(CombinedMat)/len(CombinedMat)  
    #so, combined mat controls the preferences. Hence when we use only one sim,
    #this step should not be a problem
    Preference = Preference * Preference #this scaling might not be needed always?


    #for making self dependent packages as singleton clusters 
    #if this is not done, AP will put this pkg in some other cluster 
    ColSums = numpy.sum(CombinedMat, axis=0)
    RowSums = numpy.sum(CombinedMat,axis=1)
    for Index in xrange(0, len(CombinedMat)):
        if 0 == RowSums[Index]  and 0 == ColSums[Index]: # this should not happen, coz pruned mat has no self dep pkgs - check with debuging
            CombinedMat[Index][Index] = 1
            Preference[Index] = 1

    af = AffinityPropagation(affinity = 'precomputed', verbose=True, damping=.5, preference=Preference).fit(CombinedMat)


    ClusterCenters = af.cluster_centers_indices_
    ClusteredLabels = af.labels_
    print('Estimated number of clusters: %d' % len(ClusterCenters))
    print ClusteredLabels


    '''
    pprint (PrunedLabels)
    numpy.set_printoptions(threshold='nan')
    numpy.set_printoptions(precision=2)
    CombinedMatForPrint = map(list, CombinedMat)
    pprint (CombinedMatForPrint)
    '''

    ClusterDict = {}
    for Index, Elem in enumerate(ClusteredLabels):
        if Elem not in ClusterDict.keys():
            ClusterDict[Elem] = []
            ClusterDict[Elem].append(PrunedLabels[Index])
        else:
            ClusterDict[Elem].append(PrunedLabels[Index])

    return ClusterCenters, ClusterDict


def MergeSelfDepPkgsAsSingletonCLusters (ClusterDict, SelfDepPkgIndices, Labels):

    NextKey = len(ClusterDict)
    for Index in SelfDepPkgIndices:
        ClusterDict[NextKey] = []
        ClusterDict[NextKey].append(Labels[Index])
        NextKey += 1
    print "clusters"
   
    return ClusterDict

def NormalizeDepMat (DepMat):
    RowWiseSum = DepMat.sum(axis=1)[:,None]
    RowWiseSum [RowWiseSum == 0] = 1 #if rowwise sum is zero to avoid divide by zero error
    NormalizedDepMat = DepMat/RowWiseSum
    return NormalizedDepMat


def VisualizeColorMaps(CombinedMat, NormalizedPrunedDepMat, 
    PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels):

    '''   
    #plt.grid(True)
    #plt.subplots_adjust(bottom=0.50)
    plt.pcolor(NormalizedPrunedDepMat)
    plt.colorbar(use_gridspec=True) #to resize to the tight layout format
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat)+0.5),PrunedLabels, rotation=30,ha='right') 
    #in prev line: ha = horizontal alignment - right is used to make label terminate the the center of the grid
    plt.title("NormalizedPrunedDepMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout() #to resize so that all labels are visible
    #plt.savefig('foo.pdf',figsize=(4,4),dpi=600) # to save image as pdf, fig size may or maynot be used
    plt.show()
    

    plt.pcolor(PrunedSemanticSimMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels, rotation=45,ha='right')
    plt.title("PrunedSemanticSimMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout()
    plt.show()
    '''
    
    plt.pcolor(OntologySimilarityMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat) + 0.5),PrunedLabels, rotation=45,ha='right')
    plt.title("OntologySimilarityMat",fontsize=20,verticalalignment='bottom')
    plt.xlabel('Packages')
    plt.ylabel('Packages')
    plt.tight_layout()
    plt.show()

'''
    plt.pcolor(CombinedMat)
    plt.colorbar(use_gridspec=True)
    plt.yticks(numpy.arange(0.5,len(CombinedMat)), PrunedLabels)
    plt.xticks(numpy.arange(0.5,len(CombinedMat)),PrunedLabels, rotation=45,ha='right')
    plt.title("CombinedMat",fontsize=20,verticalalignment='bottom')
    plt.tight_layout()
    plt.savefig('foo.pdf',dpi=600)   
    plt.show()
'''
    
def InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels):
    ClusterCenters,ClusterDict = PerformApCluster(CombinedMat, PrunedLabels)

    ClusterDict = MergeSelfDepPkgsAsSingletonCLusters (ClusterDict, 
        SelfDepPkgIndices, Labels)

    pprint (ClusterDict)

    PredictedLabels = FormPredictedLabels(ClusterDict, Labels)
    print "Predcited Labels ", PredictedLabels
        

    for Index, Center in enumerate(ClusterCenters):
        print "Center Index, Center Label: ",Center,  PrunedLabels[Center]

    return ClusterDict


def SaveClustResultToFile (ClusterDict, FileName):

    FH = open(FileName, 'w')
    for ClusterIdAsKey in ClusterDict:
        LineToPrint = str(int(ClusterIdAsKey)+1) + ","
        for Elem in ClusterDict[ClusterIdAsKey]:
            LineToPrint = LineToPrint + Elem + " "
        LineToPrint = LineToPrint[:-1]
        print>>FH, LineToPrint
    FH.close()


    #GT = [Line.split( )[1].strip() for Line in open(sys.argv[5], 'r').readlines()]
    #print 'Adj Rand Index = ', sklearn.metrics.adjusted_rand_score(GT, PredictedLabels)


def GetCombinedProMat(PDGMat,SemSimMat, TaxSimMat, WStruct, WSem, WTax):
    if (0 == WStruct and 0 == WSem and 0 == WTax):
        return numpy.zeros (PDGMat.shape)

    CombinedMat = numpy.ones (PDGMat.shape)
    if (0 != WStruct):
        CombinedMat = CombinedMat * PDGMat
    if (0 != WSem):
        CombinedMat = CombinedMat * SemSimMat
    if (0 != WTax):
        CombinedMat = CombinedMat * TaxSimMat
        
    return CombinedMat    
    

'''
main
'''
def main():

    Labels = [Label.strip() for Label in open (str (sys.argv [1]), 'r').readlines()]

    StructDependencyMat = numpy.genfromtxt(sys.argv[2], dtype=float) #similarity mat
    
    TreeDistInfoMat = numpy.genfromtxt(sys.argv[4], dtype=float) #Distance mat

    WStruct = float(sys.argv[5]) #weight for struc sim (range [0,1])
    WSem = float(sys.argv[6]) #weight for struc sim (range [0,1])
    WTax = float(sys.argv[7]) #weight for struc sim (range [0,1])

    #Weights have no significance
    #Hardcoding the weights as prod similarity is used
    #WStruct = 1 
    #WSem = 1
    #WTax = 1

    ProdClustResultSaveFName = str(sys.argv[8]) 

    SumClustResultSaveFName = str(sys.argv[9])

    SimType = str(sys.argv[10])

    #Have tuned to use only Li similarity
    #my ssimalrity is not required
    #SimType = "sli"

    if 1 >= len(Labels):
        #no elements to cluster
        PrintClusterResultsAndExit(Labels)  

    #Identify self dep elements in the mat
    SelfDepPkgIndices = IdentifySelfDepPkgFromDepMat(StructDependencyMat)

    #pruning based on self dependency
    PrunedDepMat = PruneMat(StructDependencyMat, SelfDepPkgIndices)
    PrunedTreeDistMat = PruneMat (TreeDistInfoMat, SelfDepPkgIndices)
    PrunedLabels = PruneLabels (Labels, SelfDepPkgIndices)

    #semantic dist mat may or may not be present ;)
    if (os.path.getsize(sys.argv[3]) == 0): #sematic dist mat has content
        PrunedSemanticSimMat = numpy.identity(len(PrunedLabels))
    else:
        SemDistMat = numpy.genfromtxt(sys.argv[3], dtype=float)
        PrunedSemanticDistMat = PruneMat (SemDistMat, SelfDepPkgIndices)
        #converting dist to sim matrix
        PrunedSemanticSimMat = 1 - PrunedSemanticDistMat 

    #after pruning, all them turn out to be self dependent!
    if 0 == len(PrunedLabels):
        #no elements to cluster
        PrintClusterResultsAndExit(Labels)

    #This is a point where clustering must be performed!
    

    print "Weights WStruct, WSem, WTax: ", WStruct, WSem, WTax


    NormalizedPrunedDepMat = NormalizeDepMat (PrunedDepMat) 

    MaxTreeDepth = GetMaxLabelDepth(Labels)
    
    OntologySimilarityMat, PrunedSemanticSimMat, NormalizedPrunedDepMat  = \
        GetOntologySimMat (SimType, PrunedTreeDistMat, PrunedLabels, PrunedSemanticSimMat,
            NormalizedPrunedDepMat, MaxTreeDepth)

    #scaling sim values by weight
    ScaledNormalizedPrunedDepMat = WStruct * NormalizedPrunedDepMat
    ScaledPrunedSemanticSimMat = WSem * PrunedSemanticSimMat
    ScaledOntologySimilarityMat = WTax * OntologySimilarityMat
    
    
    #combining using linear (sum) combination
    WeigtedSumMat = (ScaledPrunedSemanticSimMat + \
        ScaledNormalizedPrunedDepMat + ScaledOntologySimilarityMat)/3

    #combining using non linear (prod) combination
    WeigthedProdMat = GetCombinedProMat(ScaledNormalizedPrunedDepMat, 
        ScaledPrunedSemanticSimMat,ScaledOntologySimilarityMat, WStruct, WSem,
        WTax)


    CombinedMat = WeigthedProdMat #currently considering only product
    #CombinedMat = CombinedMat/CombinedMat.max() #is this required?
    CombinedMat = NormalizeInRange01(CombinedMat)

    ClusterDict = InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels)

    #VisualizeColorMaps (CombinedMat, NormalizedPrunedDepMat, PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels)

    SaveClustResultToFile (ClusterDict, ProdClustResultSaveFName)

    CombinedMat = WeigtedSumMat #currently considering only product
    #CombinedMat = CombinedMat/CombinedMat.max() #is this required?
    CombinedMat = NormalizeInRange01(CombinedMat)

    ClusterDict = InterpretClustResults (CombinedMat, PrunedLabels, SelfDepPkgIndices, Labels)

    #VisualizeColorMaps (CombinedMat, NormalizedPrunedDepMat, PrunedSemanticSimMat, OntologySimilarityMat, PrunedLabels)

    SaveClustResultToFile (ClusterDict, SumClustResultSaveFName)


if __name__ == '__main__':
    main()