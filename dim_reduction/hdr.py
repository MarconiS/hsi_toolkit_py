
"""
function [dimReductionStruct] = dimReduction(img, Parameters)

Hierarchical Dimensionality Reduction
Computes KL-divergence between every pair of bands in image and merges
bands that are similar based on KL-divergence.

Inputs:
  img: hyperspectral data cube (n_row x n_col x n_bands)
  Parameters: parameter structure defined by dimReductionParameters.m

Author: Alina Zare
Email Address: azare@ufl.edu
Created: September 12, 2008
Latest Revision: October 15, 2018
Translation to Python: Caleb Robey
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import scipy.cluster.hierarchy as sch


class dimReductionParameters():
    def __init__(self):
        #self.numBands = 7  # Reduced dimensionality Size
        self.type = 'complete'  # Type of hierarchical clustering used
        self.showH = 0  # Set to 1 to show clustering, 0 otherwise
        self.NumCenters = 369.  # Number of centers used in computing KL-divergence
        self.convFactor = 1. # conversion factor for NEON AOP data

def dimReduction(img, numBands = 7, Parameters=None):

    numRows, numCols, numDims = img.shape

    if Parameters is None:
        Parameters = dimReductionParameters()

    type = Parameters.type  # Type of Hierarchy
    showH = Parameters.showH  # Set to 1 to show clustering, 0 otherwise
    maxNumClusters = int(numBands)
    NumCenters = Parameters.NumCenters
    convFactor = Parameters.convFactor

    InputData = np.reshape(img, (numRows * numCols, numDims))
    _, KLDivergencesList, _ = computeKLDivergencesBetweenBands(InputData, NumCenters, convFactor);

    Hierarchy = sch.linkage(KLDivergencesList, type)

    band_clusters = sch.fcluster(Hierarchy, t=maxNumClusters, criterion='maxclust')
    if (showH):
        # 'mtica' gives matlab behavior
        D = sch.dendrogram(Hierarchy, 0, 'mtica')
        plt.show()

    mergedData = np.zeros((maxNumClusters, (numRows * numCols)))

    for i in range(1, maxNumClusters+1):
        mergedData[i-1, :] = np.mean(InputData[:, band_clusters == i], 1)

    mergedData = np.reshape(mergedData.T, (numRows, numCols, maxNumClusters))
    return mergedData


def computeKLDivergencesBetweenBands(InputData, NumCenters, convFactor):

    DataList = InputData / convFactor

    # compute the histograms
    Centers = np.arange(1/(2*NumCenters), 1 + 1/NumCenters, 1/NumCenters)

    hists = np.zeros((NumCenters, DataList.shape[0]))

    for count in range(DataList.shape[0]):
        hists[:, count], t = np.histogram(DataList.T[:, count], Centers)

    # Add an epsilon term to the histograms
    hists = hists + np.spacing(1)

    # compute KL Divergence
    lim = InputData.shape[1]
    KLDivergences = np.zeros((lim, lim))

    for i in np.arange(DataList.shape[1]):
        for j in np.arange(DataList.shape[1]):
            KLDivergences[i, j] = (hists[i, :] * np.log(hists[i, :] / hists[j, :])).sum() \
                                  + (hists[j, :] * np.log(hists[j, :] / hists[j, :])).sum()

    temp = KLDivergences - np.diag(np.diag(KLDivergences))
    KLDivergencesList = pdist(temp)

    return KLDivergences, KLDivergencesList, hists


def getClusters(InputData, numBands = 7, Parameters=None):

    numRows, numCols, numDims = InputData.shape


    if Parameters is None:
        Parameters = dimReductionParameters()

    type = Parameters.type  # Type of Hierarchy
    showH = Parameters.showH  # Set to 1 to show clustering, 0 otherwise
    maxNumClusters = int(numBands)
    NumCenters = Parameters.NumCenters
    convFactor = Parameters.convFactor
    InputData = np.reshape(InputData, (numRows * numCols, numDims))
    print(InputData.shape)

    _, KLDivergencesList, _ = computeKLDivergencesBetweenBands(InputData, NumCenters, convFactor);

    Hierarchy = sch.linkage(KLDivergencesList, type)

    band_clusters = sch.fcluster(Hierarchy, t=maxNumClusters, criterion='maxclust')

    return(band_clusters)
