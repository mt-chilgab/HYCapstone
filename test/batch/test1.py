import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


class DVGroup:
    def __init__(self, nameList, domainLB, domainUB, ptsNum, includeEdge):
        self.nameList = nameList
        self.propNumbers = len(self.nameList)

        self.domainLB = domainLB
        self.domainUB = domainUB
        self.interval = (self.domainUB-self.domainLB)/ptsNum
        
        # Compatibility with pyKriging domain edge setting (samplingplan.py -> rls)
        self.domain = np.arange(self.domainLB+abs(includeEdge-1)*self.interval/2, self.domainUB-abs(includeEdge-1)*self.interval/2, self.interval) 
        self.domainNormalized = np.arange(0, 1, 1/ptsNum)
        self.scalingFactor = 1/(self.domainUB-self.domainLB)

        self.currentIndex = 0
        self.indexList = list(range(0, ptsNum, 1))
    
    def setAndBindValue(self, value):
        bindPropertiesWithoutObj(self.nameList, [value]*self.propNumbers)

# Design Variable Grouping
ptsNum = 2
includeEdge = 0
thicknessGroup = DVGroup(["ThicknessShroud", "ThicknessHub", "ThicknessLEShroud", "ThicknessTEShroud", "ThicknessLEHub", "ThicknessTEHub", "ThicknessLEAve", "ThicknessTEAve"], 20, 50, ptsNum, includeEdge)
diameterGroup = DVGroup(["ds"], 50, 90, ptsNum, includeEdge)

DVGroupList = [thicknessGroup, diameterGroup]

# Optimal Latin Hypercube Sampling with pyKriging Module.
ndv = len(DVGroupList)
m = ptsNum
samplingPlan = samplingplan(ndv)
normalizedSampleList = samplingPlan.optimallhc(m)
print(normalizedSampleList)

# Revert the normalized sample list to original domain
sampleList = np.zeros((m,ndv))
for i in range(m):
    sampleList[i] = list(map(lambda x: x.domainLB+(x.domainUB-x.domainLB)*normalizedSampleList[i][DVGroupList.index(x)], DVGroupList))
print(sampleList)
