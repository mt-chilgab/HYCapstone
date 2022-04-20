# Loading another python scripts(subprocess) and iteration toolkit for property reading and binding (itertools)
# Logging feature for undisplayed exceptions(logging)
import os, sys, re
import subprocess
import itertools as it
import logging
logging.basicConfig(filename=os.getcwd()+u"\\log", level=logging.DEBUG)

# Import FreeCAD related modules
import FreeCAD, FreeCADGui, Part

# DoE(OLHS) and Metamodeling(Kriging)
import math
import numpy as np
import matplotlib.pyplot as plt
from pyKriging.samplingplan import samplingplan
from pyKriging.krige import kriging
from sklearn.covariance import EllipticEnvelope
#from pyKriging.CrossValidation import Cross_Validation

# Plotting(matplotlib, mpl_toolkits) and logging
from sty import fg, bg, ef, rs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



class box:
	def __init__(self, obj):
		obj.Proxy = self
		obj.addProperty("App::PropertyFloat", "W", "Dimensions", "Width of a beam")
		obj.addProperty("App::PropertyFloat", "H", "Dimensions", "Height of a beam")
		obj.addProperty("App::PropertyFloat", "L", "Dimensions", "Length of a beam")		

	def execute (self, obj):
            b = Part.makeBox(obj.W ,obj.H ,obj.L)
            obj.Shape = b

class DVGroup:
    def __init__(self, nameList, domainLB, domainUB, ptsNum, includeEdge):
        self.nameList = nameList
        self.propNumbers = len(self.nameList)

        self.domainLB = domainLB
        self.domainUB = domainUB
        self.scalingFactor = self.domainUB-self.domainLB
        self.interval = self.scalingFactor/ptsNum
        
        # Compatibility with pyKriging domain edge setting (samplingplan.py -> rls)
        self.domain = np.arange(self.domainLB+abs(includeEdge-1)*self.interval/2, self.domainUB-abs(includeEdge-1)*self.interval/2, self.interval) 
        self.domainNormalized = np.arange(abs(includeEdge-1)*0.5/ptsNum, 1-abs(includeEdge-1)*0.5/ptsNum, 1/ptsNum)

        self.currentIndex = 0
        self.indexList = list(range(0, ptsNum, 1))
    
    def bindValue(self, value):
        value = float(value)
        bindPropertiesWithoutObj(self.nameList, [value]*self.propNumbers)


# Known types are: float, int, float vector(fvec), int vector(ivec), boolean(bool)
def typeParser(typ, value):
    if(typ == "float"):
        if(value):
            return float(value)
        else:
            return None
    
    elif(typ == "int"):
        if(value):
            return int(value)
        else:
            return None

    elif(typ == "bool"):
        if((value == "true") | (value == "True") | (value == "t") | (value == "T")):
            return True
        elif((value == "false") | (value == "False") | (value == "f") | (value == "F")):
            return False
        else:
            return None
    
    elif(typ == "fvec"):
        insideParenthesis = re.search(r'(?<=[\(\{\[<])[\s\t]*.+[\s\t]*(?=[\)\}\]>])', value)
        if(insideParenthesis):
            ip = value[insideParenthesis.start():insideParenthesis.end()]
            return tuple(map(lambda value: float(value), ip.split(",")))
        else:
            return None
           
    elif(typ == "ivec"):
        insideParenthesis = re.search(r'(?<=[\(\{\[<])[\s\t]*.+[\s\t]*(?=[\)\}\]>])', value)
        if(insideParenthesis):
            ip = value[insideParenthesis.start():insideParenthesis.end()]
            return tuple(map(lambda value: int(value), ip.split(",")))
        else:
            return None
    
    else:
        return None 

# Searches type, variable name, value in line(that does not start with space/tab/#)
def readNameValue(): 
    typeList = list()
    nameList = list()
    valueList = list()
    
    typeSearch = (lambda line: re.search(r'^float|int|fvec|ivec|bool{1}', line))
    # t is for look-around regex search -> stripped down typeSearch result in string
    nameSearch = (lambda line, t: re.search(r'(?<=^'+t+r').+[\s\t]*(?==)', line))
    valueSearch = (lambda line: re.search(r'(?<==).+$', line))
    stripPattern = re.compile(r'[\s\t]')

    with open(os.getcwd()+"\\parameters.txt", mode='r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if(not(re.search(r'^[#\s\t(matprop)]', line))):
                typ = typeSearch(line)
                if(typ):
                    t = stripPattern.sub('', line[typ.start():typ.end()])
                    typeList.append(t)
                   
                    name = nameSearch(line, t)
                    if(name):
                        n = stripPattern.sub('', line[name.start():name.end()])
                        nameList.append(n)
     
                    value = valueSearch(line)
                    if(value):
                        v = typeParser(t, stripPattern.sub('', line[value.start():value.end()]))
                        if(v is not None):
                            valueList.append(v)
    f.close()

    if( (len(typeList) != len(nameList)) | (len(typeList) != len(valueList)) | (len(nameList) != len(valueList)) ):
        return None

    return (nameList, valueList)

# Bind properties.txt -> FreeCAD object, only for properties that are in updateList
def bindProperties(obj, updateList, nameValueList):
    bindOK = False
    
    if(bool(nameValueList) & bool(updateList)):
        for (name, value) in zip(nameValueList[0], nameValueList[1]):
            if((name in obj.PropertiesList) & (name in updateList)):
                setattr(obj, name, value)
        bindOK = True

    if(not bindOK):
        FreeCAD.Console.PrintMessage("Binding properties to given object failed.\n")

    return bindOK

# Bind properties without object input
def bindPropertiesWithoutObj(nameList, valueList):
    for (name, value) in zip(nameList, valueList):
        obj = findObj(name)
        if(name in obj.PropertiesList):
            setattr(obj, name, value)
            print("\tbind to "+str(name)+": "+str(value))

# Finds which object given property belongs to
def findObj(propertyName):
    returnValue = 0
 
    for obj in FreeCAD.ActiveDocument.Objects:
        if propertyName in obj.PropertiesList:
            returnValue = obj
    
    return returnValue

def translatePredictionToMatlab(dvNameList, sampleValueVector, k):
    k.updateData()
    k.updatePsi()

    # As sample points on kriging instance are normalized, we should de-normalized it.
    # We do this by pre-defined scalingFactor list.
    samplePoints = np.zeros((k.X.shape[0], k.X.shape[1]))
    scalingFactorList = np.zeros(k.X.shape[1])
    for i in range(samplePoints.shape[0]):
        samplePoints[i] = k.inversenormX(k.X[i])
        print("matlab prediction sample point "+str(i)+": ", samplePoints[i])

    for i in range(samplePoints.shape[1]):
        scalingFactorList[i] = k.normRange[i][1]-k.normRange[i][0]

    pList = k.pl.tolist()
    thetaList = k.theta.tolist()
    weightList = np.dot(np.linalg.inv(k.Psi), sampleValueVector-k.mu*np.ones(samplePoints.shape[0])).tolist()[-1]
    
    # Translate basis expression to MATLAB syntax
    # list for theta / p / dvNameList has same element number = ndv 
    basisList = []
    for i in range(samplePoints.shape[0]):
        basisExponent = []
        for (j, theta, p) in zip(range(samplePoints.shape[1]), thetaList, pList):
            basisExponent.append(str(theta)+"*"+"abs(("+str(samplePoints[i][j])+"-"+str(dvNameList[j])+")/"+str(scalingFactorList[j])+")^"+str(p))
        basisList.append("exp(-("+"+".join(basisExponent)+"))")

    for i in range(len(basisList)):
        basisList[i] = "("+str(weightList[i])+")"+"*"+basisList[i]

    return str(k.mu)+"+"+"+".join(basisList)

def translatePrediction(pointsList, sampleValueVector, k):
    k.updateData() 
    k.updatePsi()
    
    # As sample points on kriging instance are normalized, we should de-normalized it.
    # We do this by pre-defined scalingFactor list.
    samplePoints = np.zeros((k.X.shape[0], k.X.shape[1]))
    scalingFactorList = np.zeros(k.X.shape[1])
    for i in range(samplePoints.shape[0]):
        samplePoints[i] = k.inversenormX(k.X[i])

    for i in range(samplePoints.shape[1]):
        scalingFactorList[i] = k.normRange[i][1]-k.normRange[i][0]

    pList = k.pl.tolist()
    thetaList = k.theta.tolist()
    weightList = np.dot(np.linalg.inv(k.Psi), sampleValueVector-k.mu*np.ones(samplePoints.shape[0])).tolist()[-1]
    
    resultValueList = []
    for point in pointsList:
        resultValue = 0 
        for i in range(samplePoints.shape[0]):
            basisExponentValue = 0
            for (j, theta, p) in zip(range(samplePoints.shape[1]), thetaList, pList):
                basisExponentValue += -1*theta*np.power(abs((samplePoints[i][j]-point[j])/scalingFactorList[j]), p)
            resultValue += weightList[i]*np.exp(basisExponentValue)
        resultValueList.append(resultValue+k.mu)

    return resultValueList

def matlabPlot():
    string = []
    string.append('x1=100:1:200;\n')
    string.append('x2=100:1:200;\n\n')
    string.append('s = 101;\n')
    string.append('for i = 1:s\n')
    string.append('\tfor j = 1:s\n')
    string.append('\t\tx = x1(1,i);\n')
    string.append('\t\ty = x2(1,j);\n')
    string.append('\t\tf(j,i) = ')
    string.append(str(translatePredictionToMatlab(["x", "y"], np.array(constraintFuncValue), krig)))
    string.append(';\n')
    string.append('\tend\n')
    string.append('end\n\n')
    string.append('surf(x1,x2,f)\n\n')
    string.append('hold on\n\n')
    string.append(r'[X,Y] = readvars("valueset.txt");'+'\n')
    string.append(r'[a,b,c,Z] = readvars("objlist.txt");'+'\n\n')
    string.append(r'scatter3(X,Y,Z,"filled","MarkerFaceColor","r");')

    with open('testplot.m','w',encoding='UTF-8') as f:
        for line in string:
            f.write(line)

def printsampleList():
    str_list = []
    for i in range(math.floor(m)):
        mapped_line = sampleList[i]
        str_line = []
        for j in range(ndv):
            str_line.append(format(mapped_line[j],'>15,.6E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('valueset.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

def printobjList():
    str_list = []
    for i in range(math.floor(m)):
        obj_line = constraintFuncValue[i]
        str_line = []
        str_line.append(format(obj_line,'>15,.6E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('objlist.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

# Plots Kriging prediction in 3D surface plot. Select two design variables from DVGroupList with dvIndex1, dvIndex2
# and then assign values for another design variables with anotherDVs
def plotPrediction(DVGroupList, dvIndex1, dvIndex2, anotherDVs, div, sampleValueVector, k):
    if dvIndex1 > dvIndex2:
        dvIndex1, dvIndex2 = dvIndex2, dvIndex1

    if len(anotherDVs)+2 == len(DVGroupList):
        dv1 = np.linspace(DVGroupList[dvIndex1].domainLB, DVGroupList[dvIndex1].domainUB, div).tolist()
        dv2 = np.linspace(DVGroupList[dvIndex2].domainLB, DVGroupList[dvIndex2].domainUB, div).tolist()
        
        ptsList = list(it.product(dv1, dv2))
        if len(anotherDVs) > 0:
            ptsList = [ anotherDVs[0:dvIndex1]+[pt[0]]+anotherDVs[dvIndex1:dvIndex2-1]+[pt[1]]+anotherDVs[dvIndex2-1:] for pt in ptsList ]
        pred = np.array(translatePrediction(ptsList, sampleValueVector, k))
       
        # For scatter plot, extract samplePts including Design Variable 1, 2 values
        samplePts = np.zeros((k.X.shape[0], 2))
        for i in range(k.X.shape[0]):
            samplePtList = k.inversenormX(k.X[i]).tolist()
            samplePts[i] = np.array([samplePtList[dvIndex1], samplePtList[dvIndex2]])
        samplePts = samplePts.T
        print("samplePts used for scatter(plotPrediction)", samplePts)
        
        dv1, dv2 = np.meshgrid(np.array(dv1), np.array(dv2))
        pred = np.reshape(pred, (div, div))
        analytic = 1.2*np.power(10, 9)/np.power(dv2,2)/dv1

        fig = plt.figure(figsize = (14, 9))
        ax = plt.axes(projection = '3d')
        ax.set_xlabel('Design Variable 1')
        ax.set_ylabel('Design Variable 2') 
        myCmap = plt.get_cmap('plasma')

        surf = ax.plot_surface(dv1, dv2, pred, cmap=myCmap, edgecolor='none')
        surfAnalytic = ax.plot_wireframe(dv1, dv2, analytic, cmap='seismic')
        scatter = ax.scatter(samplePts[0], samplePts[1], sampleValueVector, marker='x', s=100, c='black')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_title('Kriging Prediction')

        plt.show()

    else:
        return


if __name__ == "__main__":
    FCPath = u'C:\\Users\\Grant\\AppData\\Local\\Programs\\FreeCAD 0.19'

    try:
        FreeCADGui.showMainWindow()
        FreeCADGui.updateGui()

        with open(FCPath+u'\\data\\Mod\\Start\\StartPage\\LoadNew.py') as file:
            exec(file.read())

        initialNameValueList = readNameValue()

        myObj = FreeCAD.ActiveDocument.addObject("Part::FeaturePython", "box") 
        box(myObj)
        if(bindProperties(myObj, initialNameValueList[0], initialNameValueList)):
            myObj.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too
            FreeCAD.ActiveDocument.recompute()

        path_os = os.getcwd()
        path_gui = re.sub(r'\\', '/',path_os)


        # Design Variable Grouping
        m = 10
        ndv = 2
        includeEdge = 0
        WGroup = DVGroup(["W"], 100, 1000, m, includeEdge)
        HGroup = DVGroup(["H"], 100, 1000, m, includeEdge)

        DVGroupList = [WGroup, HGroup]
        if len(DVGroupList) != ndv:
            print("DVGroupList length is not equal to given number of design variables")
            exit()


        print("\n\n")
        # Optimal Latin Hypercube Sampling with pyKriging Module.
        samplingPlan = samplingplan(ndv)
        normalizedSampleList = samplingPlan.optimallhc(m)
        print("\nOptimal LHS Result(normalized):\n", normalizedSampleList)


        # Revert the normalized sample list to original domain
        def denormalizePtsList(DVGroupList, normalizedSampleList):
            normalizedSampleList = normalizedSampleList.tolist()
            sampleList = np.zeros((len(normalizedSampleList), len(DVGroupList)))
            for i in range(len(normalizedSampleList)):
                sampleList[i] = list(map(lambda x: x.domainLB+x.scalingFactor*normalizedSampleList[i][DVGroupList.index(x)], DVGroupList))
       
            return sampleList.tolist()
        
        sampleList = denormalizePtsList(DVGroupList, normalizedSampleList)
        print("\nOptimal LHS Result:\n", sampleList)

        # Experiment with points calculated from OLHS
        def experiment(DVGroupList, denormalizedSampleList, expNum=None):
            objectiveFuncValue = []
            constraintFuncValue = []
            
            print("\n")
            for exp in denormalizedSampleList:
                if expNum is None:
                    print(fg.red+"Experiment #"+str(denormalizedSampleList.index(exp)+1)+fg.rs)
                    print("Binding value for experiment #"+str(denormalizedSampleList.index(exp)+1))
                elif type(expNum).__name__ == 'int':
                    print(fg.red+"Experiment #"+str(expNum)+fg.rs)
                    print("Binding value for experiment #"+str(expNum))
               
                for i in range(len(DVGroupList)):
                    DVGroupList[i].bindValue(exp[i])  
                    
                for obj in FreeCAD.ActiveDocument.Objects:
                    obj.touch()
                FreeCAD.ActiveDocument.recompute()
                print("Recompute ok")

                __objs__ = []
                __objs__.append(FreeCAD.getDocument("Unnamed").getObject("box"))
                Part.export(__objs__,path_gui+u"/box.step")
                del __objs__
                print("Step export ok")

                preprocExec = subprocess.Popen([sys.executable, os.getcwd()+u"\\preprocess.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)         
                preprocSTDOUT, preprocSTDERR = preprocExec.communicate() 
                #print("!"+preprocSTDOUT.decode('UTF-8')+"\n!!"+preprocSTDERR.decode('UTF-8'))

                ccxDir = os.getcwd()+u"\\ccx\\etc"
                femExec = subprocess.Popen([ccxDir+u"\\runCCXnoCLS.bat", os.getcwd()+u"\\ex1static1.inp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                femSTDOUT, femSTDERR = femExec.communicate()
                #print("!"+femSTDOUT.decode('UTF-8')+"\n!!"+femSTDERR.decode('UTF-8'))

                postprocExec = subprocess.Popen([sys.executable, os.getcwd()+u"\\postprocess.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                postprocSTDOUT, postprocSTDERR = postprocExec.communicate()
                os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\caseSaver.ps1 "+str(exp[0])+"_"+str(exp[1]))
                #print("!"+postprocSTDOUT.decode('UTF-8')+"\n!!"+postprocSTDERR.decode('UTF-8'))

                constraintFuncValue.append(float(postprocSTDOUT.decode('UTF-8').strip('\r\n')))
                objectiveFuncValue.append(exp[0]*exp[1])
     
                print(fg.green+"Objective function value: "+str(objectiveFuncValue[-1])+fg.rs)
                print(fg.green+"Constraint function value: "+str(constraintFuncValue[-1])+"\n\n"+fg.rs)

            return objectiveFuncValue, constraintFuncValue

        objectiveFuncValue, constraintFuncValue = experiment(DVGroupList, sampleList)


        # Outlier Detection with MCD
        def detectOutliersMCD(sampleList, constraintFuncValue, deleteOutlierPts=True, infill=False, samplePtsBeforeInfill=None):
            yObserved = np.array(constraintFuncValue)
            X = np.array(sampleList)
            
            augX = np.append(X, np.reshape(yObserved, (yObserved.shape[0],1)), axis=1)
            print("augX: ", augX)

            outlierDetection = EllipticEnvelope(contamination=0.1)
            print("\nDetecting Outliers with Minimum Covariance Determinant (MCD): ")
            yHat = outlierDetection.fit_predict(augX)
            print("Outliers(-1): ", yHat)
            mask = yHat != -1
            
            # This if statement treats sample points and observation at the point as inlier
            # so that only infill points can be detected as outlier.
            # Should decide whether this is backed with statistical theories or not...
            if infill or samplePtsBeforeInfill is not None:
                for i in range(samplePtsBeforeInfill):
                    mask[i] = True
            if deleteOutlierPts:
                sampleListInlier, constraintFuncValueInlier = X[mask].tolist(), yObserved[mask].tolist()
                print("Outlier removed dataset: \n\tsampleList: ", sampleList)
                print("\tConstraint Func. Value: ", constraintFuncValue)

                return sampleListInlier, constraintFuncValueInlier, mask

            else:
                return mask


        # Simple median-absoulte-deviation(MAD) based 1D outlier detection for checkup
        def detectOutliersMAD(constraintFuncValue, threshold=3.5):
            median = np.median(constraintFuncValue, axis=0)
            diff = np.abs(constraintFuncValue - median*np.ones(constraintFuncValue.shape[0]))
            mad = np.median(diff)
            modifiedZScore = 0.6745*diff/mad

            return modifiedZScore > threshold


        # Redo experiments 5 times on outlier points
        outlierMask = detectOutliersMCD(sampleList, constraintFuncValue, deleteOutlierPts=False)
        for i in range(len(outlierMask)):
            testObservations = []
            if not outlierMask[i]:
                print("Re-doing experiment for detected outlier point: ", sampleList[i])
                expRep = 7

                for j in range(expRep):
                    testObservations.append(*experiment(DVGroupList, [sampleList[i]], j+1)[1])
                testObservations = np.array(testObservations)

                checkupMask = testObservations[detectOutliersMAD(testObservations).tolist()]
                testObservations = testObservations[checkupMask]

                print("Outlier removed checkup observations: ", testObservations)

                testObservations.append(constraintFuncValue[i])
                checkupMask = testObservations[detectOutliersMAD(testObservations).tolist()]

                if checkupMask[-1]:
                    outlierMask[i] = True

        print("rearranged outlier mask: ", outlierMask)
        sampleList, constraintFuncValue = sampleList[outlierMask], constraintFuncValue[outlierMask]
                
                    
        # Kriging the result
        krig = kriging(np.array(sampleList), np.array(constraintFuncValue))
        krig.train()


        # Infill points for Kriging + outlier detection with MCD
        infillNum = 2
        outilerMask = []
        print("\nExperiments for Infill Points: ")
        for i in range(infillNum):
            # infill criteria are either 'ei' (expected improvement) or 'error' (point of biggest MSE)
            # and also, remember to set addPoint=False so we can decide whether a infill point and its observation is outlier.
            newPts = krig.infill(1, method='ei', addPoint=False).tolist()

            sampleNumBeforehand = len(sampleList)
            sampleList += newPts
            for j in range(len(newPts)):
                constraintFuncValue.append(*experiment(DVGroupList, [newPts[j]], 3*i+j+1)[1])

            sampleList, constraintFuncValue, outlierMask = detectOutliersMCD(sampleList, constraintFuncValue, deleteOutlierPts=True, infill=True, samplePtsBeforeInfill=sampleNumBeforehand) 
            print("outlier removed: ", constraintFuncValue) 
            for j in range(sampleNumBeforehand, len(sampleList), 1):
                krig.addPoint(np.array(sampleList[j]), constraintFuncValue[j])
                krig.train()
                        
            print("sampleList: ", sampleList, "\n")
            print("constFuncValue: ", constraintFuncValue, "\n")
            print("outlierMask: ", outlierMask, "\n")
            print("k.X length: ", krig.X.shape[0], "\n")
            print("k.y length: ", krig.y.shape[0], "\n")

        # Cross Validation of the model


        # Print the result and result assessment
        predictedResult = translatePrediction(np.array(sampleList), np.array(constraintFuncValue), krig)

        print("\nKriging result of constraint function - Max. von Mises Stress: \n"+str(translatePredictionToMatlab(["x", "y"], np.array(constraintFuncValue), krig)))
        print("\n\nConstraint function values acquired with FEA: \n"+str(constraintFuncValue))
        print("\n\nConstraint function values estimated with Kriging: \n"+str(predictedResult))
        print("\n\nR^2 = "+str(krig.rsquared(np.array(constraintFuncValue), np.array(predictedResult))))

        #matlabPlot()
        #printsampleList()
        #printobjList()
        plotPrediction(DVGroupList, 0, 1, [], 100, np.array(constraintFuncValue), krig)

    except Exception as e:
        logging.info(e)

    print(fg.red+"\nClear FEA related files? (y/n)"+fg.rs)
    clear = input()
    if clear == 'y' or clear == 'Y':
        os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\clean.ps1 123")
    
    print(fg.red+"\nClear txt files(valueset, objlist, comparison)? (y/n)"+fg.rs)
    clearTxt = input()
    if clearTxt == 'y' or clearTxt == 'Y':
        os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\cleanTxt.ps1")
    
    exit()

