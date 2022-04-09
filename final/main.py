# Opening FreeCAD LoadNew.py / Iteration toolkit for property reading and binding
import os, sys, re
import subprocess
import itertools as it

# Import FreeCAD related modules
import FreeCAD, FreeCADGui, Part

# DoE(OLHS) and Metamodeling(Kriging)
import math
import numpy as np
import matplotlib.pyplot as plt
from pyKriging.samplingplan import samplingplan
from pyKriging.krige import kriging
#from pyKriging.CrossValidation import Cross_Validation

from sty import fg, bg, ef, rs
import logging
logging.basicConfig(filename=os.getcwd()+u"\\log", level=logging.DEBUG)



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



if __name__ == "__main__":
    FCPath = #@FCPath Placeholder

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
        m = 20
        ndv = 2
        includeEdge = 0
        WGroup = DVGroup(["W"], 10, 1000, m, includeEdge)
        HGroup = DVGroup(["H"], 10, 1000, m, includeEdge)

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
        def experiment(DVGroupList, denormalizedSampleList):
            objectiveFuncValue = []
            constraintFuncValue = []
            
            print("\n")
            for exp in denormalizedSampleList:
                print(fg.red+"Experiment #"+str(denormalizedSampleList.index(exp)+1)+fg.rs)
                print("Binding value for experiment #"+str(denormalizedSampleList.index(exp)+1))

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
                #print("!"+postprocSTDOUT.decode('UTF-8')+"\n!!"+postprocSTDERR.decode('UTF-8'))

                constraintFuncValue.append(float(postprocSTDOUT.decode('UTF-8').strip('\r\n')))
                objectiveFuncValue.append(exp[0]*exp[1])
     
                print(fg.green+"Objective function value: "+str(objectiveFuncValue[-1])+fg.rs)
                print(fg.green+"Constraint function value: "+str(constraintFuncValue[-1])+"\n\n"+fg.rs)

            return objectiveFuncValue, constraintFuncValue

        objectiveFuncValue, constraintFuncValue = experiment(DVGroupList, sampleList)


        # Kriging the result
        krig = kriging(np.array(sampleList), np.array(constraintFuncValue))
        krig.train()


        # Infill points for Kriging
        infillNum = math.floor(m/4)
        print("\nExperiments for Infill Points: ")
        for i in range(infillNum):
            newPts = krig.infill(1).tolist()
            for pt in newPts:
                infillConstraintFuncValue = experiment(DVGroupList, newPts)[1]
                constraintFuncValue = constraintFuncValue + infillConstraintFuncValue
                krig.addPoint(pt, infillConstraintFuncValue)
                sampleList.append(pt)
            krig.train()
        

        # Cross Validation of the model


        # Print the result and result assessment
        predictedResult = translatePrediction(np.array(sampleList), np.array(constraintFuncValue), krig)

        print("\nKriging result of constraint function - Max. von Mises Stress: \n"+str(translatePredictionToMatlab(["x", "y"], np.array(constraintFuncValue), krig)))
        print("\n\nConstraint function values acquired with FEA: \n"+str(constraintFuncValue))
        print("\n\nConstraint function values estimated with Kriging: \n"+str(predictedResult))
        print("\n\nR^2 = "+str(krig.rsquared(np.array(constraintFuncValue), np.array(predictedResult))))

    except Exception as e:
        logging.info(e)

    print(fg.red+"\nClear FEA related files? (y/n)"+fg.rs)
    clear = input()
    if clear == 'y' or clear == 'Y':
        os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\clean.ps1 12")
    exit()

