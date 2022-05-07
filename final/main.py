# Loading another python scripts(subprocess) and iteration toolkit for property reading and binding (itertools)
# Logging feature for undisplayed exceptions(logging)
import os, sys, re
import subprocess
import itertools as it
import logging
logging.basicConfig(filename=os.getcwd()+u"\\log", level=logging.DEBUG)

# Import FreeCAD related modules
import FreeCAD, FreeCADGui
import Part, BOPTools.SplitAPI

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



class Meridional:
    def __init__(self,obj):
        obj.Proxy = self  
        obj.addProperty("App::PropertyFloat", "R", "Dimensions", "Radius of Hub Circle").R=187.17
        obj.addProperty("App::PropertyVector", "HubCenter", "Center of Geometry", "Point of a Hub Circle Center").HubCenter = (-45.6,219.4,0)
        obj.addProperty("App::PropertyFloat", "a", "Dimensions", "Major Radius of Shroud Ellipse").a = 105
        obj.addProperty("App::PropertyFloat", "b", "Dimensions", "Minor Radius of Shroud Ellipse").b = 95.25
        obj.addProperty("App::PropertyVector", "EllipseCenter", "Center of Geometry", "Point of a Shroud Ellipse Center").EllipseCenter = (21.3,208,0)
        obj.addProperty("App::PropertyFloat", "D2", "Dimensions", "Diameter of the impeller").D2 = 400
        obj.addProperty("App::PropertyFloat", "ds", "Dimensions", "Diameter of the shaft").ds = 70
        obj.addProperty("App::PropertyFloat", "ThicknessHub", "Dimensions", "Thickness of hub").ThicknessHub = 10

    def execute (self, obj):
        draftAxis = FreeCAD.Vector(0,0,1)
        HubCircle = Part.Circle(obj.HubCenter , draftAxis, obj.R)        
        ShroudEllipse = Part.Ellipse(obj.EllipseCenter, obj.a, obj.b)
        AveCurve = Part.Ellipse((obj.HubCenter+obj.EllipseCenter)/2, (obj.R+obj.a)/2, (obj.R+obj.b)/2)

        HCS = HubCircle.toShape()
        SES = ShroudEllipse.toShape()
        AES = AveCurve.toShape()

        SP1 = FreeCAD.Vector(0,0,0)
        SP2 = FreeCAD.Vector(0,obj.D2/2,0)
        SP3 = FreeCAD.Vector(obj.D2/2,obj.D2/2,0)
        SP4 = FreeCAD.Vector(obj.EllipseCenter.x,0,0)
        SP5 = FreeCAD.Vector(obj.EllipseCenter.x,obj.D2/2,0)

        splitLine1 = Part.LineSegment(SP1,SP2).toShape()
        splitLine2 = Part.LineSegment(SP2,SP3).toShape()
        splitLine3 = Part.LineSegment(SP4,SP5).toShape()

        Bases = Part.Compound([HCS,SES, AES])

        cutted1 = BOPTools.SplitAPI.booleanFragments([Bases,splitLine1,splitLine2],"Split",0.0)
        
        HubEdge = Part.Compound([cutted1.Edges[9]])
        ShroudEdge = Part.Compound([cutted1.Edges[12]])
        AveEdge = Part.Compound([cutted1.Edges[15]])
        Edges = Part.Compound([HubEdge,ShroudEdge, AveEdge])

        cutted2 = BOPTools.SplitAPI.booleanFragments([Edges,splitLine3],"Split",0.0) 

        HubEdge1 = Part.Compound([cutted2.Edges[4]])
        HubEdge2 = Part.Compound([cutted2.Edges[5]])
        ShroudEdge1 = Part.Compound([cutted2.Edges[6]])
        ShroudEdge2 = Part.Compound([cutted2.Edges[7]])
        AveEdge = Part.Compound([cutted2.Edges[9]])

        inletEdge = Part.Compound([cutted1.Edges[1], cutted1.Edges[2]])
        bladeEdge = Part.Compound([cutted2.Edges[1], cutted2.Edges[2]])
        outletEdge = Part.Compound([cutted1.Edges[5], cutted1.Edges[6]])

        w = Part.Compound([HubEdge1,HubEdge2,ShroudEdge1,ShroudEdge2,inletEdge,bladeEdge,outletEdge])

        HubSurface = HubEdge.revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        ShroudSurface = ShroudEdge.revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        obj.Shape = Part.Compound([w, ShroudSurface, HubSurface])

class ModelOfBlade3D:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyFloat", "AngleBeta1Shroud", "Angles of a Shroud streamline", "Inlet angle").AngleBeta1Shroud=20
        obj.addProperty("App::PropertyFloat", "AngleBeta2Shroud", "Angles of a Shroud streamline", "Outlet angle").AngleBeta2Shroud=85
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint1Shroud=(25,24.5,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint2Shroud=(50,29,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint3Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint3Shroud=(75,34,0)
        obj.addProperty("App::PropertyBool", "AnglesChartShroud", "Angles of a Shroud streamline", "It is shows chart angle Beta-Streamline").AnglesChartShroud=False

         
        obj.addProperty("App::PropertyFloat", "AngleBeta1Hub", "Angles of a Hub streamline", "Inlet angle").AngleBeta1Hub=40.05
        obj.addProperty("App::PropertyFloat", "AngleBeta2Hub", "Angles of a Hub streamline", "Outlet angle").AngleBeta2Hub=83.15
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint1Hub=(25,45,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint2Hub=(50,50,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint3Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active").AngleBetaPoint3Hub=(75,55,0)
        obj.addProperty("App::PropertyBool", "AnglesChartHub", "Angles of a Hub streamline", "It is shows chart angle Beta-Streamline").AnglesChartHub=False

        obj.addProperty("App::PropertyFloat", "AngleAlpha1", "Relative theta delay between hub and shroud streamline", "Inlet angle").AngleAlpha1=0
        obj.addProperty("App::PropertyFloat", "AngleAlpha2", "Relative theta delay between hub and shroud streamline", "Outlet angle").AngleAlpha2=10
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint1", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleAlphaPoint1=(25,3,0)
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint2", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleAlphaPoint2=(50,5,0)
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint3", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active").AngleAlphaPoint3=(75,7,0)
 
        obj.addProperty("App::PropertyInteger", "N", "Number of the calculation points").N=1000
 
    def execute(self, obj):
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")
        N = obj.N

        # Adds the last vertex because it uses forward finite differentials
        hubEdge = Mer.Shape.Edges[1]
        hubEdgeDiscret = hubEdge.discretize(Number = N)
        shroudEdge = Mer.Shape.Edges[3]
        shroudEdgeDiscret = shroudEdge.discretize(Number = N)
        aveEdge = Mer.Shape.Edges[4]
        aveEdgeDiscret = aveEdge.discretize(Number = N)

        def streamlineBladeByBeta(betaPointsList, AngleBetaInitial, AngleBetaFinal, DiscretEdgeList):
            betaPoints  = [FreeCAD.Vector(0, AngleBetaInitial, 0)] + betaPointsList + [FreeCAD.Vector(100, AngleBetaFinal, 0)]
            betaCurve = Part.BSplineCurve()
            betaCurve.interpolate(betaPoints)
            betaCurveDiscret = []

            for i in range(0, N, 1):
                lowerBound = FreeCAD.Vector(100.*float(i)/float(N-1), -180, 0)
                upperBound = FreeCAD.Vector(100.*float(i)/float(N-1), 180, 0)

                constLine = Part.LineSegment(lowerBound, upperBound)
                betaIntersect = betaCurve.intersectCC(constLine)
                
                betaCurveDiscret.append(betaIntersect)


            # From ds / (r tan(beta)) = dTheta,
            # As we have beta curve and can calculate ds and r, we can calculate theta value for each meridional discretized points
            # -> would lead to physical postion on a streamline
           
            # Get the beta value(beta curve) of corresponding meridional discretized points
            beta_i = list(map(lambda beta: beta[0].Y, betaCurveDiscret))
            betaMid_i = [(beta_i[i+1]+beta_i[i])/2 for i in range(0, N-1, 1)]
            
            # Get the value of r's for each discretized points of hub/shroud edge
            r_i = list(map(lambda merPosVec: merPosVec.y, DiscretEdgeList))
            rMid_i = [(r_i[i+1]+r_i[i])/2 for i in range(0, N-1, 1)]

            # Get the length of meridional differential line segments(ds) corresponding to each meridional discretized points
            ds_i = []
            for i in range(0, N-1, 1):
                xDirLength = DiscretEdgeList[i+1].x - DiscretEdgeList[i].x
                yDirLength = DiscretEdgeList[i+1].y - DiscretEdgeList[i].y
                ds_i.append(np.sqrt(np.power(xDirLength, 2)+np.power(yDirLength, 2)))
            
            # Now calculate corresponding theta of corresponding meridional discretized points
            dTheta_i = [180.*ds_i[i]/(rMid_i[i]*np.pi*np.tan(np.pi*betaMid_i[i]/180.)) for i in range(0, N-1, 1)]
            Theta_i = [0, ]
            for i in range(0, N-1, 1):
                Theta_i.append(Theta_i[i] + dTheta_i[i])
            
            # Finally, get the physical coordinate in Cartesian coordinate xyz
            physCoordX_i = list(map(lambda merPosVec: merPosVec.x, DiscretEdgeList))
            physCoordY_i = list(map(lambda merPosVec, theta: merPosVec.y*np.cos(np.pi*theta/180.), DiscretEdgeList, Theta_i))
            physCoordZ_i = list(map(lambda merPosVec, theta: merPosVec.y*np.sin(np.pi*theta/180.), DiscretEdgeList, Theta_i))

            # B-Spline of physCoords = desired streamline
            coordList = list(map(lambda X, Y, Z: FreeCAD.Vector(X, Y, Z), physCoordX_i, physCoordY_i, physCoordZ_i))
        
            streamline = Part.BSplineCurve()
            streamline.interpolate(coordList)

            return streamline.toShape(), beta_i, Theta_i

        
        def relativeStreamlineByAlpha(alphaPointsList, AngleAlphaInitial, AngleAlphaFinal, DiscretEdgeList, DiscretEdgeListRelative, thetaiRelative):
            alphaPoints = [FreeCAD.Vector(0, AngleAlphaInitial, 0)] + alphaPointsList + [FreeCAD.Vector(100, AngleAlphaFinal, 0)]
            alphaCurve = Part.BSplineCurve()
            alphaCurve.interpolate(alphaPoints)

            alphaCurveDiscret = []
            gammaDiscret = []

            for i in range(0, N, 1):
                lowerBound = FreeCAD.Vector(100.*float(i)/float(N-1), -180, 0)
                upperBound = FreeCAD.Vector(100.*float(i)/float(N-1), 180, 0)
                constLine = Part.LineSegment(lowerBound, upperBound)
                
                alphaIntersect = alphaCurve.intersectCC(constLine)
                alphaCurveDiscret.append(alphaIntersect)

            for i in range(0, N, 1):
                discretPtsXDistance = DiscretEdgeListRelative[i].x - DiscretEdgeList[i].x
                discretPtsYDistance = DiscretEdgeListRelative[i].y - DiscretEdgeList[i].y

                discretPtsDistance = np.sqrt(np.power(discretPtsXDistance, 2) + np.power(discretPtsYDistance, 2))
                
                gammaDiscret.append(2*np.arcsin(discretPtsDistance*np.tan(np.pi*alphaIntersect[0].Y/180.)/(2*DiscretEdgeList[i].y)))
        
            physCoordList = []
            rotMat = lambda gamma: np.array([[1, 0, 0], [0, np.cos(gamma), np.sin(gamma)], [0, -1*np.sin(gamma), np.cos(gamma)]])
            for i in range(0, N, 1):
                physCoord = FreeCAD.Vector(np.dot(rotMat(-np.pi*thetaiRelative[i]/180.+gammaDiscret[i]), DiscretEdgeList[i]))
                physCoordList.append(physCoord)

            streamline = Part.BSplineCurve()
            streamline.interpolate(physCoordList)
            
            return streamline.toShape()


        streamlineHub, betaiHub, thetaiHub = streamlineBladeByBeta([obj.AngleBetaPoint1Hub, obj.AngleBetaPoint2Hub, obj.AngleBetaPoint3Hub], obj.AngleBeta1Hub, obj.AngleBeta2Hub, hubEdgeDiscret)       
        streamlineShroud, betaiShroud, thetaiShroud = streamlineBladeByBeta([obj.AngleBetaPoint1Shroud, obj.AngleBetaPoint2Shroud, obj.AngleBetaPoint3Shroud], obj.AngleBeta1Shroud, obj.AngleBeta2Shroud, shroudEdgeDiscret)
        #streamlineShroud = relativeStreamlineByAlpha([obj.AngleAlphaPoint1, obj.AngleAlphaPoint2, obj.AngleAlphaPoint3], obj.AngleAlpha1, obj.AngleAlpha2, shroudEdgeDiscret, hubEdgeDiscret, thetaiHub)
       
        streamlineHubBSpline = Part.BSplineCurve()
        streamlineShroudBSpline = Part.BSplineCurve()

        streamlineHubBSpline.interpolate(streamlineHub.discretize(Number = 10))
        streamlineShroudBSpline.interpolate(streamlineShroud.discretize(Number = 10))
        
        # Two options available: using beta profile only, using both beta and alpha profile
        bladeSurf = Part.makeLoft([streamlineShroudBSpline.toShape(), streamlineHubBSpline.toShape()])
        
        obj.Shape = bladeSurf

class Blades:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyFloat", "ThicknessLEShroud", "Shroud profile", "Value of the LE").ThicknessLEShroud = 1.3
        obj.addProperty("App::PropertyFloat", "ThicknessPoint1Shroud", "Shroud profile", "Point for B-Spline thickness").ThicknessPoint1Shroud = 1.3
        obj.addProperty("App::PropertyFloat", "ThicknessPoint2Shroud", "Shroud profile", "Point for B-Spline thickness").ThicknessPoint2Shroud = 1.3
        obj.addProperty("App::PropertyFloat", "ThicknessTEShroud", "Shroud profile", "Value of the TE").ThicknessTEShroud = 1.3

        obj.addProperty("App::PropertyFloat", "ThicknessLEHub", "Hub profile", "Value of the thickness LE").ThicknessLEHub = 3.0
        obj.addProperty("App::PropertyFloat", "ThicknessPoint1Hub", "Hub profile", "Point for B-Spline thickness").ThicknessPoint1Hub = 3.0
        obj.addProperty("App::PropertyFloat", "ThicknessPoint2Hub", "Hub profile", "Point for B-Spline thickness").ThicknessPoint2Hub = 3.0
        obj.addProperty("App::PropertyFloat", "ThicknessTEHub", "Hub profile", "Value of the thickness TE").ThicknessTEHub = 3.0

        obj.addProperty("App::PropertyBool", "TraillingEdgeEllipse", "Type of the LE and TE", "Type of the trailling edge").TraillingEdgeEllipse = False
        obj.addProperty("App::PropertyInteger", "LeadingEdgeType", "Type of the LE and TE", "Type of the leading edge").LeadingEdgeType = 1
        obj.addProperty("App::PropertyInteger", "TraillingEdgeType", "Type of the LE and TE", "Type of the trailling edge" ).TraillingEdgeType = 1
        obj.addProperty("App::PropertyInteger", "NumberOfBlades", "Number of blades").NumberOfBlades=24

        obj.addProperty("App::PropertyBool", "FullDomainCFD", "CFD", "Create full CFD Domain").FullDomainCFD = False
        obj.addProperty("App::PropertyBool", "PeriodicDomainCFD", "CFD", "Create periodic CFD Domain").PeriodicDomainCFD = False
        obj.addProperty("App::PropertyFloat", "HalfD3toD2", "CFD", "Value of half relationship D3/D2").HalfD3toD2 = 1.2

    def execute (self, obj):
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")
        Blade = FreeCAD.ActiveDocument.getObject("ModelOfBlade3D")

        # Creation of the profile of streamlines:
        BladeFace = Blade.Shape.Faces[0]
        BladeSurface = BladeFace.Surface
        BladeEdgeShroud = BladeFace.Edges[0]
        BladeEdgeHub = BladeFace.Edges[2]

        BladeEdgeTe = BladeFace.Edges[1]

        R2 = Mer.D2/2.
               
        def thicknessProfile (ThicknessLE, ThicknessPoint1, ThicknessPoint2, ThicknessTE, BladeStreamlineEdge, BladeFace, BladeSurface, NameOfDisk, EdgeTe, LECoeff1, TECoeff1,TraillingEdgeEllipse, UVoutlet, Extend):	
            BladesEdgeDicretForLe = BladeStreamlineEdge.discretize(Distance = ThicknessLE/4.)
            BladesEdgeDicretForSide = BladeStreamlineEdge.discretize(Number = 20)

            BladesEdgeDicret = []
            for i in range (0, 10, 1):
                BladesEdgeDicret.append(BladesEdgeDicretForLe[i])

            for i in range (3, len(BladesEdgeDicretForSide), 1):
                BladesEdgeDicret.append(BladesEdgeDicretForSide[i])

            OutletVector = BladeFace.valueAt(Extend, UVoutlet)

            BladesEdgeDicret.append(OutletVector)

            # Creation of Thickness curve of the Shroud
            ThicknessPoints = [FreeCAD.Vector(0, ThicknessLE, 0), FreeCAD.Vector(40, ThicknessPoint1, 0), FreeCAD.Vector(75, ThicknessPoint2, 0), FreeCAD.Vector(100, ThicknessTE, 0)]
            ThicknessCurve = Part.BSplineCurve()
            ThicknessCurve.interpolate(ThicknessPoints)
            ThicknessCurveDiscret = []
            for i in range (0, len(BladesEdgeDicret), 1):
                vector_line1 = FreeCAD.Vector(float(i)/(float(len(BladesEdgeDicret))/100.), -180, 0)
                vector_line2 = FreeCAD.Vector(float(i)/(float(len(BladesEdgeDicret))/100.), 180, 0)
                line = Part.LineSegment(vector_line1, vector_line2)
                ThicknessIntersect = ThicknessCurve.intersectCC(line)
                ThicknessCurveDiscret.append(ThicknessIntersect)

                vu = []
            for i in range (0, len(BladesEdgeDicret), 1):
                vuVector = BladeSurface.parameter(BladesEdgeDicret[i])
                vu.append(vuVector)

                            

            normalPressure = []
            for i in range (0, len(vu), 1):
                normalVector = BladeFace.normalAt(vu[i][0], vu[i][1])
                normalPressure.append(normalVector.normalize())

            normalSuction = []
            for i in range (0, len(vu), 1):
                normalVector = BladeFace.normalAt(vu[i][0], vu[i][1])
                normalSuction.append(normalVector.normalize())

            if TraillingEdgeEllipse == False:

                pressureSide = []
                for i in range (2*LECoeff1, len(normalPressure), 1):
                    vectorPressureSide = normalPressure[i]
                    valueThickness = ThicknessCurveDiscret[i][0]
                    vectorPressureSide = vectorPressureSide.multiply(valueThickness.Y/2.)
                    vectorPressureSide = vectorPressureSide.add(BladesEdgeDicret[i])
                    pressureSide.append(vectorPressureSide)

                suctionSide = []
                for i in range (2*LECoeff1, len(normalSuction), 1):
                    vectorSuctionSide = normalSuction[i]
                    valueThickness = ThicknessCurveDiscret[i][0]
                    vectorSuctionSide = vectorSuctionSide.multiply(-valueThickness.Y/2.)
                    vectorSuctionSide = vectorSuctionSide.add(BladesEdgeDicret[i])
                    suctionSide.append(vectorSuctionSide)

            else:

                pressureSide = []
                for i in range (2*LECoeff1, len(normalPressure)-2*TECoeff1, 1):
                    vectorPressureSide = normalPressure[i]
                    valueThickness = ThicknessCurveDiscret[i][0]
                    vectorPressureSide = vectorPressureSide.multiply(valueThickness.Y/2.)
                    vectorPressureSide = vectorPressureSide.add(BladesEdgeDicret[i])
                    pressureSide.append(vectorPressureSide)

                suctionSide = []
                for i in range (2*LECoeff1, len(normalSuction)-2*TECoeff1, 1):
                    vectorSuctionSide = normalSuction[i]
                    valueThickness = ThicknessCurveDiscret[i][0]
                    vectorSuctionSide = vectorSuctionSide.multiply(-valueThickness.Y/2.)
                    vectorSuctionSide = vectorSuctionSide.add(BladesEdgeDicret[i])
                    suctionSide.append(vectorSuctionSide)


            # Points of the LE Shroud curve
            LePointsList = []

            for i in range (1, LECoeff1*2):
                lengthLePi = ThicknessLE/2*np.sqrt(1.-(i*ThicknessLE/4.)**2/(float(LECoeff1)/2.*ThicknessLE)**2)
                LePointsList.append(lengthLePi)
            LePointsList.reverse()

            LeCurvePressure = [BladesEdgeDicret[0]]
            for i in range (1, len(LePointsList)+1):
                vectorPressureLe = normalPressure[i]
                vectorPressureLe = vectorPressureLe.multiply(LePointsList[i-1])
                vectorPressureLe = vectorPressureLe.add(BladesEdgeDicret[i])
                LeCurvePressure.append(vectorPressureLe)
            LeCurvePressure.append(pressureSide[0])
            LeCurvePressure.reverse()

            LeCurveSuction = []
            for i in range (1, len(LePointsList)+1):
                vectorSuctionLe = normalSuction[i]
                vectorSuctionLe = vectorSuctionLe.multiply(-LePointsList[i-1])
                vectorSuctionLe = vectorSuctionLe.add(BladesEdgeDicret[i])
                LeCurveSuction.append(vectorSuctionLe)
            LeCurveSuction.append(suctionSide[0])
            
            Le = LeCurvePressure+LeCurveSuction

            # Line of the TE Shroud curve
            if TraillingEdgeEllipse == False:
            
                TeP1 = pressureSide[-1]
                TeP2 = suctionSide[-1]
                TeSpline = Part.LineSegment(TeP1, TeP2)
        
            else:
    ########################################## For realization ####################				
                TePointsList = []

                for i in range (1, TECoeff1*2):
                    lengthTePi = ThicknessTE/2*np.sqrt(1.-(i*ThicknessTE/4.)**2/(float(TECoeff1)/2.*ThicknessTE)**2)
                    TePointsList.append(lengthTePi)
                TePointsList.reverse()

                TeCurvePressure = [BladesEdgeDicret[-1]]
                for i in range (-2, -len(TePointsList)-2, -1):
                    vectorPressureTe = normalPressure[i]
                    vectorPressureTe = vectorPressureTe.multiply(TePointsList[-i-2])
                    vectorPressureTe = vectorPressureTe.add(BladesEdgeDicret[i])
                    TeCurvePressure.append(vectorPressureTe)
                TeCurvePressure.append(pressureSide[-1])
                TeCurvePressure.reverse()

                TeCurveSuction = []
                for i in range (-2, -len(TePointsList)-2, -1):
                    vectorSuctionTe = normalSuction[i]
                    vectorSuctionTe = vectorSuctionTe.multiply(-TePointsList[-i-2])
                    vectorSuctionTe = vectorSuctionTe.add(BladesEdgeDicret[i])
                    TeCurveSuction.append(vectorSuctionTe)
                TeCurveSuction.append(suctionSide[-1])
                
                Te = TeCurvePressure+TeCurveSuction
                TeSpline = Part.BSplineCurve()
                TeSpline.interpolate(Te)
    ################################################### THE END ######################

            pressureSide2 = Part.BSplineCurve()
            pressureSide2.interpolate(pressureSide)
            suctionSide2 = Part.BSplineCurve()
            suctionSide2.interpolate(suctionSide)

            pressureSide2Discr = pressureSide2.discretize(Number = 100)
            suctionSide2Discr = suctionSide2.discretize(Number = 100)



            e = Part.BSplineCurve()
            e.interpolate(pressureSide2Discr)
            e2 = Part.BSplineCurve()
            e2.interpolate(suctionSide2Discr)
            e3 = Part.BSplineCurve()
            e3.interpolate(Le)


            w = Part.Wire([e3.toShape(), e.toShape(), e2.toShape(), TeSpline.toShape()])

            return w


    ##########################################################################


        

        ShroudProfile = thicknessProfile(obj.ThicknessLEShroud, obj.ThicknessPoint1Shroud, obj.ThicknessPoint2Shroud, obj.ThicknessTEShroud, BladeEdgeShroud, BladeFace, BladeSurface, "Hub", BladeEdgeTe, obj.LeadingEdgeType, obj.TraillingEdgeType, obj.TraillingEdgeEllipse, 0.0, 1.05)
        HubProfile = thicknessProfile(obj.ThicknessLEHub, obj.ThicknessPoint1Hub, obj.ThicknessPoint2Hub, obj.ThicknessTEHub, BladeEdgeHub, BladeFace, BladeSurface, "Shroud", BladeEdgeTe, obj.LeadingEdgeType, obj.TraillingEdgeType, obj.TraillingEdgeEllipse, 1.0, 1.05)

        ### execute에 포함된 내용
        BladeEdgeShroudDiscret = BladeEdgeShroud.discretize(Distance = obj.ThicknessLEShroud/4.)

        uv = []
        for i in range (0, len(BladeEdgeShroudDiscret), 1):
            uv_value = BladeSurface.parameter(BladeEdgeShroudDiscret[i])
            uv.append(uv_value)

        tangentShroud = []
        for i in range (0, len(BladeEdgeShroudDiscret), 1):
            tangentVector = BladeFace.tangentAt(uv[i][0], uv[i][1])
            tangentShroud.append(tangentVector)

        curveShroud2 = []
        for i in range (0, len(BladeEdgeShroudDiscret), 1):
            vectorCurveShroud = tangentShroud[i][1]
            valueTranslate = obj.ThicknessLEShroud*1.5
            vectorCurveShroud = vectorCurveShroud.multiply(-valueTranslate)
            vectorCurveShroud = vectorCurveShroud.add(BladeEdgeShroudDiscret[i])
            curveShroud2.append(vectorCurveShroud)
        
        splineShroud2 = Part.BSplineCurve()
        splineShroud2.interpolate(curveShroud2)

        BladeEdgeHubDiscret = BladeEdgeHub.discretize(Distance = obj.ThicknessLEHub/4.)

        uvHub = []
        for i in range (0, len(BladeEdgeHubDiscret), 1):
            uv_valueHub = BladeSurface.parameter(BladeEdgeHubDiscret[i])
            uvHub.append(uv_valueHub)		

        tangentHub = []
        for i in range (0, len(BladeEdgeHubDiscret), 1):
            tangentVectorHub = BladeFace.tangentAt(uvHub[i][0], uvHub[i][1])
            tangentHub.append(tangentVectorHub)

        curveHub2 = []
        for i in range (0, len(BladeEdgeHubDiscret), 1):
            vectorCurveHub = tangentHub[i][1]
            valueTranslate = obj.ThicknessLEHub
            vectorCurveHub = vectorCurveHub.multiply(valueTranslate)
            vectorCurveHub = vectorCurveHub.add(BladeEdgeHubDiscret[i])
            curveHub2.append(vectorCurveHub)
        splineHub2 = Part.BSplineCurve()
        splineHub2.interpolate(curveHub2)
        BladeEdgeHubReverse = BladeEdgeHub.reverse()

        faceShroud2 = Part.makeRuledSurface(splineShroud2.toShape(), BladeEdgeShroud)
        faceHub2 = Part.makeRuledSurface(BladeEdgeHub, splineHub2.toShape())

        BladeEdgeTeShroud2 = faceShroud2.Edges[1]
        BladeEdgeTeHub2 = faceHub2.Edges[1] 

        ShroudProfile2 = thicknessProfile(obj.ThicknessLEShroud, obj.ThicknessPoint1Shroud, obj.ThicknessPoint2Shroud, obj.ThicknessTEShroud, splineShroud2, faceShroud2, faceShroud2.Surface, "Hub", BladeEdgeTeShroud2, obj.LeadingEdgeType, obj.TraillingEdgeType, obj.TraillingEdgeEllipse, 0.0, BladeEdgeShroud.Length*1.05)
        HubProfile2 = thicknessProfile(obj.ThicknessLEHub, obj.ThicknessPoint1Hub, obj.ThicknessPoint2Hub, obj.ThicknessTEHub, splineHub2, faceHub2, faceHub2.Surface, "Shroud", BladeEdgeTeHub2, obj.LeadingEdgeType, obj.TraillingEdgeType, obj.TraillingEdgeEllipse, 1.0, BladeEdgeHub.Length*1.05)
        
        BladeSurfaceModel = Part.makeLoft([ShroudProfile2, HubProfile2])

        if obj.TraillingEdgeEllipse == False:

            ListEdges1 = [BladeSurfaceModel.Edges[0], BladeSurfaceModel.Edges[4], BladeSurfaceModel.Edges[7], BladeSurfaceModel.Edges[10]]
            Surface1 = Part.makeFilledFace(ListEdges1)

            ListEdges2 = [BladeSurfaceModel.Edges[2], BladeSurfaceModel.Edges[6], BladeSurfaceModel.Edges[9], BladeSurfaceModel.Edges[11]]
            Surface2 = Part.makeFilledFace(ListEdges2)
            
            ListSurfaces = Part.Compound([BladeSurfaceModel.Faces[0], BladeSurfaceModel.Faces[1], BladeSurfaceModel.Faces[2], BladeSurfaceModel.Faces[3], Surface1, Surface2])
            ListSurfaces.removeSplitter()
            BladeShell = Part.makeShell(ListSurfaces.Faces)
            BladeShell.removeSplitter()
            BladeSolidInit = Part.makeSolid(BladeShell)

        # Cut of the TE 

        FaceShroudCut =Mer.Shape.Faces[0]

        Shroudline = FaceShroudCut.Edges[2]
        ShroudVertex1 = Shroudline.Vertexes[0]
        ShroudVertex2 = Shroudline.Vertexes[1]

        Spnt1 = ShroudVertex1.Point
        Spnt2 = ShroudVertex2.Point
        Spnt3 = FreeCAD.Vector(Spnt1[0],Spnt2[1],0)

        LineShroudBlock = Part.LineSegment(FreeCAD.Vector(0, Spnt2[1],0), FreeCAD.Vector(2*Spnt2[0], Spnt2[1],0))
        LineHubBlock = Part.LineSegment(FreeCAD.Vector(0, Spnt2[1]*1.5,0), FreeCAD.Vector(2*Spnt2[0], Spnt2[1]*1.5,0))
        LineTopBlock = Part.LineSegment(FreeCAD.Vector(0, Spnt2[1],0), FreeCAD.Vector(0, Spnt2[1]*1.5,0))
        LineDownBlock = Part.LineSegment(FreeCAD.Vector(2*Spnt2[0], Spnt2[1]*1.5,0), FreeCAD.Vector(2*Spnt2[0], Spnt2[1],0))

        FaceShroudBlock = LineShroudBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceHubBlock = LineHubBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceTopBlock = LineTopBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceDownBlock = LineDownBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        BlockShell = Part.makeShell([FaceShroudBlock, FaceHubBlock, FaceTopBlock, FaceDownBlock])
        BlockSolid = Part.makeSolid(BlockShell)

        BladeSolidSecond = BladeSolidInit.cut(BlockSolid)


        if obj.FullDomainCFD == False and obj.PeriodicDomainCFD == False:
            #Cut of the Shroud Side of Blade 
            R2 = Mer.D2/2.
            rs = Mer.ds/2.
            t = Mer.ThicknessHub

            #Cut of the Shroud Side of Blade 
            LineTopShroudCut = Part.LineSegment(Spnt2,Spnt3)
            FaceTopShroudCut = LineTopShroudCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            LineInletShroudCut = Part.LineSegment(Spnt3, Spnt1)
            FaceInletShroudCut = LineInletShroudCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            ShroudCutBLockShell = Part.makeShell([FaceShroudCut, FaceTopShroudCut, FaceInletShroudCut])
            ShroudCutBLockSolid = Part.makeSolid(ShroudCutBLockShell)

            BladeSolidThree = BladeSolidSecond.cut(ShroudCutBLockSolid)

            # Cut of the Hub Side Of Blade

            FaceHubCut = Mer.Shape.Faces[1]#Hub face

            Hubline = FaceHubCut.Edges[2]
            HubVertex2 = Hubline.Vertexes[0]
            HubVertex3 = Hubline.Vertexes[1]

            Hpnt22 = HubVertex2.Point
            Hpnt3 = HubVertex3.Point

            Hpnt1 = FreeCAD.Vector(Hpnt22[0],rs, 0)
            Hpnt2 = FreeCAD.Vector(Hpnt22[0],(Hpnt22[1]+rs)/2., 0)
            Hpnt4 = FreeCAD.Vector(Hpnt3[0]+t/2., Hpnt3[1], 0)
            Hpnt5 = FreeCAD.Vector(Hpnt3[0]+t/2., (Hpnt22[1]+rs)/2., 0)
            Hpnt6 = FreeCAD.Vector(Hpnt3[0]+R2, Hpnt3[1], 0)
            Hpnt7 = FreeCAD.Vector(Hpnt3[0]+R2, rs, 0)

            # Cut of the Hub Side Of Blade
            Cut1 = Part.LineSegment(Hpnt1,Hpnt2)
            Rot1 = Cut1.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut2 = Part.LineSegment(Hpnt2,Hpnt5)
            Rot2 = Cut2.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut3 = Part.LineSegment(Hpnt5,Hpnt4)
            Rot3 = Cut3.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut4 = Part.LineSegment(Hpnt4,Hpnt6)
            Rot4 = Cut4.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut5 = Part.LineSegment(Hpnt6,Hpnt7)
            Rot5 = Cut5.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut6 = Part.LineSegment(Hpnt7,Hpnt1)
            Rot6 = Cut6.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            HubCutBLockShell = Part.makeShell([Rot1, Rot2, Rot3, Rot4, Rot5,Rot6])
            HubCutBLockSolid = Part.makeSolid(HubCutBLockShell)	

            BladeSolid = BladeSolidThree.cut(HubCutBLockSolid)	

            # Creation of a massive of blades
            AngleRotateBlade = 360./float(obj.NumberOfBlades)
           
            BladesList = []
            for i in range (0, obj.NumberOfBlades, 1):
                BladeSolidi = BladeSolid.copy()
                BladeSolidi.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), 0.5*AngleRotateBlade + AngleRotateBlade*(i))
                BladesList.append(BladeSolidi)

        elif obj.FullDomainCFD==True and obj.PeriodicDomainCFD == False:### Full Domain CFD 형상
            print('a')
        elif obj.FullDomainCFD==False and obj.PeriodicDomainCFD == True:###Periodic Domain CFD 형상
            print('b')

        obj.Shape = Part.Compound(BladesList)

class Hub:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyFloat", "BezierPoint1", "Hub profile", "Length of Hub").BezierPoint1 = 35
        obj.addProperty("App::PropertyFloat", "BezierPoint2", "Hub profile", "Length of Hub").BezierPoint2 = -5
        obj.addProperty("App::PropertyFloat", "BezierPoint3", "Hub profile", "Length of Hub").BezierPoint3 = 0
        obj.addProperty("App::PropertyFloat", "BezierPoint4", "Hub profile", "Length of Hub").BezierPoint4 = 10

    def execute (self, obj):
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")

        rs = Mer.ds/2.
        t = Mer.ThicknessHub

        FaceHubCut = Mer.Shape.Faces[1]
        Hubline = FaceHubCut.Edges[2]

        Vertex2 = Hubline.Vertexes[0]
        Vertex3 = Hubline.Vertexes[1]

        pnt2 = Vertex2.Point
        pnt3 = Vertex3.Point

        pnt1 = FreeCAD.Vector(0,rs,0)
        pnt2 = Vertex2.Point
        pnt3 = Vertex3.Point
        pnt4 = FreeCAD.Vector(pnt3[0]+t,pnt3[1],0)
        pnt5 = FreeCAD.Vector(pnt4[0]+obj.BezierPoint4,pnt4[1]-55,0)
        pnt6 = FreeCAD.Vector(pnt4[0]+obj.BezierPoint3,pnt4[1]-95,0)
        pnt7 = FreeCAD.Vector(pnt4[0]+obj.BezierPoint2,pnt4[1]-135,0)
        pnt8 = FreeCAD.Vector(pnt4[0]+obj.BezierPoint1,pnt4[1]-175,0)
        pnt9 = FreeCAD.Vector(pnt4[0]+obj.BezierPoint1,rs,0)

        BezierP = [pnt4, pnt5, pnt6, pnt7,pnt8]


        inletEdge = Part.LineSegment(pnt1,pnt2)
        FaceinletEdge = inletEdge.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        thicknessEdge = Part.LineSegment(pnt3,pnt4)
        FacethicknessEdge = thicknessEdge.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360) 
		
        Bezierline = Part.BezierCurve()
        Bezierline.setPoles(BezierP)
        FaceBezierline = Bezierline.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0),360)

        finalEdge = Part.LineSegment(pnt8,pnt9)
        FacefinalEdge = finalEdge.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        shaftEdge = Part.LineSegment(pnt9,pnt1)
        FaceshaftEdge = shaftEdge.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        HubShell = Part.makeShell([FaceHubCut, FaceinletEdge, FacethicknessEdge, FaceBezierline, FacefinalEdge, FaceshaftEdge])
        HubSolid = Part.makeSolid(HubShell)

        obj.Shape = HubSolid


class DV:
    def __init__(self, nameList, domainLB, domainUB, ptsNum, includeEdge):
        self.nameList = nameList
        self.domainLB = domainLB
        self.domainUB = domainUB
        self.scalingFactor = self.domainUB-self.domainLB
        self.interval = self.scalingFactor/ptsNum 
    
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

def printobjDVList():
    str_list = []
    for i in range(len(sampleListObj)):
        mapped_line = sampleListObj[i]
        str_line = []
        for j in range(ndv):
            str_line.append(format(mapped_line[j],'>15,.8E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('objdvList.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

def printconstrDVList():
    str_list = []
    for i in range(len(sampleListConstr)):
        mapped_line = sampleListConstr[i]
        str_line = []
        for j in range(ndv):
            str_line.append(format(mapped_line[j],'>15,.8E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('constrdvList.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

def printConstrList():
    str_list = []
    for i in range(len(constraintFuncValue)):
        constr_line = constraintFuncValue[i]
        str_line = []
        str_line.append(format(constr_line,'>15,.8E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('constrList.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

def printObjList():
    str_list = []
    for i in range(len(objectiveFuncValue)):
        obj_line = objectiveFuncValue[i]
        str_line = []
        str_line.append(format(obj_line,'>15,.8E'))
        str_line2 = ''.join(str_line)+'\n'
        str_list.append(str_line2)
    str_list[-1]=str_list[-1][:-1]

    with open('objList.txt','w',encoding='UTF-8') as f:
        for value in str_list:
            f.write(value)

def matlabObjective():
    string = []
    string.append('function [obj] = objective(value)\n\n')
    string.append('global value_his;\n')
    string.append('global obj_his;\n')
    string.append('obj = ')
    string.append(str(translatePredictionToMatlab(["value(1)", "value(2)", "value(3)", "value(4)"], np.array(objectiveFuncValue), krigObj)))
    string.append(';\n\n')
    string.append('value_his = [value_his; value];\n')
    string.append('obj_his=[obj_his;obj];\n')
    string.append('end')

    with open('objective.m','w',encoding='UTF-8') as f:
        for line in string:
            f.write(line)


def matlabConstraint():
    string = []
    string.append('function [c,ceq]=nonlcon(value)\n\n')
    string.append('global von_his;\n')
    string.append('von = ')
    string.append(str(translatePredictionToMatlab(["value(1)", "value(2)", "value(3)", "value(4)"], np.array(constraintFuncValue), krigConstr)))
    string.append(';\n\n')
    string.append('c= von-503;\n')
    string.append('ceq=[];\n\n')
    string.append('von_his = [von_his;von];\n\n')
    string.append('end')

    with open('nonlcon.m','w',encoding='UTF-8') as f:
        for line in string:
            f.write(line)
        f.close

# Plots Kriging prediction in 3D surface plot. Select two design variables from DVList with dvIndex1, dvIndex2
# and then assign values for another design variables with anotherDVs
def plotPrediction(DVList, dvIndex1, dvIndex2, anotherDVs, funcName, div, sampleValueVector, k):
    if dvIndex1 > dvIndex2:
        dvIndex1, dvIndex2 = dvIndex2, dvIndex1

    if len(anotherDVs)+2 == len(DVList):
        dv1 = np.linspace(DVList[dvIndex1].domainLB, DVList[dvIndex1].domainUB, div).tolist()
        dv2 = np.linspace(DVList[dvIndex2].domainLB, DVList[dvIndex2].domainUB, div).tolist()
        
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
        #analytic = 1.2*np.power(10, 9)/np.power(dv2,2)/dv1

        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection='3d')
        myCmap = plt.get_cmap('plasma')

        surf = ax.plot_surface(dv1, dv2, pred, cmap=myCmap, edgecolor='none')
        #surfAnalytic = ax.plot_wireframe(dv1, dv2, analytic, cmap='seismic')
        scatter = ax.scatter(samplePts[0], samplePts[1], sampleValueVector, marker='x', s=100, c='black')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        if not funcName[0].isupper():
            charList = list(map(lambda c: c, funcName))
            charList[0] = charList[0].upper()
            funcName = ''.join(charList)

        ax.set_title("Kriging Prediction of "+str(funcName))

        plt.show()

    else:
        return


if __name__ == "__main__":
    FCPath = #@FCPath Placeholder

    try:
        FreeCADGui.showMainWindow()
        FreeCADGui.updateGui()

        with open(FCPath+u'\\data\\Mod\\Start\\StartPage\\LoadNew.py') as file:
            exec(file.read())

        initialNameValueList = readNameValue()

        myObj = FreeCAD.ActiveDocument.addObject("Part::FeaturePython", "Meridional") 
        Meridional(myObj)
        if(bindProperties(myObj, initialNameValueList[0], initialNameValueList)):
            myObj.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")

        myObj2 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'ModelOfBlade3D')
        ModelOfBlade3D(myObj2)
        if(bindProperties(myObj2, initialNameValueList[0], initialNameValueList)):
            myObj2.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too
        Blade = FreeCAD.ActiveDocument.getObject("ModelOfBlade3D")

        myObj3 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'Blades')
        Blades(myObj3)
        if(bindProperties(myObj3, initialNameValueList[0], initialNameValueList)):
            myObj3.ViewObject.Proxy = 0 # this is mand atory unless we code the ViewProvider too
        Bla = FreeCAD.ActiveDocument.getObject("Blades")
        
        myObj4 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'Hub')
        Hub(myObj4)
        myObj4.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too

        FreeCAD.ActiveDocument.recompute()


        path_os = os.getcwd()
        path_gui = re.sub(r'\\', '/',path_os)


        # Design Variable Grouping
        m = 30
        ndv = 4
        includeEdge = 0
        bPoint1 = DV(["BezierPoint1"],0,70,m,includeEdge)
        bPoint2 = DV(["BezierPoint2"],-70,60,m,includeEdge)
        bPoint3 = DV(["BezierPoint3"],-40,40,m,includeEdge)
        bPoint4 = DV(["BezierPoint4"],-10,30,m,includeEdge)

        DVList = [bPoint1,bPoint2,bPoint3,bPoint4]
        if len(DVList) != ndv:
            print("DVList length is not equal to given number of design variables")
            exit()


        print("\n\n")
        # Optimal Latin Hypercube Sampling with pyKriging Module.
        samplingPlan = samplingplan(ndv)
        normalizedSampleList = samplingPlan.optimallhc(m)
        print("\nOptimal LHS Result(normalized):\n", normalizedSampleList)


        # Revert the normalized sample list to original domain
        def denormalizePtsList(DVList, normalizedSampleList):
            normalizedSampleList = normalizedSampleList.tolist()
            sampleList = np.zeros((len(normalizedSampleList), len(DVList)))
            for i in range(len(normalizedSampleList)):
                sampleList[i] = list(map(lambda x: x.domainLB+x.scalingFactor*normalizedSampleList[i][DVList.index(x)], DVList))
       
            return sampleList.tolist()
        
        sampleList = denormalizePtsList(DVList, normalizedSampleList)
        print("\nOptimal LHS Result:\n", sampleList)

        # Experiment with points calculated from OLHS
        def experiment(DVList, denormalizedSampleList, expNum=None):
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
               
                for i in range(len(DVList)):
                    DVList[i].bindValue(exp[i])  
                    
                for obj in FreeCAD.ActiveDocument.Objects:
                    obj.touch()
                FreeCAD.ActiveDocument.recompute()
                print("Done recomputing")

                if Bla.FullDomainCFD == False and Bla.PeriodicDomainCFD == False:
                    if "Fusion" in list(map(lambda obj: obj.Name, FreeCAD.ActiveDocument.Objects)):
                        FreeCAD.ActiveDocument.removeObject("Fusion")

                    Fusion = FreeCAD.ActiveDocument.addObject("Part::MultiFuse","Fusion")
                    FreeCAD.ActiveDocument.Fusion.Shapes = [FreeCAD.ActiveDocument.Blades,FreeCAD.ActiveDocument.Hub]
                    FreeCAD.ActiveDocument.recompute()
                    
                    __objs__=[]
                    __objs__.append(FreeCAD.getDocument("Unnamed").getObject("Fusion"))
                    Part.export(__objs__,path_gui+u"/impeller.step")
                    del __objs__
                else:
                    __objs__=[]
                    __objs__.append(FreeCAD.getDocument("Unnamed").getObject("Blades"))
                    Part.export(__objs__,path_gui+u"/impeller.step")
                    del __objs__ 
                print("Done exporting step file")

                preprocExec = subprocess.Popen([sys.executable, os.getcwd()+u"\\preprocess.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)         
                preprocSTDOUT, preprocSTDERR = preprocExec.communicate() 
                #print("!"+preprocSTDOUT.decode('UTF-8')+"\n!!"+preprocSTDERR.decode('UTF-8'))
                print("Done preprocessing")

                ccxDir = os.getcwd()+u"\\ccx\\etc"
                femExec = subprocess.Popen([ccxDir+u"\\runCCXnoCLS.bat", os.getcwd()+u"\\Static_analysis.inp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                femSTDOUT, femSTDERR = femExec.communicate()
                #print("!"+femSTDOUT.decode('UTF-8')+"\n!!"+femSTDERR.decode('UTF-8'))
                print("Done running ccx")               

                postprocExec = subprocess.Popen([sys.executable, os.getcwd()+u"\\postprocess.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                postprocSTDOUT, postprocSTDERR = postprocExec.communicate()
                os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\caseSaver.ps1 "+str(exp[0])+"_"+str(exp[1])+"_"+str(exp[2])+"_"+str(exp[3]))
                #print("!"+postprocSTDOUT+"\n!!"+postprocSTDERR)
                print("Done postprocessing") 

                objectiveFuncValue.append(FreeCAD.ActiveDocument.Fusion.Shape.Volume/np.power(10, 6))
                constraintFuncValue.append(float(postprocSTDOUT.decode('UTF-8').strip('\r\n'))/100)
               
                print(fg.green+"Objective function value (dm^3): "+str(objectiveFuncValue[-1])+fg.rs)
                print(fg.green+"Constraint function value : "+str(constraintFuncValue[-1])+"x 100MPa"+"\n\n"+fg.rs)

            return objectiveFuncValue, constraintFuncValue

        objectiveFuncValue, constraintFuncValue = experiment(DVList, sampleList)


        # Outlier Detection with MCD
        def detectOutliersMCD(sampleList, funcValue, infill=False, samplePtsBeforeInfill=None):
            yObserved = np.array(funcValue)
            X = np.array(sampleList)
            
            augX = np.append(X, np.reshape(yObserved, (yObserved.shape[0],1)), axis=1)
            #print("augX: ", augX)

            outlierDetection = EllipticEnvelope(contamination=0.005)
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
            
            return mask

        # Simple median-absoulte-deviation(MAD) based 1D outlier detection for checkup
        # Returns whether an observation is an outlier or not: True if outlier
        def detectOutliersMAD(funcValue, threshold=3.5):
            median = np.median(funcValue, axis=0)
            diff = np.abs(funcValue - median*np.ones(funcValue.shape[0]))
            mad = np.median(diff, axis=0)

            print("mad: ", mad)
            if mad == 0:
                return median
            else:
                modifiedZScore = 0.6745*diff/mad
                return np.array(list(map(lambda Z: Z < threshold, modifiedZScore)))
            

        # Redo experiments on detected outlier points
        def verifyOutliers(DVList, sampleList, funcIndex, funcValue, outlierMask, expRep):
            testObservations = []
            outlierMask = outlierMask.tolist()

            if len(sampleList) == len(funcValue) == len(outlierMask):
                for i in range(len(outlierMask)):
                    if not outlierMask[i]:
                        print("Re-doing experiment for detected outlier point: ", sampleList[i])

                        for j in range(expRep):
                            testObservations.append(*experiment(DVList, [sampleList[i]], j+1)[funcIndex])
                        testObservations = np.array(testObservations)
                        print("Initial checkup observations: ", testObservations)
                        
                        checkupMask = detectOutliersMAD(testObservations)

                        # MAD is zero: checkupMask is median of testObservations
                        if len(checkupMask.shape) == 0:
                            if np.abs(1-checkupMask/funcValue[i]) < 0.01:
                                outlierMask[i] = True
                        
                        # MAD is nonzero: checkUpMask is verified with MAD
                        else: 
                            testObservations = testObservations[checkupMask]

                            print("Outlier removed checkup observations: ", testObservations)
                            testObservations = np.append(testObservations, [funcValue[i]], axis=0)
                            checkupMask = detectOutliersMAD(testObservations)

                            if checkupMask[-1]:
                                print("Prior experiment result seems ok\n")
                                outlierMask[i] = True

            return outlierMask
        
        outlierMaskObj = detectOutliersMCD(sampleList, objectiveFuncValue)
        outlierMaskObj = verifyOutliers(DVList, sampleList, 0, objectiveFuncValue, outlierMaskObj, 3)

        outlierMaskConstr = detectOutliersMCD(sampleList, constraintFuncValue)
        outlierMaskConstr = verifyOutliers(DVList, sampleList, 1, constraintFuncValue, outlierMaskConstr, 3)
        
        print("Rearranged outlier mask of objective func.: ", outlierMaskObj)
        print("Rearranged outlier mask of constraint func.: ", outlierMaskConstr)

        sampleListObjBeforeInfill, objectiveFuncValueBeforeInfill = np.array(sampleList)[np.array(outlierMaskObj)].tolist(), np.array(objectiveFuncValue)[np.array(outlierMaskObj)].tolist()
        sampleListConstrBeforeInfill, constraintFuncValueBeforeInfill = np.array(sampleList)[np.array(outlierMaskConstr)].tolist(), np.array(constraintFuncValue)[np.array(outlierMaskConstr)].tolist()
                
                    
        # Kriging the result
        print("Training for objective function Kriging prediction\n")
        krigObj = kriging(np.array(sampleListObjBeforeInfill), np.array(objectiveFuncValueBeforeInfill))
        krigObj.train()

        print("Training for constraint function Kriging prediction\n")
        krigConstr = kriging(np.array(sampleListConstrBeforeInfill), np.array(constraintFuncValueBeforeInfill))
        krigConstr.train()


        # Infill points for Kriging + outlier detection with MCD
        
        def doInfill(DVList, sampleList, funcValue, funcIndex, krig, infillRep, infillNum):
            print("\nExperiments for Infill Points: ")
            for i in range(infillRep):
                # infill criteria are either 'ei' (expected improvement) or 'error' (point of biggest error)
                # and also, remember to set addPoint=False so we can decide whether a infill point and its observation is outlier.
                newPts = krig.infill(infillNum, method='ei', addPoint=False).tolist()

                sampleNumBeforehand = len(sampleList)
                sampleList += newPts
                for j in range(len(newPts)):
                    funcValue.append(*experiment(DVList, [newPts[j]], infillNum*i+j+1)[funcIndex])

                outlierMask = verifyOutliers(DVList, sampleList, funcIndex, funcValue, detectOutliersMCD(sampleList, funcValue, infill=True, samplePtsBeforeInfill=sampleNumBeforehand), 5)
                sampleList, funcValue = np.array(sampleList)[np.array(outlierMask)].tolist(), np.array(funcValue)[np.array(outlierMask)].tolist()
                
                print("outlier removed: ", sampleList, funcValue) 
                for j in range(sampleNumBeforehand, len(sampleList), 1):
                    krig.addPoint(np.array(sampleList[j]), np.array(funcValue[j]))
                    krig.train()
                            
                print("sampleList: ", sampleList, "\n")
                print("constFuncValue["+str(funcIndex)+"]: ", funcValue, "\n")
                print("outlierMask: ", outlierMask, "\n")
                print("k.X length: ", krig.X.shape[0], "\n")
                print("k.y length: ", krig.y.shape[0], "\n")

            return sampleList, funcValue, krig
       
        print("\nObjective function infill: \n")
        sampleListObj, objectiveFuncValue, krigObj = doInfill(DVList, sampleListObjBeforeInfill, objectiveFuncValueBeforeInfill, 0, krigObj, 10, 1)
        print("\nConstraint function infill: \n")
        sampleListConstr, constraintFuncValue, krigConstr = doInfill(DVList, sampleListConstrBeforeInfill, constraintFuncValueBeforeInfill, 1, krigConstr, 10, 1)
    

        # Print the result and result assessment
        predictedObjectiveResult = translatePrediction(np.array(sampleListObj), np.array(objectiveFuncValue), krigObj)
        predictedConstraintResult = translatePrediction(np.array(sampleListConstr), np.array(constraintFuncValue), krigConstr)

        print("\n\nKriging result of objective function - Volume of impeller: \n"+str(translatePredictionToMatlab(["z1", "z2", "z3", "z4"], np.array(objectiveFuncValue), krigObj)))
        print("\n\nObjective function values acquired with FreeCAD: \n"+str(objectiveFuncValue))
        print("\n\nObjective function values acquired with Kriging: \n"+str(predictedObjectiveResult))
        print("\n\nObject. func. R^2 = "+str(krigObj.rsquared(np.array(objectiveFuncValue), np.array(predictedObjectiveResult))))
 
        print("\n\nKriging result of constraint function - Max. von Mises Stress: \n"+str(translatePredictionToMatlab(["z1", "z2", "z3", "z4"], np.array(constraintFuncValue), krigConstr)))
        print("\n\nConstraint function values acquired with FEA: \n"+str(constraintFuncValue))
        print("\n\nConstraint function values estimated with Kriging: \n"+str(predictedConstraintResult))
        print("\n\nConstr. func. R^2 = "+str(krigConstr.rsquared(np.array(constraintFuncValue), np.array(predictedConstraintResult))))



        printobjDVList()
        printconstrDVList()
        printObjList()
        printConstrList()
        matlabObjective()
        matlabConstraint()
        plotPrediction(DVList, 1, 2, [70], "Volume", 100, np.array(objectiveFuncValue), krigObj)
        plotPrediction(DVList, 1, 2, [70], "maxVonMises", 100, np.array(constraintFuncValue), krigConstr)

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

