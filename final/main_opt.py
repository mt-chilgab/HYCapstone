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
        obj.addProperty("App::PropertyFloat", "R", "Dimensions", "Radius of Hub Circle")
        obj.addProperty("App::PropertyVector", "HubCenter", "Center of Geometry", "Point of a Hub Circle Center")
        obj.addProperty("App::PropertyFloat", "a", "Dimensions", "Major Radius of Shroud Ellipse")
        obj.addProperty("App::PropertyFloat", "b", "Dimensions", "Minor Radius of Shroud Ellipse")
        obj.addProperty("App::PropertyVector", "EllipseCenter", "Center of Geometry", "Point of a Shroud Ellipse Center")
        obj.addProperty("App::PropertyFloat", "D2", "Dimensions", "Diameter of the impeller")
        obj.addProperty("App::PropertyFloat", "ds", "Dimensions", "Diameter of the shaft")
        obj.addProperty("App::PropertyFloat", "ThicknessHub", "Dimensions", "Thickness of hub")

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
        obj.addProperty("App::PropertyFloat", "AngleBeta1Shroud", "Angles of a Shroud streamline", "Inlet angle")
        obj.addProperty("App::PropertyFloat", "AngleBeta2Shroud", "Angles of a Shroud streamline", "Outlet angle")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint3Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyBool", "AnglesChartShroud", "Angles of a Shroud streamline", "It is shows chart angle Beta-Streamline")

         
        obj.addProperty("App::PropertyFloat", "AngleBeta1Hub", "Angles of a Hub streamline", "Inlet angle")
        obj.addProperty("App::PropertyFloat", "AngleBeta2Hub", "Angles of a Hub streamline", "Outlet angle")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleBetaPoint3Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyBool", "AnglesChartHub", "Angles of a Hub streamline", "It is shows chart angle Beta-Streamline")

        obj.addProperty("App::PropertyFloat", "AngleAlpha1", "Relative theta delay between hub and shroud streamline", "Inlet angle")
        obj.addProperty("App::PropertyFloat", "AngleAlpha2", "Relative theta delay between hub and shroud streamline", "Outlet angle")
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint1", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint2", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active")
        obj.addProperty("App::PropertyVector", "AngleAlphaPoint3", "Relative theta delay between hub and shroud streamline", "Point of a B-spline, z-coodrinate non active")
 
        obj.addProperty("App::PropertyInteger", "N", "Number of the calculation points")
 
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
        obj.addProperty("App::PropertyFloat", "ThicknessLEShroud", "Shroud profile", "Value of the LE")
        obj.addProperty("App::PropertyFloat", "ThicknessPoint1Shroud", "Shroud profile", "Point for B-Spline thickness")
        obj.addProperty("App::PropertyFloat", "ThicknessPoint2Shroud", "Shroud profile", "Point for B-Spline thickness")
        obj.addProperty("App::PropertyFloat", "ThicknessTEShroud", "Shroud profile", "Value of the TE")

        obj.addProperty("App::PropertyFloat", "ThicknessLEHub", "Hub profile", "Value of the thickness LE")
        obj.addProperty("App::PropertyFloat", "ThicknessPoint1Hub", "Hub profile", "Point for B-Spline thickness")
        obj.addProperty("App::PropertyFloat", "ThicknessPoint2Hub", "Hub profile", "Point for B-Spline thickness")
        obj.addProperty("App::PropertyFloat", "ThicknessTEHub", "Hub profile", "Value of the thickness TE")

        obj.addProperty("App::PropertyBool", "TraillingEdgeEllipse", "Type of the LE and TE", "Type of the trailling edge")
        obj.addProperty("App::PropertyInteger", "LeadingEdgeType", "Type of the LE and TE", "Type of the leading edge")
        obj.addProperty("App::PropertyInteger", "TraillingEdgeType", "Type of the LE and TE", "Type of the trailling edge" )
        obj.addProperty("App::PropertyInteger", "NumberOfBlades", "Number of blades")

        obj.addProperty("App::PropertyBool", "FullDomainCFD", "CFD", "Create full CFD Domain")
        obj.addProperty("App::PropertyBool", "PeriodicDomainCFD", "CFD", "Create periodic CFD Domain")
        obj.addProperty("App::PropertyFloat", "HalfD3toD2", "CFD", "Value of half relationship D3/D2")

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
        obj.addProperty("App::PropertyFloat", "BezierPoint1", "Hub profile", "Length of Hub")
        obj.addProperty("App::PropertyFloat", "BezierPoint2", "Hub profile", "Length of Hub")
        obj.addProperty("App::PropertyFloat", "BezierPoint3", "Hub profile", "Length of Hub")
        obj.addProperty("App::PropertyFloat", "BezierPoint4", "Hub profile", "Length of Hub")

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


if __name__ == "__main__":

    FCPath = "C:\\Program Files\\FreeCAD 0.19"

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
        if(bindProperties(myObj4, initialNameValueList[0], initialNameValueList)):
            myObj4.ViewObject.Proxy = 0 # this is mand atory unless we code the ViewProvider too
        myObj4.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too

        FreeCAD.ActiveDocument.recompute()

        path_os = os.getcwd()
        path_gui = re.sub(r'\\', '/',path_os)


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
        os.system(u"powershell.exe "+os.getcwd()+u"\\scripts\\caseSaver.ps1 optCase")
        #print("!"+postprocSTDOUT+"\n!!"+postprocSTDERR)
        print("Done postprocessing") 

        objectiveFuncValue = FreeCAD.ActiveDocument.Fusion.Shape.Volume/np.power(10, 6)
        constraintFuncValue = float(postprocSTDOUT.decode('UTF-8').strip('\r\n'))/100

        print(fg.green+"Objective function value (dm^3): "+str(objectiveFuncValue)+fg.rs)
        print(fg.green+"Constraint function value : "+str(constraintFuncValue)+"x 100MPa"+"\n\n"+fg.rs)

    except Exception as e:
        logging.info(e)

    exit()

