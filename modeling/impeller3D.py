# Modules for property update check
import re
import time
import itertools
import sys, os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import FreeCAD, FreeCADGui, Part



class Meridional:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyFloat", "ds", "Dimensions", "Diameterof a shaft")#.ds
        obj.addProperty("App::PropertyFloat", "D2", "Dimensions", "Diameter of an impeller")#.D2=1079
        obj.addProperty("App::PropertyFloat", "d0", "Dimensions", "Diameter of a hub")#.d0=100
        obj.addProperty("App::PropertyFloat", "D0", "Dimensions", "Suction diameter")#.D0=850
        obj.addProperty("App::PropertyFloat", "b2", "Dimensions", "Width of a outlet")#.b2=250
        obj.addProperty("App::PropertyFloat", "L", "Dimensions", "Length")#.L=300
        obj.addProperty("App::PropertyBool", "CylindricalBlades", "Type of blades", "Number of streamlines")#.CylindricalBlades=False
        obj.addProperty("App::PropertyPercent", "LePositionShroud", "Place of a Leading edge, %")#.LePositionShroud#=15
        obj.addProperty("App::PropertyPercent", "LePositionHub", "Place of a Leading edge, %")#.LePositionHub#=55
        obj.addProperty("App::PropertyPercent", "LePositionAverage", "Place of a Leading edge, %")#.LePositionAverage#=40

    def execute (self, obj):
        import Part, FreeCAD, DraftTools, Draft
        
        R2 = obj.D2/2.
        r0 = obj.d0/2.
        R0 = obj.D0/2.

        # Outer shroud points for cubic Bezier
        shroudP1 = FreeCAD.Vector(0, R0,0)      # (0, 425, 0)
        shroudP2 = FreeCAD.Vector(obj.L,R0,0)   # (300, 425, 0)
        shroudP3 = FreeCAD.Vector(obj.L,(R2+R0)/2.,0)   # (300, 480, 0)
        shroudP4 = FreeCAD.Vector(obj.L,R2,0)   # (300, 535, 0)

        # Inner hub points for cubic Bezier
        hubP1 = FreeCAD.Vector(0,r0,0)  # (0, 50, 0) 
        hubP2 = FreeCAD.Vector(obj.L+obj.b2,r0,0)    # (550, 50, 0)
        hubP3 = FreeCAD.Vector(obj.L+obj.b2,(R2+R0)/2,0)    # (550, 
        hubP4 = FreeCAD.Vector(obj.L+obj.b2, R2,0)

        # Mid points of shrould / hub generation curve
        aveCurveP1 = FreeCAD.Vector(0, (R0+r0)/2., 0)
        aveCurveP2 = FreeCAD.Vector((obj.L+obj.b2+obj.L)/2.,(R0+r0)/2., 0)
        aveCurveP3 = FreeCAD.Vector((obj.L+obj.b2+obj.L)/2., (R2+R0)/2., 0)
        aveCurveP4 = FreeCAD.Vector((obj.L+obj.b2+obj.L)/2., R2, 0)

        shroudP = [shroudP1, shroudP2, shroudP3, shroudP4]
        hubP = [hubP1, hubP2, hubP3, hubP4]
        aveCurveP = [aveCurveP1, aveCurveP2, aveCurveP3, aveCurveP4]
        
        shroud = Part.BezierCurve()
        shroud.setPoles(shroudP)
        shroud.toShape()
        shroudDiscret = shroud.discretize(Number = 100)

        hub = Part.BezierCurve()
        hub.setPoles(hubP)
        hub.toShape()
        hubDiscret = hub.discretize(Number = 100)			

        aveCurve = Part.BezierCurve()
        aveCurve.setPoles(aveCurveP)
        aveCurve.toShape()
        aveCurveDiscret = aveCurve.discretize(Number = 100)

        inlet1 = Part.LineSegment(shroudP1,FreeCAD.Vector(0, (R0+r0)/2., 0)).toShape()
        inlet2 = Part.LineSegment(FreeCAD.Vector(0, (R0+r0)/2., 0),hubP1).toShape()
        outlet1 = Part.LineSegment(shroudP4,FreeCAD.Vector((obj.L+obj.b2+obj.L)/2., R2, 0)).toShape()
        outlet2 = Part.LineSegment(FreeCAD.Vector((obj.L+obj.b2+obj.L)/2., R2, 0), hubP4).toShape()

        # Creation of the separating meridional plane
        shroud1Discret = shroudDiscret[0:(obj.LePositionShroud+1)]
        shroud2Discret = shroudDiscret[obj.LePositionShroud:101]

        hub1Discret = hubDiscret[0:(obj.LePositionHub+1)]
        hub2Discret = hubDiscret[obj.LePositionHub:101]

        aveCurve1Discret = aveCurveDiscret[0:(obj.LePositionAverage+1)]
        aveCurve2Discret = aveCurveDiscret[obj.LePositionAverage:101]

        # Creation of the Leading Edge curve
        LePoints = [shroud2Discret[0],aveCurve2Discret[0], hub2Discret[0]]
        LeCurve = Part.BSplineCurve()
        LeCurve.interpolate(LePoints)
        
        LeCurveDiscret = LeCurve.discretize(Number = 100)

        Le1CurveDiscret = LeCurveDiscret[0:51]
        Le2CurveDiscret = LeCurveDiscret[50:101]

        # Creation a Wire of the Meridional plane of a blades
        shroud1 = Part.BSplineCurve()
        shroud1.interpolate(shroud1Discret)

        shroud2 = Part.BSplineCurve()
        shroud2.interpolate(shroud2Discret)

        hub1 = Part.BSplineCurve()
        hub1.interpolate(hub1Discret)

        hub2 = Part.BSplineCurve()
        hub2.interpolate(hub2Discret)

        Le1Curve = Part.BSplineCurve()
        Le1Curve.interpolate(Le1CurveDiscret)

        Le2Curve = Part.BSplineCurve()
        Le2Curve.interpolate(Le2CurveDiscret)

        aveCurve1 = Part.BSplineCurve()
        aveCurve1.interpolate(aveCurve1Discret)

        aveCurve2 = Part.BSplineCurve()
        aveCurve2.interpolate(aveCurve2Discret)

        if obj.CylindricalBlades==False:
            w = Part.Wire([inlet1,inlet2, shroud1.toShape(), shroud2.toShape(), outlet1, outlet2, hub1.toShape(), hub2.toShape(), Le1Curve.toShape(), Le2Curve.toShape(), aveCurve1.toShape(), aveCurve2.toShape()], closed = False)
        else:
            w = Part.Wire([inlet1,inlet2, shroud1.toShape(), shroud2.toShape(), outlet1, outlet2, hub1.toShape(), hub2.toShape(), Le1Curve.toShape(), Le2Curve.toShape()], closed = False)


        shroudSurface = shroud.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        hubSurface = hub.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        obj.Shape = Part.Compound([w, shroudSurface, hubSurface])

    def onChanged(self, obj, prop):
        FreeCAD.Console.PrintMessage("Updated property "+str(prop)+" of object "+str(obj.Name)+"\n")
     


class ModelOfBlade3D:
    def __init__(self, obj): 
        obj.Proxy = self 
        obj.addProperty("App::PropertyFloat", "AngleBeta1Shroud", "Angles of a Shroud streamline", "Inlet angle")#.AngleBeta1Shroud=15
        obj.addProperty("App::PropertyFloat", "AngleBeta2Shroud", "Angles of a Shroud streamline", "Outlet angle")#.AngleBeta2Shroud=25
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active")#.AngleBetaPoint1Shroud=(40,18,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Shroud", "Angles of a Shroud streamline", "Point of a B-spline, z-coodrinate non active")#.AngleBetaPoint2Shroud=(75,22.5,0)
        obj.addProperty("App::PropertyBool", "AnglesChartShroud", "Angles of a Shroud streamline", "It is shows chart angle Beta-Streamline")#.AnglesChartShroud=False

        obj.addProperty("App::PropertyFloat", "AngleBeta1Hub", "Angles of a Hub streamline", "Inlet angle")#.AngleBeta1Hub=32
        obj.addProperty("App::PropertyFloat", "AngleBeta2Hub", "Angles of a Hub streamline", "Outlet angle")#.AngleBeta2Hub=25
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active")#.AngleBetaPoint1Hub=(40,29,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Hub", "Angles of a Hub streamline", "Point of a B-spline, z-coodrinate non active")#.AngleBetaPoint2Hub=(75,27,0)
        obj.addProperty("App::PropertyBool", "AnglesChartHub", "Angles of a Hub streamline", "It is shows chart angle Beta-Streamline")#.AnglesChartShroud=False

        obj.addProperty("App::PropertyFloat", "AngleBeta1Ave", "Angles of an Average streamline (when Cylindrical blade = False)")#.AngleBeta1Ave=26
        obj.addProperty("App::PropertyFloat", "AngleBeta2Ave", "Angles of an Average streamline (when Cylindrical blade = False)")#.AngleBeta2Ave=25
        obj.addProperty("App::PropertyVector", "AngleBetaPoint1Ave", "Angles of an Average streamline (when Cylindrical blade = False)")#.AngleBetaPoint1Ave=(40,25.5,0)
        obj.addProperty("App::PropertyVector", "AngleBetaPoint2Ave", "Angles of an Average streamline (when Cylindrical blade = False)")#.AngleBetaPoint2Ave=(75,25,0)
        obj.addProperty("App::PropertyBool", "AnglesChartAverage", "Angles of an Average streamline (when Cylindrical blade = False)")#.AnglesChartAverage=False
        obj.addProperty("App::PropertyInteger", "N", "Number of the calculation points")#.N=1000

    def execute (self,obj):
        import Part, FreeCAD
        import numpy as np
        import matplotlib.pyplot as plt

        N=obj.N
        Mer=FreeCAD.ActiveDocument.getObject("Meridional")
        shroudEdge = Mer.Shape.Edges[3]
        shroudEdgeDiscret = shroudEdge.discretize(Number = N)
        shroudEdgeDiscret.append(shroudEdge.lastVertex().Point)

        hubEdge = Mer.Shape.Edges[7]
        hubEdgeDiscret = hubEdge.discretize(Number = N)
        hubEdgeDiscret.append(hubEdge.lastVertex().Point)

        N = len(shroudEdgeDiscret)

        # Blade와 Hub가 맞닿는 부분의 curve(theta angle streamline)를 parametric 하게 정의
        def streamlineBlade (AngleBeta1, AngleBetaPoint1, AngleBetaPoint2, AngleBeta2, DiscretEdgeOfMeridional):
            
            # Creation of Beta angle curve
            # Beta angle curve: beta angle vs. radial position (%)을 도시한 curve로, 중심 shaft에서 멀어질 수록 beta angle이 어떻게 변하는지에 대한 profile
            # 여기서 Beta angle curve는 r = d0/2, D0/2 즉 radial position이 0, 100%일 때와 AngleBetaPoint1, AngleBetaPoint2에서 정의된 두 개의 (radial position (%), beta angle, 0)
            # 이렇게 총 4개의 point를 B-Spline으로 보간하여 생성한다. 생성된 beta angle curve는 N개의 points로 discretization 되어 Theta angle streamline 정의 시 betai에 저장됨.
            betaPoints = [FreeCAD.Vector(0, AngleBeta1, 0), AngleBetaPoint1, AngleBetaPoint2 ,FreeCAD.Vector(100, AngleBeta2, 0)]
            betaCurve = Part.BSplineCurve()
            betaCurve.interpolate(betaPoints)
            betaCurveDiscret = []
            for i in range (1, (N+1), 1):
                vector_line1 = FreeCAD.Vector(float(i)/(float(N)/100.), -180, 0)
                vector_line2 = FreeCAD.Vector(float(i)/(float(N)/100.), 180, 0)
                line = Part.LineSegment(vector_line1, vector_line2)
                betaIntersect = betaCurve.intersectCC(line)
                betaCurveDiscret.append(betaIntersect)

            # Calculation of the Theta angle streamline
            # Theta angle streamline: class Meridional에서 정의한 Meridional Plane과 생성될 blade가 교차하며 생기는 streamline을 의미
            # 여기서 Theta angle streamline은
            #                                   
            #                                tan(beta) = dtheta/ds          (s는 meridional의 parameter([0, 1]))
            #                                ds/dl = 1/(y tan(beta))        (dl = y dtheta) -> 한 개의 Meridional Curve를 기준으로 작업하게 되므로, r을 y로 대체 가능
            #
            # 임을 사용하여 구분구적법을 통해 (s, theta), 즉 Meridional Plane 위에서의 좌표를 얻고 그에 대응되는 (x, y, z) 좌표를 얻음으로써 계산된다.
            ## Meridional curve의 discretization (N등분)
            ri = []
            for i in range (0,N,1):
                vector_i = DiscretEdgeOfMeridional[i]
                ri.append(vector_i.y) 
            
            ## Beta angle curve의 discretization (N등분)
            betai = []
            for i in range (0,N,1):
                vector_betai = betaCurveDiscret[i][0]
                betai.append(vector_betai.Y)
            
            ## ds/dl의 계산
            BfuncList = []
            for i in range (0,N,1):
                BfuncList.append(1./(ri[i]*np.tan(betai[i]*np.pi/180.)))

            ## 평균값 사용
            BfuncListSr = []
            for i in range (0, (N-1), 1):
                funcX = (BfuncList[i]+BfuncList[i+1])/2.
                BfuncListSr.append(funcX)

            ## Meridional curve의 discretization 된 point들의 x, y좌표 저장 
            ds_x = []
            ds_y = []
            for i in range (0, N, 1):
                vector_Edge = DiscretEdgeOfMeridional[i]
                ds_x.append(vector_Edge.x)
                ds_y.append(vector_Edge.y)

            ## ds = sqrt((dx)^2 + (dy)^2) 
            ds = []
            for i in range (0, (N-1), 1):
                ds.append(np.sqrt((ds_x[i+1]-ds_x[i])**2+(ds_y[i+1]-ds_y[i])**2))

            ## dtheta = tan(beta)*ds
            dTheta = []
            for i in range (0,(N-1),1):
                dTheta.append(ds[i]*BfuncListSr[i]*180./np.pi)

            ## theta = integral(tan(beta) ds)
            dThetaSum = [dTheta[0]]
            for i in range (0,(N-2),1):
                dThetaSum.append(dThetaSum[i]+dTheta[i+1])

            # Coordinates is XYZ of the streamline
            ## 위에선 한 개의 Meridional Curve에 대해서만 다루므로 r 대신 y 사용했지만, theta angle streamline은 모든 coordinate에 변화가 있음!!
            ## 따라서 x^i + rcos(theta)^j + rsin(theta)^k가 위치 벡터임을 사용!
            coord_x = ds_x

            coord_y = [ds_y[0]]
            for i in range(0, (N-1), 1):
                coord_y.append(ds_y[i]*np.cos(np.pi*dThetaSum[i]/180))

            coord_z = [0.0]
            for i in range(0, (N-1), 1):
                coord_z.append(ds_y[i]*np.sin(np.pi*dThetaSum[i]/180))

            # Streamline
            list_of_vectors = []
            for i in range (0, N, 1):
                vector = FreeCAD.Vector(coord_x[i],coord_y[i], coord_z[i])
                list_of_vectors.append(vector)
            
            ## discretization 되어있는, (x, y, z) coord 형태의 theta angle streamline을 B-Spline으로 보간하고
            ## streamline 곡선 / beta angle curve과 이에 대응되는 theta값 리스트를 리턴 
            streamline = Part.BSplineCurve()
            streamline.interpolate(list_of_vectors)
            return streamline.toShape(), betai, dThetaSum


        # Beta angle vs. theta를 출력
        def chartBetaTheta (betai, dThetaSum, AnglesChart):
            if AnglesChart == True:
                x_streamline = np.linspace(0, 1, N)
                y_beta = betai
                y_theta = [0.0]
                for i in range(0, (N-1), 1):
                    y = y_theta.append(dThetaSum[i])
                plt.plot(x_streamline, y_beta, x_streamline, y_theta)
                plt.xlabel("Streamline from 0 to 1")
                plt.ylabel("Angle Beta and Angle Theta, [degree]")
                plt.legend(["Beta angle","Theta angle = "+"%.2f" %dThetaSum[-1]], loc='upper right', bbox_to_anchor=(0.5, 1.00))
                plt.grid()
                plt.show()

        # Shroud 쪽에서 theta angle streamline 생성
        streamlineShroud, betaiShroud, dThetaShroudSum = streamlineBlade(obj.AngleBeta1Shroud, obj.AngleBetaPoint1Shroud, obj.AngleBetaPoint2Shroud, obj.AngleBeta2Shroud, shroudEdgeDiscret)
        streamlineShroudDisc = streamlineShroud.discretize(Number = 10)
        streamlineShroudBad = Part.BSplineCurve()
        streamlineShroudBad.interpolate(streamlineShroudDisc)

        # Hub 쪽에서 theta angle streamline 생성
        streamlineHub, betaiHub, dThetaHubSum = streamlineBlade(obj.AngleBeta1Hub, obj.AngleBetaPoint1Hub, obj.AngleBetaPoint2Hub, obj.AngleBeta2Hub, hubEdgeDiscret)
        streamlineHubDisc = streamlineHub.discretize(Number = 10)
        streamlineHubBad = Part.BSplineCurve()
        streamlineHubBad.interpolate(streamlineHubDisc)

        # Beta angle vs. theta 차트 출력
        chartShroud = chartBetaTheta(betaiShroud, dThetaShroudSum, obj.AnglesChartShroud)
        chartHub = chartBetaTheta(betaiHub, dThetaHubSum, obj.AnglesChartHub)

        # Cylindrical Blade를 생성하기 위해:
        # class Meridional에서 execute 된, average meridional curve를 기준으로 theta angle streamline(정확히 streamline이라는 보장은 없지만) 생성
        if Mer.CylindricalBlades == False:
            aveCurveEdge = Mer.Shape.Edges[11]
            aveCurveEdgeDiscret = aveCurveEdge.discretize(Number = obj.N)
            aveCurveEdgeDiscret.append(aveCurveEdge.lastVertex().Point)

            ## average theta angle streamline 생성 후 10 point를 선정하여(discretization), 이를 B-Spline으로 보간
            streamlineAverage, betaiAve, dThetaAveSum = streamlineBlade(obj.AngleBeta1Ave, obj.AngleBetaPoint1Ave, obj.AngleBetaPoint2Ave, obj.AngleBeta2Ave, aveCurveEdgeDiscret)
            streamlineAverageDisc = streamlineAverage.discretize(Number = 10)
            streamlineAveBad = Part.BSplineCurve()
            streamlineAveBad.interpolate(streamlineAverageDisc)	
            
            chartAve = chartBetaTheta(betaiAve, dThetaAveSum, obj.AnglesChartAverage)
            
            #pointMatrix1 = [streamlineHubDisc, streamlineAverageDisc, streamlineShroudDisc]
            #sur1 = Part.BSplineSurface()
            #sur1.interpolate(pointMatrix1)
            
            ## makeLoft로 보간
            c = Part.makeLoft([streamlineShroudBad.toShape(), streamlineAveBad.toShape(),streamlineHubBad.toShape()])

        # Cylindrical Blade가 아닌 경우엔 그냥 shroud와 hub 쪽의 theta angle streamline을 makeLoft로 보간
        else:
            #pointMatrix1 = [streamlineHubDisc, streamlineShroudDisc]

            #sur1 = Part.BSplineSurface()
            #sur1.interpolate(pointMatrix1)

            c = Part.makeLoft([streamlineShroudBad.toShape(), streamlineHubBad.toShape()])

        obj.Shape = c

    def onChanged(self, obj, prop):
        FreeCAD.Console.PrintMessage("Updated property "+str(prop)+" of object "+str(obj.Name)+"\n") 



class Blades:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyFloat", "ThicknessLEShroud", "Shroud profile", "Value of the LE")#.ThicknessLEShroud = Mer.D2*0.02
        obj.addProperty("App::PropertyVector", "ThicknessPoint1Shroud", "Shroud profile", "Point for B-Spline thickness")#.ThicknessPoint1Shroud = (40, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyVector", "ThicknessPoint2Shroud", "Shroud profile", "Point for B-Spline thickness")#.ThicknessPoint2Shroud = (75, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyFloat", "ThicknessTEShroud", "Shroud profile", "Value of the TE")#.ThicknessTEShroud = Mer.D2*0.02

        obj.addProperty("App::PropertyFloat", "ThicknessLEHub", "Hub profile", "Value of the thickness LE")#.ThicknessLEHub = Mer.D2*0.02
        obj.addProperty("App::PropertyVector", "ThicknessPoint1Hub", "Hub profile", "Point for B-Spline thickness")#.ThicknessPoint1Hub = (40, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyVector", "ThicknessPoint2Hub", "Hub profile", "Point for B-Spline thickness")#.ThicknessPoint2Hub = (75, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyFloat", "ThicknessTEHub", "Hub profile", "Value of the thickness TE")#.ThicknessTEHub = Mer.D2*0.02

        obj.addProperty("App::PropertyFloat", "ThicknessLEAve", "Average profile", "Value of the thickness LE")#.ThicknessLEAve = Mer.D2*0.02
        obj.addProperty("App::PropertyVector", "ThicknessPoint1Ave", "Average profile", "Value of the thickness")#. ThicknessPoint1Ave = (40, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyVector", "ThicknessPoint2Ave", "Average profile", "Value of The thickness")#. ThicknessPoint2Ave = (75, Mer.D2*0.02, 0)
        obj.addProperty("App::PropertyFloat", "ThicknessTEAve", "Average profile", "Value of the thickness TE")#. ThicknessTEAve = Mer.D2*0.02

        obj.addProperty("App::PropertyBool", "TraillingEdgeEllipse", "Type of the LE and TE", "Type of the trailling edge")#.TraillingEdgeEllipse = False
        obj.addProperty("App::PropertyInteger", "LeadingEdgeType", "Type of the LE and TE", "Type of the leading edge")#.LeadingEdgeType = 1
        obj.addProperty("App::PropertyInteger", "TraillingEdgeType", "Type of the LE and TE", "Type of the trailling edge" )#.TraillingEdgeType = 1
        obj.addProperty("App::PropertyInteger", "NumberOfBlades", "Number of blades")#.NumberOfBlades=7

        obj.addProperty("App::PropertyBool", "FullDomainCFD", "CFD", "Create full CFD Domain")#.FullDomainCFD = False
        obj.addProperty("App::PropertyBool", "PeriodicDomainCFD", "CFD", "Create periodic CFD Domain")#.PeriodicDomainCFD = False
        obj.addProperty("App::PropertyFloat", "HalfD3toD2", "CFD", "Value of half relationship D3/D2")#.HalfD3toD2 = 1.2

        obj.addProperty("App::PropertyFloat", "ThicknessShroud", "Shroud", "Value of the Thickness Shroud")#.ThicknessShroud = Mer.D2*0.02
        obj.addProperty("App::PropertyFloat", "ThicknessHub", "Hub", "Value of the Thickness Hub")#.ThicknessHub = Mer.D2*0.02

    def execute (self, obj):
        import Part, FreeCAD, DraftTools, Draft
        import numpy as np
        import copy

        # Creation of the profile of streamlines:
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")
        Blade = App.ActiveDocument.getObject("ModelOfBlade3D")
        
        ## 이름 그대로임. BladeFace/BladeSurface = ModelOfBlade3D의 blade surface, BladeEdgeShroud/BladeEdgeHub/BladeEdgeTe = ModelOfBlade3D의 shroud, hub, TE 쪽 edge
        BladeFace = Blade.Shape.Faces[0]
        BladeSurface = BladeFace.Surface
        BladeEdgeShroud = BladeFace.Edges[0]
        BladeEdgeHub = BladeFace.Edges[2]
        BladeEdgeTe = BladeFace.Edges[1]

        R2 = Mer.D2/2.
        R0 = Mer.D0/2.
        r0 = Mer.d0/2.

        # LE에서 TE로 가면서 thickness가 어떻게 변할지 그 profile을 정의
        def thicknessProfile (ThicknessLE, ThicknessPoint1, ThicknessPoint2, ThicknessTE, BladeStreamlineEdge, BladeFace, BladeSurface, NameOfDisk, EdgeTe, LECoeff1, TECoeff1,TraillingEdgeEllipse, UVoutlet, Extend):
        ### NameOfDisk, EdgeTe, LECoeff1, TECoeff1,TraillingEdgeEllipse, UVoutlet, Extend):			
            BladesEdgeDicretForLe = BladeStreamlineEdge.discretize(Distance = ThicknessLE/4.)
            BladesEdgeDicretForSide = BladeStreamlineEdge.discretize(Number = 20)

            if Mer.CylindricalBlades == False:
                BladesEdgeDicret = []
                for i in range (0, 10, 1):
                    BladesEdgeDicret.append(BladesEdgeDicretForLe[i])

                for i in range (3, len(BladesEdgeDicretForSide), 1):
                    BladesEdgeDicret.append(BladesEdgeDicretForSide[i])

                OutletVector = BladeFace.valueAt(Extend, UVoutlet)

                BladesEdgeDicret.append(OutletVector)
            else:
                BladesEdgeDicret = []
                for i in range (0, 10, 1):
                    BladesEdgeDicret.append(BladesEdgeDicretForLe[i])

                for i in range (3, len(BladesEdgeDicretForSide), 1):
                    BladesEdgeDicret.append(BladesEdgeDicretForSide[i])

                OutletVector = BladeFace.valueAt(1.1*BladeEdgeShroud.Length, UVoutlet)

                BladesEdgeDicret.append(OutletVector)

    #################
            # Creation of Thickness curve of the Shroud
            ThicknessPoints = [FreeCAD.Vector(0, ThicknessLE, 0), ThicknessPoint1, ThicknessPoint2, FreeCAD.Vector(100, ThicknessTE, 0)]
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
        BladeEdgeShroudDiscret = BladeEdgeShroud.discretize(Distance = obj.ThicknessLEShroud/4.)###thicknessProfile내에도 동일한 함수 있음

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

        if Mer.CylindricalBlades == False:

            # Creation of the average streamline in surface blade
            aveMerLine = Mer.Shape.Edges[11]
            revolveAveLine = aveMerLine.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), 360)

            aveCurveIntersect = revolveAveLine.Surface.intersectSS(BladeSurface)
            aveCurve = aveCurveIntersect[0]

            AveProfile = thicknessProfile(obj.ThicknessLEAve, obj.ThicknessPoint1Ave, obj.ThicknessPoint2Ave, obj.ThicknessTEAve, aveCurve, BladeFace, BladeSurface, "Average", BladeEdgeTe, obj.LeadingEdgeType, obj.TraillingEdgeType, obj.TraillingEdgeEllipse, 0.5, 1.05)

            #BladeSurfaceModel = Part.makeLoft([ShroudProfile2, ShroudProfile, AveProfile, HubProfile, HubProfile2])
            BladeSurfaceModel = Part.makeLoft([ShroudProfile2,AveProfile, HubProfile2])
        else:
            #BladeSurfaceModel = Part.makeLoft([ShroudProfile2, ShroudProfile, HubProfile, HubProfile2])
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

        LineShroudBlock = Part.LineSegment(FreeCAD.Vector(Mer.L/2, R2), FreeCAD.Vector(Mer.L/2, R2*1.2))
        LineHubBlock = Part.LineSegment(FreeCAD.Vector((Mer.L+Mer.b2)*2, R2), FreeCAD.Vector((Mer.L+Mer.b2)*2, R2*1.2))
        LineTopBlock = Part.LineSegment(FreeCAD.Vector(Mer.L/2, R2*1.2), FreeCAD.Vector((Mer.L+Mer.b2)*2, R2*1.2))
        LineDownBlock = Part.LineSegment(FreeCAD.Vector(Mer.L/2, R2), FreeCAD.Vector((Mer.L+Mer.b2)*2, R2))

        FaceShroudBlock = LineShroudBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceHubBlock = LineHubBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceTopBlock = LineTopBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        FaceDownBlock = LineDownBlock.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        BlockShell = Part.makeShell([FaceShroudBlock, FaceHubBlock, FaceTopBlock, FaceDownBlock])
        BlockSolid = Part.makeSolid(BlockShell)

        BladeSolidSecond = BladeSolidInit.cut(BlockSolid)


        if obj.FullDomainCFD == False and obj.PeriodicDomainCFD == False:###그냥 Blade Shape만 만들기
            #Cut of the Shroud Side of Blade 
            FaceShroudCut =Mer.Shape.Faces[0]		
            LineTopShroudCut = Part.LineSegment(FreeCAD.Vector(Mer.L, R2, 0), FreeCAD.Vector(0, R2, 0))
            FaceTopShroudCut = LineTopShroudCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            LineInletShroudCut = Part.LineSegment(FreeCAD.Vector(0, R2, 0), FreeCAD.Vector(0, R0, 0))
            FaceInletShroudCut = LineInletShroudCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            ShroudCutBLockShell = Part.makeShell([FaceShroudCut, FaceTopShroudCut, FaceInletShroudCut])
            ShroudCutBLockSolid = Part.makeSolid(ShroudCutBLockShell)

            BladeSolidThree = BladeSolidSecond.cut(ShroudCutBLockSolid)

            # Cut of the Hub Side Of Blade
            #FaceHubCut = Mer.Shape.Faces[1]
            #LineTopHubCut = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2, R2, 0), FreeCAD.Vector(Mer.L+Mer.b2*2, R2, 0))
            #FaceTopHubCut = LineTopHubCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            #LineSideHubCut = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2*2, R2, 0), FreeCAD.Vector(Mer.L+Mer.b2*2, 0, 0))
            #FaceSideHubCut = LineSideHubCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            #LineSide2HubCut = Part.LineSegment(FreeCAD.Vector(0, r0, 0), FreeCAD.Vector(0, 0, 0))
            #FaceSide2HubCut = LineSide2HubCut.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            #HubCutBLockShell = Part.makeShell([FaceHubCut, FaceTopHubCut, FaceSideHubCut, FaceSide2HubCut])
            #HubCutBLockSolid = Part.makeSolid(HubCutBLockShell)	

            #BladeSolid = BladeSolidThree.cut(HubCutBLockSolid)	
            
            Cut1 = Part.LineSegment(FreeCAD.Vector(0,0,0),FreeCAD.Vector(0,(Mer.ds+Mer.d0)/4.,0))
            Rot1 = Cut1.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut2 = Part.LineSegment(FreeCAD.Vector(0,(Mer.ds+Mer.d0)/4.,0),FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub/2.,(Mer.ds+Mer.d0)/4.,0))
            Rot2 = Cut2.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut3 = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub/2.,(Mer.ds+Mer.d0)/4.,0),FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub/2.,Mer.D2/2.,0))
            Rot3 = Cut3.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut4 = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub/2.,Mer.D2/2.,0),FreeCAD.Vector(Mer.L+2*Mer.b2+obj.ThicknessHub,Mer.D2/2.,0))
            Rot4 = Cut4.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            Cut5 = Part.LineSegment(FreeCAD.Vector(Mer.L+2*Mer.b2+obj.ThicknessHub,Mer.D2/2.,0),FreeCAD.Vector(Mer.L+2*Mer.b2+obj.ThicknessHub,0,0))
            Rot5 = Cut5.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            
            HubCutBLockShell = Part.makeShell([Rot1, Rot2, Rot3, Rot4, Rot5])
            HubCutBLockSolid = Part.makeSolid(HubCutBLockShell) 

            BladeSolid = BladeSolidThree.cut(HubCutBLockSolid)

            BladesList = []
            # ### Creation of hub Solid
            # rs = Mer.ds/2.
            # FaceHubCut = Mer.Shape.Faces[1]
            # # HubEdge1 = Mer.Shape.Edges[6]
            # # HubEdge2 = Mer.Shape.Edges[7]
            # HubEdgeOutlet = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2,R2,0), FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub,R2,0))
            # FaceHubEdgeOutlet = HubEdgeOutlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            # HubEdgeBottom = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub,R2,0), FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub,rs,0))
            # FaceHubEdgeBottom = HubEdgeBottom.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            # HubEdgeShaft = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+obj.ThicknessHub,rs,0), FreeCAD.Vector(0,rs,0))
            # FaceHubEdgeShaft = HubEdgeShaft.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
            # HubEdgeInlet = Part.LineSegment(FreeCAD.Vector(0,r0,0), FreeCAD.Vector(0,rs,0))
            # FaceHubEdgeInlet = HubEdgeInlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            # HubShell = Part.makeShell([FaceHubCut, FaceHubEdgeOutlet, FaceHubEdgeBottom, FaceHubEdgeShaft, FaceHubEdgeInlet])
            # HubSolid = Part.makeSolid(HubShell)
            # HubList = []
            # HubList.append(HubSolid)


        # Creation of a massive of blades		
            AngleRotateBlade = 360./float(obj.NumberOfBlades)
            # BladesList = []
            for i in range (0, obj.NumberOfBlades, 1):
                BladeSolidi = BladeSolid.copy()
                BladeSolidi.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), AngleRotateBlade*(i))
                BladesList.append(BladeSolidi)
                # HubSolid = HubSolid.fuse(BladeSolidi)

            # BladesList.append(HubSolid)


        elif obj.FullDomainCFD==True and obj.PeriodicDomainCFD == False:### Full Domain CFD 형상
        # Creation of a massive of blades without cut of shroud and hub parts
            AngleRotateBlade = 360./float(obj.NumberOfBlades)
            Blades = []
            for i in range (0, obj.NumberOfBlades, 1):
                BladeSolidi = BladeSolidSecond.copy()
                BladeSolidi.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), AngleRotateBlade*(i))
                Blades.append(BladeSolidi)
        # Creation of a CFD part without blades
            FaceShroudCut =Mer.Shape.Faces[0]
            FaceHubCut = Mer.Shape.Faces[1]

            EdgeInlet = Part.LineSegment(FreeCAD.Vector(0,r0,0), FreeCAD.Vector(0,R0, 0))
            FaceInlet = EdgeInlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            EdgeShroud2 = Part.LineSegment(FreeCAD.Vector(Mer.L, R2, 0), FreeCAD.Vector(Mer.L, R2*obj.HalfD3toD2, 0))
            FaceShroud2 = EdgeShroud2.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            EdgeHub2 = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2, R2, 0), FreeCAD.Vector(Mer.L+Mer.b2, R2*obj.HalfD3toD2, 0))
            FaceHub2 = EdgeHub2.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            EdgeOutlet = Part.LineSegment(FreeCAD.Vector(Mer.L, R2*obj.HalfD3toD2, 0), FreeCAD.Vector(Mer.L+Mer.b2, R2*obj.HalfD3toD2, 0))
            FaceOutlet = EdgeOutlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

            CFDDomainShell = Part.makeShell([FaceInlet, FaceShroudCut, FaceShroud2, FaceOutlet, FaceHub2, FaceHubCut])
            CFDDomainSolid = Part.makeSolid(CFDDomainShell)
            CFDDomainSolid.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), -(360/obj.NumberOfBlades)/2)

            CFDDomain = CFDDomainSolid.cut(Part.Compound(Blades))

            BladesList = [CFDDomain]

        elif obj.FullDomainCFD==False and obj.PeriodicDomainCFD == True:###Periodic Domain CFD 형상
        # Creation Periodic CFD Domain
            AngleRotateFacePeriodic = 180./float(obj.NumberOfBlades)

            EdgeInlet = Part.LineSegment(FreeCAD.Vector(0,r0,0), FreeCAD.Vector(0,R0, 0)).toShape()
            EdgeShroud = Mer.Shape.Edges[2]
            EdgeHub = Mer.Shape.Edges[6]
            EdgeLE = Blade.Shape.Edges[3]
            WireInlet = Part.Wire([EdgeInlet, EdgeShroud, EdgeLE, EdgeHub])
            FaceInletCFD = Part.Face(WireInlet)


            FacePeriodic1 = BladeFace.copy()
            FacePeriodic1.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), AngleRotateFacePeriodic)
            FaceInletCFD.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), AngleRotateFacePeriodic)


            CFDSolid1 = FacePeriodic1.revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), -2*AngleRotateFacePeriodic)
            CFDSolid2 = FaceInletCFD.revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), -2*AngleRotateFacePeriodic)
            
        
        # Creation of the outlet part of periodic CFD domain
            EdgeFirstOut = CFDSolid1.Edges[1]
            VertexFirstOut = EdgeFirstOut.firstVertex()
            VectorFirstOut = VertexFirstOut.Point
            VectorFirstZero = FreeCAD.Vector(Mer.L,0,0)
            LineFirstOut = Part.LineSegment(VectorFirstZero, VectorFirstOut).toShape()
            VectorFirstOutAdd = LineFirstOut.valueAt(LineFirstOut.Length*obj.HalfD3toD2)
            LineFirstOutAdd = Part.LineSegment(VectorFirstOut, VectorFirstOutAdd).toShape()

            EdgeSecondOut = CFDSolid1.Edges[4]
            VertexSecondOut = EdgeSecondOut.firstVertex()
            VectorSecondOut = VertexSecondOut.Point
            VectorSecondZero = FreeCAD.Vector(Mer.L+Mer.b2, 0, 0)
            LineSecondOut = Part.LineSegment(VectorSecondZero, VectorSecondOut).toShape()
            VectorSecondOutAdd = LineSecondOut.valueAt(LineSecondOut.Length*obj.HalfD3toD2)
            LineSecondOutAdd = Part.LineSegment(VectorSecondOut, VectorSecondOutAdd).toShape()

            LineOutlet = Part.LineSegment(VectorFirstOutAdd, VectorSecondOutAdd).toShape()
            ListEdgesOutletBlock = [CFDSolid1.Edges[5], LineFirstOutAdd, LineSecondOutAdd, LineOutlet]
            FaceOutBlock = Part.makeFilledFace(ListEdgesOutletBlock)

            SolidOutletCFD = FaceOutBlock.revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), -2*AngleRotateFacePeriodic)
            

            CFDSolid3 = CFDSolid1.fuse(CFDSolid2)
            CFDSolid = CFDSolid3.fuse(SolidOutletCFD)
            CFDSolid = CFDSolid.cut(BladeSolidSecond)
            BladesList = [CFDSolid]

        #@
        obj.Shape = Part.makeSolid(Part.Compound(BladesList))
        #obj.Shape = BladeSolidInit

    def onChanged(self, obj, prop):
        FreeCAD.Console.PrintMessage("Updated property "+str(prop)+" of object "+str(obj.Name)+"\n") 



class Hub:
    def __init__(self, obj):
        obj.Proxy = self

    def execute(self, obj):
        import Part, FreeCAD
        Mer = FreeCAD.ActiveDocument.getObject("Meridional")
        Bla = FreeCAD.ActiveDocument.getObject("Blades")
                
        R2 = Mer.D2/2.
        R0 = Mer.D0/2.
        r0 = Mer.d0/2.
        rs = Mer.ds/2.
        
        FaceHubCut = Mer.Shape.Faces[1]
        HubEdgeOutlet = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2,R2,0), FreeCAD.Vector(Mer.L+Mer.b2+Bla.ThicknessHub,R2,0))
        FaceHubEdgeOutlet = HubEdgeOutlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        HubEdgeBottom = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+Bla.ThicknessHub,R2,0), FreeCAD.Vector(Mer.L+Mer.b2+Bla.ThicknessHub,rs,0))
        FaceHubEdgeBottom = HubEdgeBottom.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        HubEdgeShaft = Part.LineSegment(FreeCAD.Vector(Mer.L+Mer.b2+Bla.ThicknessHub,rs,0), FreeCAD.Vector(0,rs,0))
        FaceHubEdgeShaft = HubEdgeShaft.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)
        HubEdgeInlet = Part.LineSegment(FreeCAD.Vector(0,r0,0), FreeCAD.Vector(0,rs,0))
        FaceHubEdgeInlet = HubEdgeInlet.toShape().revolve(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 360)

        HubShell = Part.makeShell([FaceHubCut, FaceHubEdgeOutlet, FaceHubEdgeBottom, FaceHubEdgeShaft, FaceHubEdgeInlet])
        HubSolid = Part.makeSolid(HubShell)
        HubList = []
        
        #@
        obj.Shape = HubSolid
        #HubList.append(HubSolid)
        #obj.Shape = Part.Compound(HubList)    

                             


# Handles prop file(parameters.txt inside os.getcwd()) modification: makes updated props list and updates it.
class PropfileChangeEventHandler(PatternMatchingEventHandler):
    def __init__(self):
        PatternMatchingEventHandler.__init__(self, patterns=['parameters.txt'], ignore_directories=True)

    def on_modified(self, event):
        #@
        global upd
        nameValueList = readNameValue()

        print("\nhandling parameters.txt update")
        obj1 = FreeCAD.ActiveDocument.getObject("Meridional")
        obj2 = FreeCAD.ActiveDocument.getObject("ModelOfBlade3D")
        obj3 = FreeCAD.ActiveDocument.getObject("Blades")

        obj1UpdateCheck = checkPropertiesUpdate(obj1, nameValueList)
        obj2UpdateCheck = checkPropertiesUpdate(obj2, nameValueList)
        obj3UpdateCheck = checkPropertiesUpdate(obj3, nameValueList)
        
        #@
        upd1, upd2, upd3 = False, False, False
        if(obj1UpdateCheck):
            print("obj1 upd: ", obj1UpdateCheck)
            upd1 = True
            if(bindProperties(obj1, obj1UpdateCheck, nameValueList)):
                FreeCAD.Console.PrintMessage("Successfully updated property for Meridional\n")
            else:
                FreeCAD.Console.PrintMessage("Failed to update property for Meridional\n")
         
        if(obj2UpdateCheck):
            print("obj2 upd: ", obj2UpdateCheck)
            upd2 = True
            if(bindProperties(obj2, obj2UpdateCheck, nameValueList)):
                FreeCAD.Console.PrintMessage("Successfully updated property for ModelOfBlade3D\n")
            else:
                FreeCAD.Console.PrintMessage("Failed to update property for ModelOfBlade3D\n")

        if(obj3UpdateCheck):
            print("obj3 upd: ", obj3UpdateCheck)
            upd3 = True
            if(bindProperties(obj3, obj3UpdateCheck, nameValueList)):
                FreeCAD.Console.PrintMessage("Successfully updated property for Blades\n")
            else:
                FreeCAD.Console.PrintMessage("Failed to update property for Blades\n")

        upd = any([upd1, upd2, upd3])


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
            if(not(re.search(r'^[#\s\t]', line))):
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


# Writes FreeCAD object property to parameters.txt
# nameValueList should be properties type/name/value from FreeCAD.ActiveDocument.getObject(~).PropertiesList
### Call only after parameter update loop
def writeNameValue(nameValueList):
    typeSearch = (lambda line: re.search(r'^float|int|fvec|ivec|bool{1}', line))
    # t is for look-around regex search -> stripped down typeSearch result in string
    nameSearch = (lambda line, t: re.search(r'(?<=^'+t+r').+[\s\t]*(?==)', line))
    valueSearch = (lambda line: re.search(r'(?<==).+$', line))
    stripPattern = re.compile(r'[\s\t]')

    anyPropUpd = False
    fileNameList = list()
    delLinesIndex = list()
    with open(os.getcwd()+"\\parameters.txt", mode='r', encoding='UTF-8') as f:
        lines = f.readlines()
        for (line, i) in zip(lines, range(0, len(lines), 1)):
            if(not(re.search(r'^[#\s\t]', line))):
                typ = typeSearch(line)
                if(typ):
                    t = stripPattern.sub('', line[typ.start():typ.end()])
                
                    name = nameSearch(line, t)
                    if(name):
                        n = stripPattern.sub('', line[name.start():name.end()])
                        if(n):
                            fileNameList.append(n)

                    value = valueSearch(line)
                    if(value):
                        v = typeParser(t, stripPattern.sub('', line[value.start():value.end()]))
                        if(bool(n) & (v is not None)):
                            try:
                                N = nameValueList[0].index(n)
                                V = typeParser(t, str(nameValueList[1][N]))
                                if(v != V):
                                    lines[i] = re.sub(r'(?<==).+$', ' '+str(V), line)
                                    print("Changed property "+str(n)+" from "+str(v)+" to "+str(V)+".")
                                    anyPropUpd |= True
                            except ValueError:
                                delLinesIndex.append(i)
                                print("Deleted property "+str(n)+".")
                                anyPropUpd |= True

        for (i, j) in zip(delLinesIndex, range(0, len(delLinesIndex), 1)):
            del lines[i-j]

        for (appName, i) in zip(nameValueList[0], range(0, len(nameValueList[0]), 1)):
            appTypeName = type(nameValueList[1][i]).__name__
            if(not appName in fileNameList):
                if(appTypeName == "Vector"):
                    if(type(nameValueList[1][i][0]).__name__ == "float"):
                        appTypeName = "fvec"
                    elif(type(nameValueList[1][i][0]).__name__ == "int"):
                        appTypeName = "ivec"
                lines.append(str(appTypeName)+" "+str(appName)+" = "+str(typeParser(appTypeName, str(nameValueList[1][i])))+"\n")
                print("Added property "+str(appName)+" to parameters_upd.txt")
                anyPropUpd |= True
    f.close()
    if(anyPropUpd):
        with open(os.getcwd()+"\\parameters_upd.txt", mode='w', encoding='UTF-8') as f:
            f.seek(0)
            f.truncate()
            f.writelines(lines)
        f.close()


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


# Check and adds properties that obj.PropertiesList != parameters.txt to updateList returns it
def checkPropertiesUpdate(obj, nameValueList):
    updateList = list()
    
    if(nameValueList):
        for (name, value) in zip(nameValueList[0], nameValueList[1]):
            if(name in obj.PropertiesList):
                V = getattr(obj, name)
                if(V != value):
                    if(type(V).__name__ == "Vector"):
                        if(any(map(lambda i, j: i!=float(j), V, value))):
                            updateList.append(name)
                    else:
                        updateList.append(name)
    if(updateList):
        return updateList
    else:
        return None




print("FCStd, step 파일을 저장할 위치?: ")
savePath = input()

print("\nFreeCAD 설치 위치? (e.g. C:\\Users\\계정 이름\\AppData\\Local\\Programs\\FreeCAD 0.19) ?:")
freecadPath = input()

FreeCADGui.showMainWindow()
FreeCADGui.updateGui()

with open(freecadPath+"\\data\\Mod\\Start\\StartPage\\LoadNew.py") as file:
    exec(file.read())


# Creation of all three objects
initialNameValueList = readNameValue()

myObj = FreeCAD.ActiveDocument.addObject("Part::FeaturePython", "Meridional") 
Meridional(myObj)
if(bindProperties(myObj, initialNameValueList[0], initialNameValueList)):
    #@
    print("\nbind1 ok")
    readNameValueObj = list(map(lambda i: getattr(myObj, i), myObj.PropertiesList)) 
    print(readNameValueObj)

    myObj.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too
    #FreeCAD.ActiveDocument.recompute()

myObj2 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'ModelOfBlade3D')
ModelOfBlade3D(myObj2)
if(bindProperties(myObj2, initialNameValueList[0], initialNameValueList)):
    #@
    print("\nbind2 ok")
    readNameValueObj = list(map(lambda i: getattr(myObj2, i), myObj2.PropertiesList))
    print(readNameValueObj)

    myObj2.ViewObject.Proxy = 0 # this is mandatory unless we code the ViewProvider too
    #FreeCAD.ActiveDocument .recompute()

myObj3 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'Blades')
Blades(myObj3)
if(bindProperties(myObj3, initialNameValueList[0], initialNameValueList)):
    #@
    print("\nbind3 ok")
    readNameValueObj = list(map(lambda i: getattr(myObj3, i), myObj3.PropertiesList))
    print(readNameValueObj)

    myObj3.ViewObject.Proxy = 0 # this is mand atory unless we code the ViewProvider too
    #FreeCAD.ActiveDocument.recompute()

myObj4 = FreeCAD.ActiveDocument.addObject('Part::FeaturePython', 'Hub')
Hub(myObj4)
myObj4.ViewObject.Proxy = 0
FreeCAD.ActiveDocument.recompute()


# View
FreeCAD.ActiveDocument.getObject("Meridional").Visibility = False
FreeCAD.ActiveDocument.getObject("ModelOfBlade3D").Visibility = False
FreeCAD.ActiveDocument.getObject("Blades").Visibility = True
FreeCAD.ActiveDocument.getObject("Hub").Visibility = True
FreeCADGui.ActiveDocument.getObject("Blades").Transparency = 0
FreeCADGui.ActiveDocument.getObject("Hub").Transparency = 0
FreeCADGui.updateGui()


# Start watchdog observer and handler for text file modification check
wdgObserver = Observer()
wdgHandler = PropfileChangeEventHandler()

wdgObserver.schedule(wdgHandler, os.getcwd(), recursive=False)
wdgObserver.start()


# Run text file-based parameter update loop while FreeCAD GUI is on or the document is still online.
try:
    while(True):
        FreeCADGui.updateGui()
        try:
            if(upd):
                print("\nRecomputing...\n")
                # Force recompute as we need to recompute every single object in ActiveDocument
                for obj in FreeCAD.ActiveDocument.Objects:
                    obj.touch()
                FreeCAD.ActiveDocument.recompute()
                print("Done recomputing...\n\n\n")
                upd = False
        except NameError:
            continue
except KeyboardInterrupt:
    wdgObserver.stop()
wdgObserver.join()


# update parameters.txt with current object parameters
excludeNameList = ["ExpressionEngine", "Label", "Label2", "Shape", "Proxy", "Visibility", "Placement"]
finalNameList1 = list(filter(lambda name: False if name in excludeNameList else True, FreeCAD.ActiveDocument.getObject("Meridional").PropertiesList))
finalValueList1 = list(map(lambda name: getattr(FreeCAD.ActiveDocument.getObject("Meridional"), name), finalNameList1))

finalNameList2 = list(filter(lambda name: False if name in excludeNameList else True, FreeCAD.ActiveDocument.getObject("ModelOfBlade3D").PropertiesList))
finalValueList2 = list(map(lambda name: getattr(FreeCAD.ActiveDocument.getObject("ModelOfBlade3D"), name), finalNameList2))

finalNameList3 = list(filter(lambda name: False if name in excludeNameList else True, FreeCAD.ActiveDocument.getObject("Blades").PropertiesList))
finalValueList3 = list(map(lambda name: getattr(FreeCAD.ActiveDocument.getObject("Blades"), name), finalNameList3))

finalNameList = finalNameList1 + finalNameList2 + finalNameList3
finalValueList = finalValueList1 + finalValueList2 + finalValueList3

print("\nupdating to file parameters.txt, if needed. take a look at parameters_upd.txt for updated parameters.")
writeNameValue([finalNameList, finalValueList])
print("updating current object parameters to file finished.\n")


# Register all three objects to __objs__ for step file, and finally save them in save path.
Meridional = FreeCAD.ActiveDocument.getObject("Meridional")
ModelOfBlade3D = FreeCAD.ActiveDocument.getObject("ModelOfBlade3D")
Blades = FreeCAD.ActiveDocument.getObject("Blades")
Hub = FreeCAD.ActiveDocument.getObject("Hub")
__objs__ = []

if((Blades.FullDomainCFD == False) & (Blades.PeriodicDomainCFD == False)):
    if("Fusion" not in list(map(lambda objs: objs.Name, FreeCAD.ActiveDocument.Objects))):
        Fusion = FreeCAD.ActiveDocument.addObject("Part::MultiFuse", "Fusion")
    Fusion = FreeCAD.ActiveDocument.getObject("Fusion")
    Fusion.Shapes = [Blades, Hub]
    __objs__.append(Fusion)

    FreeCAD.ActiveDocument.recompute()
#else:


# Export step and save FCStd file.
Part.export(__objs__, savePath+"\\impeller3D.stl")
FreeCAD.ActiveDocument.saveAs(savePath+"\\impeller3D.FCStd")
