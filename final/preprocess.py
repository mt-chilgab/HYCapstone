import gmsh
import math
import re
import os, sys
import numpy as np


def runGmsh():
    gmsh.initialize()

    gmsh.model.add("impeller")

    # Load a STEP file (using `importShapes' instead of `merge' allows to directly retrieve the tags of the highest dimensional imported entities):
    path = os.path.dirname(os.path.abspath(__file__))
    gmsh.merge(os.path.join(path, 'gmsh\\box.opt'))
    v = gmsh.merge(os.path.join(path, 'impeller.step'))

    
    gmsh.model.mesh.generate(3)

    gmsh.write("impeller.inp")

    # Launch the GUI to see the results:
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()
    print("Mesh Generation OK")
    gmsh.finalize()


def inpManipulation():
    c = np.arange(10)

    #del __objs__
    with open('impeller.inp', 'r') as f:
        list_file = []
        NumberSurface = 0
        for line in f:
            list_file.append(line)
            if re.search('Surface',line):
                NumberSurface += 1
    f.close()

    found1 = list_file.index('*ELEMENT, type=CPS3, ELSET=Surface'+str(NumberSurface)+'\n')
    # found2 = list_file.index('*ELEMENT, type=CPS6, ELSET=Surface6\n')
    found3 = list_file.index('*ELEMENT, type=T3D2, ELSET=Line1\n') 
    found4 = list_file.index('*ELEMENT, type=C3D4, ELSET=Volume1\n') 

    comp1=list_file[found1+1:found4-1] 
    # comp2=list_file[found2+1:found4-1] 

    del list_file[found3:found4] 
    del list_file[2] 
    list_file.insert(2,'*NODE, NSET=Nall\n') 


    
    a1=[]
    b1=[]
    sz1 = len(comp1)
    for i in comp1:
        a1=i.split(',')
        sz2=len(a1)
        for j in a1:
            b1 = b1 + [int(j)]

    for i in range(1,sz1+1):
        del b1[(sz2-1)*(i-1)]

    # print(b1)

    #b=np.reshape(b,(sz1,sz2-1))
    for i in range(len(b1)):
        b1[i] = str(b1[i])
    

    string1=[',\n'.join(b1)]
    string1.insert(0,'*NSET,NSET=Nfix\n')

    with open('impeller2.inp','w') as f:
        for name in list_file:
            f.write(name)
        f.close()

    with open('fixed.nam','w') as f:
        for name in string1:
            f.write(name)
        f.close()

    # Read material properties from parameters.txt
    searchKeyword = lambda keyword: False if re.search(r'^matprop[\s\t]+\w{4,6}[\s\t]+'+str(keyword)+r'[\s\t]*(?==)', line) is None else True
    getValueString = lambda line: line.split("=")[1].strip(" ")
    with open('parameters.txt', 'r') as f:
        list_file = []
        for line in f:
            if searchKeyword("E") or searchKeyword("rho"):
                list_file.append(getValueString(line))
        f.close()

    string=[]
    string.insert(0,'*INCLUDE, INPUT=impeller2.inp \n*INCLUDE, INPUT=fixed.nam \n\n*BOUNDARY\nNfix,1,3\n')
    string.insert(1,'*Material, name=ALUMINIUM\n')
    string.insert(2,'*Elastic\n')
    string.insert(3,list_file[0])
    string.insert(4,'*Density\n')
    string.insert(5,list_file[1])
    string.insert(6,'*Solid Section, elset=Volume1, material=ALUMINIUM\n')
    string.insert(7,'*STEP\n*STATIC\n*dload\nVolume1,centrif,5484251.510,0.,0.,0.,1.,0.,0.\n')
    string.insert(8,'*EL FILE\nS\n*NODE FILE\nU\n*END STEP')
    # print(string)
    # print(list_file)


    with open('Static_analysis.inp','w') as f:
        for name in string:
            f.write(name)


if __name__ == "__main__":
    #os.system("powershell.exe "+os.getcwd()+u"\\script\\clean.ps1")
    runGmsh()
    inpManipulation()

