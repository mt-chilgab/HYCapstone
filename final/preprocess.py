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

    ### 여기서 매쉬 설정 더하려면 t6: Transfinite meshes 참고
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

    found1 = list_file.index('*ELEMENT, type=CPS6, ELSET=Surface'+str(NumberSurface)+'\n')
    # found2 = list_file.index('*ELEMENT, type=CPS6, ELSET=Surface6\n')
    found3 = list_file.index('*ELEMENT, type=T3D3, ELSET=Line1\n') ### ELEMENT 시작줄
    found4 = list_file.index('*ELEMENT, type=C3D10, ELSET=Volume1\n') ### Volume Element 시작줄

    comp1=list_file[found1+1:found4-1] ### -> Surface5에 대한 element 들
    # comp2=list_file[found2+1:found4-1] ### -> Surface6에 대한 element 들

    del list_file[found3:found4] ### line & surface element 들은 다 지우기
    del list_file[2] ### HEADING 줄 지우기
    list_file.insert(2,'*NODE, NSET=Nall\n') ### ? 아마도 문자열에 저거 추가하는거 인 듯


    #####
    a1=[]
    b1=[]
    sz1 = len(comp1)
    for i in comp1:
        a1=i.split(',')###,를 기준으로 나눠서 List에 저장 [#Element, Node1, Node2, Node3]
        sz2=len(a1)### 4
        for j in a1:
            b1 = b1 + [int(j)]### b1에 값들 append해주기

    for i in range(1,sz1+1):
        del b1[(sz2-1)*(i-1)]### #Element 지워주기?

    # print(b1)

    #b=np.reshape(b,(sz1,sz2-1))
    for i in range(len(b1)):
        b1[i] = str(b1[i])
    ### b1에 있는 값들을 str으로 저장하기?

    string1=[',\n'.join(b1)]### ,\n 추가된 string으로 변환 (조건 넣어주는 것 같음)
    #b= b.drop(index=0, axis=0)
    ###########

    # a2=[]
    # b2=[]
    # # list2=[]
    # sz3 = len(comp2)
    # for i in comp2:
    #     a2=i.split(',')
    #     sz4=len(a2)
    #     for j in a2:
    #         b2 = b2 + [int(j)]
        
    # for i in range(1,sz3+1):
    #     del b2[(sz4-1)*(i-1)]
        

    # for i in range(len(b2)):
    #     b2[i] = str(b2[i])

    # set2 = set(b2)
    # string2=[',\n'.join(set2)]

    string1.insert(0,'*NSET,NSET=Nfix\n')### string1 은 Fixed boundary SET로 설정
    # string2.insert(0,'*NSET,NSET=LOAD\n')### string2 는 Load SET로 설정


    with open('impeller2.inp','w',encoding='UTF-8') as f:
        for name in list_file:
            f.write(name)
        f.close()

    with open('fixed.nam','w',encoding='UTF-8') as f:
        for name in string1:
            f.write(name)
        f.close()
             
    # with open('ex1load.nam','w',encoding='UTF-8') as f:
    #     for name in string2:
    #         f.write(name)
    #     f.close()

    # Read material properties from parameters.txt
    searchKeyword = lambda keyword: False if re.search(r'^matprop[\s\t]+\w{4,6}[\s\t]+'+str(keyword)+r'[\s\t]*(?==)', line) is None else True
    getValueString = lambda line: line.split("=")[1].strip(" ")
    with open('parameters.txt', 'r',encoding='UTF-8') as f:
        list_file = []
        for line in f:
            if searchKeyword("E") or searchKeyword("rho"):
                list_file.append(getValueString(line))
        f.close()

    string=[]
    string.insert(0,'*INCLUDE, INPUT=impeller2.inp \n*INCLUDE, INPUT=fixed.nam \n\n*BOUNDARY\nNfix,1,3\n')
    string.insert(1,'*Material, name=STEEL\n')
    string.insert(2,'*Elastic\n')
    string.insert(3,list_file[0])
    string.insert(4,'*Density\n')
    string.insert(5,list_file[1])
    string.insert(6,'*Solid Section, elset=Volume1, material=steel\n')
    string.insert(7,'*STEP\n*STATIC\n*dload\nVolume1,centrif,986.96,0.,0.,0.,1.,0.,0.\n')
    string.insert(8,'*EL FILE\nS\n*NODE FILE\nU\n*END STEP')
    # print(string)
    # print(list_file)


    with open('Static_analysis.inp','w',encoding='UTF-8') as f:
        for name in string:
            f.write(name)
        f.close()


if __name__ == "__main__":
    os.system("powershell.exe "+os.getcwd()+u"\\script\\clean.ps1")
    runGmsh()
    inpManipulation()

