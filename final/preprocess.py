import gmsh
import math
import re
import os, sys
import numpy as np


def runGmsh():
    gmsh.initialize()

    gmsh.model.add("box")

    # Load a STEP file (using `importShapes' instead of `merge' allows to directly retrieve the tags of the highest dimensional imported entities):
    path = os.path.dirname(os.path.abspath(__file__))
    gmsh.merge(os.path.join(path, 'gmsh\\box.opt'))
    v = gmsh.merge(os.path.join(path, 'box.step'))

    ### 여기서 매쉬 설정 더하려면 t6: Transfinite meshes 참고
    gmsh.model.mesh.generate(3)

    gmsh.write("box.inp")

    # Launch the GUI to see the results:
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()

def inpManipulation():
    c = np.arange(10)

    #del __objs__
    with open('box.inp', 'r') as f:
        list_file = []
        for line in f:
            list_file.append(line)
    f.close()

    found1 = list_file.index('*ELEMENT, type=CPS3, ELSET=Surface5\n')
    found2 = list_file.index('*ELEMENT, type=CPS3, ELSET=Surface6\n')
    found3 = list_file.index('*ELEMENT, type=T3D2, ELSET=Line1\n') ### ELEMENT 시작줄
    found4 = list_file.index('*ELEMENT, type=C3D4, ELSET=Volume1\n') ### Volume Element 시작줄
    found5 = list_file.index('*ELEMENT, type=T3D2, ELSET=Line11\n')
    found6 = list_file.index('*ELEMENT, type=T3D2, ELSET=Line12\n')

    comp1=list_file[found1+1:found2-1] ### -> Surface5에 대한 element 들
    comp2=list_file[found2+1:found4-1] ### -> Surface6에 대한 element 들
    comp3=list_file[found5+1:found6-1]

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

    a2=[]
    b2=[]
    # list2=[]
    sz3 = len(comp2)
    for i in comp2:
        a2=i.split(',')
        sz4=len(a2)
        for j in a2:
            b2 = b2 + [int(j)]
        
    for i in range(1,sz3+1):
        del b2[(sz4-1)*(i-1)]
        

    for i in range(len(b2)):
        b2[i] = str(b2[i])

    set2 = set(b2)
    string2=[',\n'.join(set2)]


    a3=[]
    b3=[]

    sz5 = len(comp3)
    for i in comp3:
        a3=i.split(',')
        sz6=len(a3)
        for j in a3:
            b3 = b3 + [int(j)]
        
    for i in range(1,sz5+1):
        del b3[(sz6-1)*(i-1)]
        

    for i in range(len(b3)):
        b3[i] = str(b3[i])

    set3 = set(b3)
    string3=['\n'.join(set3)]

    string1.insert(0,'*NSET,NSET=Nfix\n')### string1 은 Fixed boundary SET로 설정
    string2.insert(0,'*NSET,NSET=LOAD\n')### string2 는 Load SET로 설정


    with open('box2.inp','w',encoding='UTF-8') as f:
        for name in list_file:
            f.write(name)
        f.close()

    with open('ex1fix.nam','w',encoding='UTF-8') as f:
        for name in string1:
            f.write(name)
        f.close()
             
    with open('ex1load.nam','w',encoding='UTF-8') as f:
        for name in string2:
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

    with open('ex1load.nam', 'r') as f:
        load_List = []
        for line in f:
            load_List.append(line)
        f.close()

    P=20000.0
    n=float(len(load_List)-1)
    Dload=P/n

    string=[]
    string.insert(0,'*INCLUDE, INPUT=box2.inp \n*INCLUDE, INPUT=ex1fix.nam \n*INCLUDE, INPUT=ex1load.nam \n\n*BOUNDARY\nNfix,1\nNfix,2\nNfix,3\n\n')
    string.insert(1,'*MATERIAL,NAME=MAT1\n*ELASTIC\n')
    string.insert(2,list_file[0])
    string.insert(3,'*DENSITY\n')
    string.insert(4,list_file[1])
    string.insert(5,'\n\n*SOLID SECTION,ELSET=Volume1,MATERIAL=MAT1\n\n*STEP\n*STATIC\n*CLOAD\nLOAD,2,')
    string.insert(6,str(Dload))
    string.insert(7,'\n\n*NODE PRINT,NSET=Nall\nU\n*EL PRINT,ELSET=Volume1\nS\n\n*NODE FILE\nU\n*EL FILE\nS\n*END STEP')
    # print(string)
    # print(list_file)

    with open('ex1static1.inp','w',encoding='UTF-8') as f:
        for name in string:
            f.write(name)
        f.close()  


if __name__ == "__main__":
    runGmsh()
    inpManipulation()
