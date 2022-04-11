import sys, os
from io import StringIO
from contextlib import contextmanager

import re, math


@contextmanager
def silenceStdout(newStream=None):
    if newStream is None:
        newStream = StringIO()
    oldStream, sys.stdout = sys.stdout, newStream
    try:
        yield sys.stdout
    finally:
        sys.stdout = oldStream
        newStream.seek(0)


def frdManipulation():
    with open('ex1static1.frd', 'r') as f:
        list_file = []
        for line in f:
            list_file.append(line)
        f.close()

    # -4  DISP
    for line in list_file:
        find1 = re.search(r'^\s{1}-[0-9]{1}\s{2}DISP',line)
        if(find1):
            found1 = list_file.index(line)
            break

    # -4  STRESS
    for line in list_file:
        find2 = re.search(r'^\s{1}-[0-9]{1}\s{2}STRESS',line)
        if(find2):
            found2 = list_file.index(line)
            break



    # print(found)
    disp = list_file[found1:found2-2]

    stress = list_file[found2:]


    with open('disp.frd','w',encoding='UTF-8') as f:
        for sline in disp:
            f.write(sline)
        f.close()

    with open('stress.frd','w',encoding='UTF-8') as f:
        for sline in stress:
            f.write(sline)
        f.close()


# Approximate Maximum von Mises Stress with p-Norm: Considering Computational Instability
# p = 10
def maxVonMises(p):
    stress_list = []

    with open(os.getcwd()+u"\\stress.frd", mode='r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if(re.search(r'^\s{1}-1[\s0-9]{10}',line)):
                vxx=line[13:25]
                vyy=line[25:37]
                vzz=line[37:49]
                vxy=line[49:61]
                vyz=line[61:73]
                vzx=line[73:85]
                vlist=[vxx,vyy,vzz,vxy,vyz,vzx]
                stress_list.append(vlist)
        f.close()

    vonMisesList=[]
    #for i in range(1,len(stress_list)):
    #    a=(1/math.sqrt(2))*((float(stress_list[i][0])-float(stress_list[i][1])) ** 2+(float(stress_list[i][1])-float(stress_list[i][2])) ** 2+(float(stress_list[i][2])-float(stress_list[i][0])) ** 2+6*((float(stress_list[i][3])) ** 2+(float(stress_list[i][4])) ** 2+(float(stress_list[i][5])) ** 2)) ** 0.5
    #    vonMisesList.append(a)

    for stressVector in stress_list:
        stressVector = list(map(lambda s: float(s), stressVector))
        stressVector[0:3] = [math.pow(normalStress-sum(stressVector[0:3])/3, 2) for normalStress in stressVector[0:3]]
        stressVector[3:6] = [math.pow(shearStress, 2)*2 for shearStress in stressVector[3:6]]
        vonMisesList.append(math.sqrt(3*sum(stressVector)/2))
   
    maxVM = max(vonMisesList)
    vonMisesList = [math.pow(s/maxVM, p) for s in vonMisesList]

    pNorm = 0
    for i in range(len(vonMisesList)):
        pNorm += vonMisesList[i]
    pNorm = maxVM*math.pow(pNorm, 1/p)

    return pNorm


if __name__ == "__main__":
    
    with silenceStdout():
        frdManipulation()

    print(maxVonMises(10)) 
