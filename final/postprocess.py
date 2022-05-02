import sys, os
from io import StringIO
from contextlib import contextmanager

import re, math
import itertools



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
    with open('Static_analysis.frd', 'r') as f:
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


    with open('disp.frd','w') as f:
        for sline in disp:
            f.write(sline)
        f.close()

    with open('stress.frd','w') as f:
        for sline in stress:
            f.write(sline)
        f.close()


def maxVonMises():
    stress_list = []
    with open(os.getcwd()+u"\\stress.frd", mode='r') as f:
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

    hjVonMisesList=[]
    jhVonMisesList=[]
    for i in range(1,len(stress_list)):
        a=(1/math.sqrt(2))*math.sqrt(math.pow(float(stress_list[i][0])-float(stress_list[i][1]), 2)+math.pow(float(stress_list[i][1])-float(stress_list[i][2]), 2)+math.pow(float(stress_list[i][2])-float(stress_list[i][0]), 2)+6*(math.pow(float(stress_list[i][3]),2)+math.pow(float(stress_list[i][4]),2)+math.pow(float(stress_list[i][5]),2)))
        hjVonMisesList.append(a)

    for stressVector in stress_list:
        stressVector = list(map(lambda s: float(s), stressVector))
        diagonalAverage = sum(stressVector[0:3])/3
        stressVector[0:3] = [ math.pow(normalStress-diagonalAverage, 2) for normalStress in stressVector[0:3] ]
        stressVector[3:6] = [ math.pow(shearStress, 2)*2 for shearStress in stressVector[3:6] ] 
        jhVonMisesList.append(math.sqrt(3*sum(stressVector)/2))
    
    hjMaxVM = max(hjVonMisesList)
    jhMaxVM = max(jhVonMisesList)
    
    # hjVonMisesList = [ math.pow(stress/hjMaxVM, p) for stress in hjVonMisesList ]
    # jhVonMisesList = [ math.pow(stress/jhMaxVM, p) for stress in jhVonMisesList ]
    
    # with open(os.getcwd()+u"\\comparison.txt", "a",encoding = 'UTF-8') as f:
    #     f.writelines("hj, jh: "+str(hjMaxVM*math.pow(sum(hjVonMisesList), 1/p))+"   "+str(jhMaxVM*math.pow(sum(jhVonMisesList), 1/p))+"\n")
    #     f.close

    return float(jhMaxVM)



if __name__ == "__main__":
    
    with silenceStdout():
        frdManipulation()

    print(float(maxVonMises()))
