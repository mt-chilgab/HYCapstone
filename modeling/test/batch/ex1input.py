with open('infor.txt', 'r') as f:
    list_file = []
    for line in f:
        list_file.append(line)
f.close()

string1=[]
string1.insert(0,'*INCLUDE, INPUT=impeller2.inp \n*INCLUDE, INPUT=fixed.nam \n\n*BOUNDARY\nNfix,1,3\n')
string1.insert(1,'*Material, name=STEEL\n')
string1.insert(2,'*Elastic\n')
string1.insert(3,list_file[0])
string1.insert(4,'*Density\n')
string1.insert(5,list_file[1])
string1.insert(6,'*Solid Section, elset=Volume1, material=steel\n')
string1.insert(7,'*STEP,nlgeom\n*STATIC\n*dload\nVolume1,centrif,98696,0.,0.,0.,1.,0.,0.\n')
string1.insert(8,'*EL FILE\nS\n*NODE FILE\nU\n*END STEP')
# print(string)
# print(list_file)


string2=[]
string2.insert(0,'*INCLUDE, INPUT=impeller2.inp \n\n')
string2.insert(1,'*Material, name=STEEL\n')
string2.insert(2,'*Elastic\n')
string2.insert(3,list_file[0])
string2.insert(4,'*Density\n')
string2.insert(5,list_file[1])
string2.insert(6,'*Solid Section, elset=Volume1, material=steel\n')
string2.insert(7,'*STEP\n*frequency\n 20\n')
string2.insert(8,'*EL FILE\nS\n*NODE FILE\nU\n*END STEP')
# print(string)
# print(list_file)

with open('Static_analysis.inp','w',encoding='UTF-8') as f:
     for name in string1:
         f.write(name)

with open('Modal_analysis.inp','w',encoding='UTF-8') as f:
     for name in string2:
         f.write(name)
