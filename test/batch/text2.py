import numpy as np

c = np.arange(10)

#del __objs__
with open('impeller.inp', 'r') as f:
    list_file = []
    for line in f:
        list_file.append(line)
f.close()

found1 = list_file.index('*ELEMENT, type=CPS3, ELSET=Surface64\n')
found2 = list_file.index('*ELEMENT, type=CPS3, ELSET=Surface65\n')
found3 = list_file.index('*ELEMENT, type=T3D2, ELSET=Line1\n')
found4 = list_file.index('*ELEMENT, type=C3D4, ELSET=Volume1\n')

comp1=list_file[found1+1:found2-1]
# comp2=list_file[found2+1:found4-1]



del list_file[found3:found4]
del list_file[2]
list_file.insert(2,'*NODE, NSET=Nall\n')


#####
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
    
print(b1)

#b=np.reshape(b,(sz1,sz2-1))
for i in range(len(b1)):
    b1[i] = str(b1[i])
    
string1=[',\n'.join(b1)]
#b= b.drop(index=0, axis=0)
###########

# a2=[]
# b2=[]
# sz3 = len(comp2)
# for i in comp2:
#     a2=i.split(',')
#     sz4=len(a2)
#     for j in a2:
#         b2 = b2 + [int(j)]
    
# for i in range(1,sz3+1):
#     del b2[(sz4-1)*(i-1)]
    
# #print(b2)

# #b=np.reshape(b,(sz1,sz2-1))
# for i in range(len(b2)):
#     b2[i] = str(b2[i])
    
# string2=[',\n'.join(b2)]

#########

string1.insert(0,'*NSET,NSET= Nfix\n')
# string2.insert(0,'*NSET,NSET=LOAD\n')


with open('impeller2.inp','w',encoding='UTF-8') as f:
     for name in list_file:
         f.write(name)


with open('fixed.nam','w',encoding='UTF-8') as f:
     for name in string1:
         f.write(name)
         
# with open('ex1load.nam','w',encoding='UTF-8') as f:
#      for name in string2:
#          f.write(name)

#print(comp1)
#print(comp2)



#a = [float(list_file[i]), float(list_file[1]), float(list_file[2])]
#print(a) ###
