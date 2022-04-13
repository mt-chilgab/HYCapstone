import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import numpy as np

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
ndv = 2
m = 20
sp = samplingplan(ndv)
sample_list = sp.optimallhc(m)
print(sample_list)

lb = [10, 10]
ub = [1000, 1000]

mapped_list = np.zeros((m,ndv))

for i in range(m):
    samp le = sample_list[i]
    for j in range(ndv):
        mapped = (ub[j]-lb[j])*sample[j]+lb[j]
        mapped_list[i][j] = mapped
print(mapped_list)

str_list = []
for i in range(m):
    mapped_line = mapped_list[i]    
    str_line = []
    for j in range(ndv):
        str_line.append(str(mapped_line[j]))
    str_line2 = ','.join(str_line)+'\n'
    str_list.append(str_line2)

with open('valueset.txt','w',encoding='UTF-8') as f:
    for i in range(m):
        for value in str_list:
            f.write(value)

# # Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin
y = testfun(mapped_list)/1000

# # Now that we have our initial data, we can create an instance of a Kriging model
k = kriging(mapped_list, y, testfunction=testfun, name='simple')  
k.train()

# # Now, five infill points are added. Note that the model is re-trained after each point is added
numiter = 5  
for i in range(numiter):  
    # print 'Infill iteration {0} of {1}....'.format(i + 1, numiter)
    newpoints = k.infill(1)
    for point in newpoints:
        k.addPoint(point, testfun(point)[0])
    k.train()

# # And plot the results
k.plot()
