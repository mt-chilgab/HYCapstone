import gmsh
import math
import os
import sys

gmsh.initialize()

gmsh.model.add("impeller")

# Load a STEP file (using `importShapes' instead of `merge' allows to directly retrieve the tags of the highest dimensional imported entities):
path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path,'gmsh_std_Option.opt'))
v = gmsh.merge(os.path.join(path,'impeller.step'))

### 여기서 매쉬 설정 더하려면 t6: Transfinite meshes 참고
gmsh.model.mesh.generate(3)

gmsh.write("impeller.inp")

# Launch the GUI to see the results:
# if '-nopopup' not in sys.argv:
gmsh.fltk.run()

gmsh.finalize()

#phsical Surface 
