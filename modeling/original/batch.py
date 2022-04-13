import subprocess

# exec(open("text.py").read()) ### parameter를 txt파일로부터 list형식으로 받아오는 것 ### 일단은 생략

### subprocess 모듈을 사용해 FreeCAD.exe로 box.py를 실행시킴
proc1=subprocess.call(['C:\\Program Files\\FreeCAD 0.19\\bin\\FreeCAD.exe', 'C:\\Users\\newton\\Desktop\\RadialTurboMacrosFreeCAD-main\\1_CentrifugalMeridionalBezier.py', 'C:\\Users\\newton\\Desktop\\RadialTurboMacrosFreeCAD-main\\2_CentrifugalCamberlineSurface.py', 'C:\\Users\\newton\\Desktop\\RadialTurboMacrosFreeCAD-main\\3_CentrifugalImpeller3D.py'])
# ### Beam.iges 파일로부터 파이썬에 있는 gmsh패키지를 이용해 Beam.inp파일 뽑아내기 (자동 meshing)
# proc2=subprocess.run(['C:\\Users\\newton\\AppData\\Local\\Programs\\Python\\Python310\\python.exe','C:\\Users\\newton\\Desktop\\Auto\\g_meshing.py'])
# ### Beam.inp파일에서 Line, Surface ELEMENT 없애주기 - Line, Surface ElEMENT가 포함되어 있으면
# proc3=subprocess.run(['C:\\Users\\newton\\AppData\\Local\\Programs\\Python\\Python310\\python.exe','C:\\Users\\newton\\Desktop\\Auto\\del_El.py'])
# ### 계산하고 data파일들 만들어주기
# proc4=subprocess.run(['C:\\Program Files (x86)\\bConverged\\CalculiX\\bin\\ccx.bat','C:\\Users\\newton\\Desktop\\Auto\\Beam_static'])
# proc4=subprocess.run(['C:\\Program Files (x86)\\bConverged\\CalculiX\\ccx\\ccx.exe','C:\\Users\\newton\\Desktop\\Auto\\Beam_static'])

### ccx.bat에서 돌려줘야 제대로 solve 됨
