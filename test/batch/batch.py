import os,subprocess
import sys

# subprocess 모듈을 사용해 FreeCAD.exe로 box.py를 실행시킴 ### 3_1에서 경로 변경
proc1=subprocess.call(['C:\\Users\\Grant\\AppData\\Local\\Programs\\FreeCAD 0.19\\bin\\FreeCAD.exe', os.getcwd()+'\\1_0_impeller.py'])

# 파이썬에 있는 gmsh 패키지를 이용해 inp파일 뽑아내기 (자동 meshing)
proc1=subprocess.Popen([sys.executable,os.getcwd()+'\\g_meshing.py'])

# Line, Surface ELEMENT 없애주기 - Line, Surface ElEMENT가 포함되어 있으면
proc1=subprocess.Popen([sys.executable,os.getcwd()+'\\text2.py'])
# 계산하고 data파일들 만들어주기
proc1=subprocess.Popen([sys.executable,os.getcwd()+'\\ex1input.py'])

# proc1=subprocess.run(['C:\\Users\\Grant\\Desktop\\CalculiX\\etc\\runCCX.bat',os.getcwd()+'\\Static_analysis.inp'])
proc1=subprocess.run(['C:\\Users\\Grant\\Desktop\\CalculiX\\etc\\runCCX.bat',os.getcwd()+'\\Modal_analysis.inp'])

