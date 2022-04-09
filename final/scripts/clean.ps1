param([String]$RmUn="12")

$RmVimSwapsList = (".*.swp", "*~*")
$RmFEAList = ("*.FCStd*", "*.step", "*.inp", "*.nam", "*.frd", "*.dat", "*.cvg", "*.sta", "*.out")
#$RmUpdFileList = ("parameters_upd.txt")

#$RmUn = Read-Host "Remove option (1: vim swps and edit histories / 2: FreeCAD, CalculiX outputs / 3: parameters_upd.txt)"
#$RmUn = Read-Host "Remove option (1: vim swps and edit histories / 2: FreeCAD, CalculiX outputs)"
 
if($RmUn -match "1"){ $RmList = $RmList + $RmVimSwapsList }
if($RmUn -match "2"){ $RmList = $RmList + $RmFEAList }
#if($RmUn -match "3"){ $RmList = $RmList + $RmUpdFileList }
 
$RmList | ForEach-Object { if(Test-Path ".\$_") {rm $_ } }
