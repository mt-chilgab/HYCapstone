$RmVimSwapsList = (".*.swp", "*~*")
$RmFCADList = ("*.FCStd*", "*.step")
$RmCCXList = ("*.cvg", "*.dat", "*.inp", "*.sta", "*.frd", "*.out", "*.nam")
$RmUpdFileList = ("parameters_upd.txt")

$RmUn = Read-Host "Remove option (1: vim swps and edit histories / 2: FreeCAD related(FCStd, STP) / 3: parameters_upd.txt / 4: CCX )"

if($RmUn -match "1"){ $RmList = $RmList + $RmVimSwapsList }
if($RmUn -match "2"){ $RmList = $RmList + $RmFCADList }
if($RmUn -match "3"){ $RmList = $RmList + $RmUpdFileList }
if($RmUn -match "4"){ $RmList = $RmList + $RmCCXList }

$RmList | ForEach-Object { if(Test-Path ".\$_") {rm $_ } }
