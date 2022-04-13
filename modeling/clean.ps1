$RmVimSwapsList = (".*.swp", "*~*")
$RmFCADList = ("*.FCStd*", "*.step")
$RmUpdFileList = ("parameters_upd.txt")

$RmUn = Read-Host "Remove option (1: vim swps and edit histories / 2: FreeCAD related(FCStd, STP) / 3: parameters_upd.txt)"

if($RmUn -match "1"){ $RmList = $RmList + $RmVimSwapsList }
if($RmUn -match "2"){ $RmList = $RmList + $RmFCADList }
if($RmUn -match "3"){ $RmList = $RmList + $RmUpdFileList }
 
$RmList | ForEach-Object { if(Test-Path ".\$_") {rm $_ } }
