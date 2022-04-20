param([String]$RmUn="12")

$RmVimSwapsList = (".*.swp", "*~*")
$RmFEAList = ("*.FCStd*", "*.inp", "*.nam", "*.frd", "*.dat", "*.cvg", "*.sta", "*.out")
$RmSTEPList = ("*.step")

if($RmUn -match "1"){ $RmList = $RmList + $RmVimSwapsList }
if($RmUn -match "2"){ $RmList = $RmList + $RmFEAList }
if($RmUn -match "3"){ $RmList = $RmList + $RmSTEPList }
 
$RmList | ForEach-Object { if(Test-Path ".\$_") {rm $_ } }
