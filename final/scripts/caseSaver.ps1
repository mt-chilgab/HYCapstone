param([String]$CaseName)

if(-Not (Test-Path ".\cases")){ (mkdir .\cases) | Out-Null }
if(-Not (Test-Path ".\cases\$CaseName")){ (mkdir .\cases\$CaseName) | Out-Null } 
if(-Not ($(ls .\cases\$CaseName) -eq $null)){
	$DirList = $(ls .\cases\$CaseName) | ForEach-Object { if($_.Mode -match "d"){ $_ } }
	if($DirList -eq $null){ 
		$MvDir = ".\cases\$CaseName\1" 
		(mkdir $MvDir) | Out-Null
	}
	else{
		$MaxNum = [System.Int32]($DirList | Measure-Object -Maximum).Maximum + [System.Int32]1
		$MvDir = ".\cases\$CaseName\$MaxNum"
		(mkdir $MvDir) | Out-Null
	}
}
else{ $MvDir = ".\cases\$CaseName" }

$MvList = (ls).Name | ForEach-Object { if($($_) -match "\.step$|\.inp$|\.nam$|\.frd$|\.dat$|\.cvg$|\.sta$|\.out$"){ $_ } }
$MvList | ForEach-Object { mv $_ $MvDir }
