if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) { 
	Start-Process PowerShell "-NoProfile -ExecutionPolicy RemoteSigned -File `"$PSCommandPath`" -opt $opt" -Verb RunAs
	exit
}

cd $PSScriptRoot
echo "Script is being launched at : $pwd`n"
$pwd_ = $pwd.Path

# Initial guess: Version 0.19
$FreeCadPath = "C:\Users\$env:UserName\AppData\Local\Programs\FreeCAD 0.19\bin"
$FreeCadVersion = Read-Host "FreeCAD Version? "

# Check FreeCAD Path while found
while(!(Test-Path $FreeCadPath)) {
	$FreeCadVersion = Read-Host "FreeCAD Version? "
	$FreeCadPath = "C:\Program Files\FreeCAD $FreeCADVersion\bin"
}

#Returns $Cmd version and whether it fits version requirement if $Cmd runs, or it returns $null
function Python-Version-Check {
	try { 
		$PyExistence = Get-Command -ErrorAction Stop "py"

		$PyList = $(py -0) | Out-String
		$MatchVersions = [Regex]::Matches($PyList, "(?<=\s\-)\d{1}.\d+(?=\-)")

		return ($MatchVersions | Where-Object { $_ -like "3.8" }).Success
	}
	catch [System.Management.Automation.CommandNotFoundException]{ return $null }
}

$PyVer = Python-Version-Check
if($PyVer){
	echo "`nInstalling virtualenv"
	py -3.8, -m pip install --upgrade pip
	py -3.8, -m pip install virtualenv
	
	$VenvLocation = Read-Host "`nLocation to create new venv "
	if(!(Test-Path $VenvLocation)) { mkdir $VenvLocation }
	
	$VenvName = Read-Host "`nName of new venv "
	cd $VenvLocation

	if(Test-Path ".\$VenvName"){
		echo "`nFolder already exists."
		cd ".\$VenvName\Scripts"
		.\activate.ps1
		cd $pwd_
		pause
		return
	}
	Invoke-Command -ArgumentList @($VenvName) -ScriptBlock {py -3.8, -m virtualenv $args[0]}

	echo "`nModifying venv/pyvenv.cfg for FreeCAD import"
	$VenvCfg = ".\$VenvName\pyvenv.cfg"

	# Changes python path to FreeCAD bin dir, by searching Python path in pyvenv.cfg with regex and replacing it.
	(Get-Content $VenvCfg) -replace '[A-Z]:.*Python\d{2,3}$?', "$FreeCadPath" | Out-File -encoding ASCII $VenvCfg

	Get-Content $VenvCfg

	echo "`nDone. Now activating..."
	cd ".\$VenvName"
	.\Scripts\activate.ps1
	
	echo "`nCopying FreeCAD modules for embedded FreeCAD usage to current venv Lib/site-packages"
	$FreeCadSPPath = "$FreeCadPath\Lib\site-packages"
	$FreeCadModPath = "$FreeCadPath\..\Mod"
	
	cd .\Lib\site-packages
	ls $FreeCadSPPath | ForEach-Object { Copy-Item -Recurse "$FreeCadSPPath\$($_.Name)" ".\$($_.Name)" }
	ls $FreeCadModPath | ForEach-Object { Copy-Item -Recurse "$FreeCadModPath\$($_.Name)" ".\$($_.Name)" }
		
	echo "`nDownloading dependency... (pyKriging, gmsh)"
	pip install pyKriging gmsh

	cd $pwd_
}
else{ echo "`nPlease install Python and Python Launcher version 3.8.x and re-launch this script.`n" }
