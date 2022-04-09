cd $PSScriptRoot
echo "Script is being launched at : $pwd`n"
$pwd_ = $pwd.Path

#Returns $Cmd version and whether it fits version requirement if $Cmd runs, or it returns $null
function Version-Check {
	param ( [String]$Cmd )
	try { 
		$Ver = (Get-Command -ErrorAction Stop $Cmd).Version
		if($Ver.Major -eq 3) {
			if($Ver.Minor -gt 8){ return [System.Tuple]::Create([String]$Ver.Major+"."+[String]$Ver.Minor, $True) }
			elseif($Ver.Minor -eq 8 -and [Regex]::Matches($Ver.Build, '^\d').Value -as [Int32] -ge 7) { 
				return [System.Tuple]::Create([String]$Ver.Major+"."+[String]$Ver.Minor, $True) 
			}
			else { return [System.Tuple]::Create([String]$Ver.Major+"."+[String]$Ver.Minor, $False) }
		}
		else { return $False }	
	}
	catch [System.Management.Automation.CommandNotFoundException]{ return $null }
}

$PyVer = Version-Check("py")
$PythonVer = Version-Check("python")

echo "`nInstalling virtualenv"
pip install virtualenv

$VenvLocation = Read-Host "`nLocation to create new venv "
if(!(Test-Path $VenvLocation)) { mkdir $VenvLocation }

$VenvName = Read-Host "`nName of new venv "
cd $VenvLocation

if(Test-Path "$VenvLocation\$VenvName"){
	echo "`nFolder already exists."
	cd "$VenvLocation\$VenvName\Scripts"
	.\activate.ps1
	cd $pwd_
	return
}
Invoke-Command -ArgumentList @($PythonVer.Item1, $VenvName) -ScriptBlock {py -$($args[0]), -m virtualenv $args[1]}

echo "`nNow activating..."
cd "$VenvLocation\$VenvName"
.\Scripts\activate.ps1

echo "`nDownloading dependency... (gmsh, subprocess)"
pip install gmsh
cd $pwd_
