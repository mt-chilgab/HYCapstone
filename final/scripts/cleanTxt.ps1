$TxtFiles = (ls).Name | ForEach-Object { if($($_) -match "\.txt$"){ $_ } }
$TxtFiles | ForEach-Object { if($($_) -ne "parameters.txt"){ rm $_ } }
