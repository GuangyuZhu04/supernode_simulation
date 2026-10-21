param([string]$InputPath,[string]$OutputDir)
$ErrorActionPreference='Stop'
$taskApp=$null
$taskDeck=$null
try {
    [void][IO.Directory]::CreateDirectory($OutputDir)
    $taskApp=New-Object -ComObject KWPP.Application
    $taskDeck=$taskApp.Presentations.Open($InputPath,-1,0,0)
    if($taskDeck.Slides.Count -ne 8){throw 'Expected eight slides.'}
    for($i=1;$i -le 8;$i++){
        $taskDeck.Slides.Item($i).Export((Join-Path $OutputDir ('slide-{0:D2}.png' -f $i)),'PNG',1600,900)
        Write-Output ('Rendered slide '+$i)
    }
} finally {
    if($null -ne $taskDeck){try{$taskDeck.Saved=-1;$taskDeck.Close()}catch{}}
    if($null -ne $taskDeck){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskDeck)}
    if($null -ne $taskApp){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskApp)}
}
