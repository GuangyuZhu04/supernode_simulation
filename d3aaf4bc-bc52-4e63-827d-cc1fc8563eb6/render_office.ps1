param([string]$InputPath,[string]$OutputDir,[int]$SlideNumber=2)
$ErrorActionPreference='Stop'
$taskApp=$null
$taskDeck=$null
try {
    [void][IO.Directory]::CreateDirectory($OutputDir)
    $taskApp=New-Object -ComObject KWPP.Application
    $taskDeck=$taskApp.Presentations.Open($InputPath,-1,0,0)
    if($taskDeck.Slides.Count -ne 6){throw 'Expected six slides.'}
    $taskDeck.Slides.Item($SlideNumber).Export((Join-Path $OutputDir ('slide-{0:D2}.png' -f $SlideNumber)),'PNG',1600,900)
    Write-Output ('Rendered slide '+$SlideNumber)
} finally {
    if($null -ne $taskDeck){try{$taskDeck.Saved=-1;$taskDeck.Close()}catch{}}
    if($null -ne $taskDeck){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskDeck)}
    if($null -ne $taskApp){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskApp)}
}
