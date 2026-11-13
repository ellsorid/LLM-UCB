param(
    [Parameter(Mandatory)][string]$Run,
    [Parameter(Mandatory)][string]$KeyFile,
    [string]$Python = 'python',
    [int]$Generations = 100,
    [int]$Population = 1000,
    [int]$Pending = 3000,
    [switch]$MatrixScoring,
    [switch]$PercentileAgents,
    [switch]$M3OS,
    [switch]$M3OSFull,
    [switch]$MatrixSP,
    [switch]$OnlineMemory
)
$ErrorActionPreference = 'Stop'
if (($M3OS -and $M3OSFull) -or (($M3OS -or $M3OSFull) -and $PercentileAgents)) { throw 'Select one search experiment' }
if ($MatrixSP -and -not $M3OSFull) { throw 'MatrixSP requires M3OSFull' }
if ($OnlineMemory -and -not $MatrixSP) { throw 'OnlineMemory requires MatrixSP' }
$GraphExperiment = $M3OS -or $M3OSFull
$env:PYTHONHASHSEED = '11'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
$env:LOKY_MAX_CPU_COUNT = '2'
$Run = [IO.Path]::GetFullPath($Run)
$KeyFile = (Resolve-Path -LiteralPath $KeyFile).Path
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$Runtime = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../work/lohc_operator_ac_runtime_01')).Path
$requiredAssets = @(
    'QM9-LOHC_new_molecules.csv', 'random_start_pop.csv',
    'fragments_insert_atom.csv', 'fragments_group_wise.csv', 'SA_Score/fpscores.pkl.gz'
) + (1..5 | ForEach-Object { "model_checkpoints/fold${_}_best.pth" })
$missingAssets = @($requiredAssets | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Runtime $_) -PathType Leaf) })
if ($missingAssets.Count) {
    throw "Provide these local files under ${Runtime}: $($missingAssets -join ', ')"
}
New-Item -ItemType Directory -Path $Run | Out-Null
$Book = Join-Path $Run 'rule_book.json'
$BookFlag = '--rule-book'
if ($MatrixScoring -or $PercentileAgents -or $GraphExperiment) {
    $Book = Join-Path $Run 'matrix_book.json'
    $BookFlag = '--matrix-book'
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'seed_data/matrix_book.json') -Destination $Book
} else {
    & $Python (Join-Path $PSScriptRoot 'rule_steering.py') (Join-Path $PSScriptRoot 'cluster_rule_experiment_01') $Book
    if ($LASTEXITCODE -ne 0) { throw 'Could not reconstruct the rule book' }
}
$Experiment = 'deepseek_rule_delta_operator_prior'
$PercentileArguments = @()
if ($M3OS) {
    $Experiment = 'm3os_top1_vs_llm_top5'
    $PercentileArguments = @('--graph-m3os')
}
if ($M3OSFull) {
    $Experiment = 'native_vs_m3os_full'
    $PercentileArguments = @('--full-m3os')
}
if ($MatrixSP) { $PercentileArguments += '--matrix-sp' }
if ($OnlineMemory) { $PercentileArguments += '--online-memory' }
if ($PercentileAgents) {
    $Experiment = 'live_scoring_and_search_agents'
    $Reference = Join-Path $Run 'comparison_percentiles.json'
    & $Python (Join-Path $PSScriptRoot 'percentile_selection.py') $Reference
    if ($LASTEXITCODE -ne 0) { throw 'Could not freeze the evaluation reference' }
    $PercentileArguments = @('--percentile-reference',$Reference)
}
@{control=$(if ($M3OS) {'molecule_ucb_top1'} else {'native_ucb'}); experiment=$Experiment; seed=11; generations=$Generations;
  population=$Population; gnn_pending=$Pending; operator_llm=(-not $GraphExperiment); operator_request_limit=$(if ($GraphExperiment) {0} else {8});
  parent_request_limit=$(if ($OnlineMemory) {[Math]::Max(0,$Generations-4)} elseif ($GraphExperiment) {8} else {0}); matrix_scoring=($MatrixScoring -or $PercentileAgents -or $GraphExperiment);
  total_agent_request_limit=$(if ($OnlineMemory) {1+6*[Math]::Max(0,$Generations-4)} elseif ($M3OSFull) {41} elseif ($M3OS) {8} else {0}); online_memory=[bool]$OnlineMemory;
  scoring_agent_request_limit=$(if ($PercentileAgents) {4} else {0}); live_scoring_c=[bool]$PercentileAgents; matrix_sp=[bool]$MatrixSP} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'ac_experiment.json') -Encoding utf8
$Worker = Join-Path $PSScriptRoot 'worker.py'
$Common = @('-u',$Worker,'--qd-ac','--runtime',$Runtime,'--generations',$Generations,
    '--population',$Population,'--gnn-pending',$Pending,$BookFlag,$Book) + $PercentileArguments
$Processes = foreach ($Arm in @('native','llm')) {
    $Arguments = $Common + @('--arm',$Arm,'--out',(Join-Path $Run $Arm))
    if ($Arm -eq 'llm') { $Arguments += @('--key-file',$KeyFile,'--operator-requests',8) }
    Start-Process -FilePath $Python -WindowStyle Hidden -PassThru `
        -ArgumentList ($Arguments | ForEach-Object { '"' + $_ + '"' }) `
        -RedirectStandardOutput (Join-Path $Run "$Arm.log") -RedirectStandardError (Join-Path $Run "$Arm.err.log")
}
$Processes | Select-Object -ExpandProperty Id
$Processes | Wait-Process
if ($Processes | Where-Object { $_.ExitCode -ne 0 }) { throw "A worker failed. See $Run." }
if ($GraphExperiment -and ($Processes.Count -eq 2)) {
    & $Python (Join-Path $PSScriptRoot 'compare_pooled_percentiles.py') --run $Run
    if ($LASTEXITCODE -ne 0) { throw "Could not compare the pooled candidates. See $Run." }
}
