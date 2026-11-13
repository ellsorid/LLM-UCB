param(
    [Parameter(Mandatory)][string]$Run,
    [Parameter(Mandatory)][string]$KeyFile,
    [string]$Python = 'python',
    [int]$Generations = 100,
    [int]$Population = 1000,
    [int]$Pending = 3000,
    [string]$Runtime = (Join-Path $PSScriptRoot '../../work/lohc_operator_ac_runtime_01')
)

$ErrorActionPreference = 'Stop'
$env:PYTHONHASHSEED = '11'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
$env:LOKY_MAX_CPU_COUNT = '2'
$Run = [IO.Path]::GetFullPath($Run)
$KeyFile = (Resolve-Path -LiteralPath $KeyFile).Path
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$Runtime = (Resolve-Path -LiteralPath $Runtime).Path
$requiredAssets = @(
    'QM9-LOHC_new_molecules.csv', 'random_start_pop.csv',
    'fragments_insert_atom.csv', 'fragments_group_wise.csv', 'SA_Score/fpscores.pkl.gz'
) + (1..5 | ForEach-Object { "model_checkpoints/fold${_}_best.pth" })
$missingAssets = @($requiredAssets | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Runtime $_) -PathType Leaf) })
if ($missingAssets.Count) {
    throw "Provide these local files under ${Runtime}: $($missingAssets -join ', ')"
}
$Worker = Join-Path $PSScriptRoot 'worker.py'
$COutput = Join-Path $Run 'llm'
& $Python -u $Worker --out $COutput --runtime $Runtime --key-file $KeyFile `
    --generations $Generations --population $Population --gnn-pending $Pending
if ($LASTEXITCODE -ne 0) { throw "C failed. See $COutput." }
