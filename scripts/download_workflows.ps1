$ErrorActionPreference = 'Stop'

$sourceUrl = 'https://github.com/wfcommons/pegasus-instances.git'
$sourceCommit = '813a2a7d3e7273200805e89f5475f9126d903eab'
$destination = Join-Path $PSScriptRoot '..\data\pegasus-instances'

if (Test-Path -LiteralPath $destination) {
    throw "Refusing to replace existing $destination"
}

git clone --depth 1 --branch v1.4 $sourceUrl $destination
$actualCommit = (git -C $destination rev-parse HEAD).Trim()
if ($actualCommit -ne $sourceCommit) {
    throw "Unexpected workflow revision: $actualCommit"
}

Write-Host "Downloaded workflow instances at $actualCommit"
