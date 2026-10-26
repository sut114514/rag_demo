$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$ragPython = Join-Path $PSScriptRoot 'ragdemo\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) { throw '缺少 ragdemo 环境，请先创建环境并安装 requirements.txt。' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env'; Write-Host '请在 .env 中填写 DeepSeek API 密钥，然后重新启动。'; exit 1 }
Write-Host '正在启动泰拉瑞亚知识库问答，默认地址 http://127.0.0.1:8000 。按 Ctrl+C 停止。'
& $ragPython 'app.py'
