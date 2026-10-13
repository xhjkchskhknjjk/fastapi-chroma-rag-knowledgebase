<#
.SYNOPSIS
RAG项目docker部署脚本
#>
Write-Host "===== 开始部署 rag-service =====" -ForegroundColor Cyan

# 容器名称、镜像tag
$containerName = "rag-service"
$imageTag = "fastapi-rag:day13"

# 停止旧容器，如果存在
$existContainer = docker ps -a --filter "name=^/$containerName$" --format '{{.Names}}'
if ($existContainer -eq $containerName)
{
    Write-Host "停止旧容器 $containerName"
    docker stop $containerName
    Write-Host "删除旧容器 $containerName"
    docker rm $containerName
}
else
{
    Write-Host "不存在旧容器，跳过停止删除步骤"
}

Write-Host "开始构建镜像 $imageTag" -ForegroundColor Cyan
docker build -t $imageTag .
if ($LASTEXITCODE -ne 0)
{
    Write-Host "镜像构建失败！终止部署" -ForegroundColor Red
    exit 1
}

Write-Host "启动新容器 $containerName" -ForegroundColor Cyan
# 只挂载数据目录，绝不挂载 /app：
#   挂 /app 会用宿主卷覆盖镜像里构建好的代码，导致容器实际运行旧版本，
#   表现为"新功能怎么部署都不生效"。会话库已挪到 /app/data。
docker run -d --name $containerName `
  -p 8000:8000 `
  --env-file .\.env `
  -v rag_uploads:/app/uploads `
  -v rag_chroma:/app/chroma_db `
  -v rag_logs:/app/logs `
  -v rag_session_data:/app/data `
  $imageTag

if ($LASTEXITCODE -ne 0)
{
    Write-Host "❌容器启动失败！" -ForegroundColor Red
    exit 1
}

Write-Host "等待服务就绪..." -ForegroundColor Cyan
# 用 HttpClient 而不是 Invoke-WebRequest：后者在本机可能因 IE 引擎未初始化而抛错，
# 导致服务其实已就绪却被判定为未就绪。
Add-Type -AssemblyName System.Net.Http
$http = New-Object System.Net.Http.HttpClient
$http.Timeout = [TimeSpan]::FromSeconds(5)
$baseUrl = "http://127.0.0.1:8000"
$ready = $false
for ($i = 0; $i -lt 90; $i++)
{
    Start-Sleep -Seconds 1
    try
    {
        $resp = $http.GetAsync("$baseUrl/docs").GetAwaiter().GetResult()
        if ([int]$resp.StatusCode -eq 200) { $ready = $true; break }
    }
    catch { }
}

if (-not $ready)
{
    Write-Host "❌服务在90秒内未就绪，请执行 docker logs $containerName 排查" -ForegroundColor Red
    exit 1
}

# ---------- 部署后校验：确认容器跑的确实是本次构建的代码 ----------
Write-Host "校验容器内代码版本..." -ForegroundColor Cyan
$hostMainHash = (Get-FileHash -Algorithm SHA256 .\main.py).Hash
$containerMainHash = (docker exec $containerName sha256sum /app/main.py 2>$null | ForEach-Object { ($_ -split ' ')[0] }).ToUpper()

if ($containerMainHash -ne $hostMainHash)
{
    Write-Host "❌容器内 main.py 与仓库当前版本不一致！" -ForegroundColor Red
    Write-Host "   宿主: $hostMainHash"
    Write-Host "   容器: $containerMainHash"
    Write-Host "   通常意味着有卷挂载遮蔽了 /app，请检查 docker run 的 -v 参数" -ForegroundColor Red
    exit 1
}
Write-Host "   main.py 版本一致 ✅" -ForegroundColor Green

Write-Host "校验鉴权行为..." -ForegroundColor Cyan
$unauth = $http.GetAsync("$baseUrl/doc/list").GetAwaiter().GetResult()
$unauthCode = [int]$unauth.StatusCode
if ($unauthCode -eq 401)
{
    Write-Host "   无token访问被拒(401) ✅" -ForegroundColor Green
}
elseif ($unauthCode -eq 200)
{
    Write-Host "❌ 无token访问 /doc/list 竟然成功了，鉴权未生效！" -ForegroundColor Red
    exit 1
}
else
{
    Write-Host "⚠️ 无token访问返回 $unauthCode（预期401），请人工确认" -ForegroundColor Yellow
}

$token = (Select-String -Path .\.env -Pattern '^API_ACCESS_TOKEN=(.+)$').Matches[0].Groups[1].Value.Trim()
if ($token)
{
    $req = New-Object System.Net.Http.HttpRequestMessage('GET', "$baseUrl/doc/list")
    $req.Headers.Add('X-Access-Token', $token)
    $okResp = $http.SendAsync($req).GetAwaiter().GetResult()
    if ([int]$okResp.StatusCode -eq 200)
    {
        Write-Host "   带token访问成功(200) ✅" -ForegroundColor Green
    }
    else
    {
        Write-Host "❌ 带token访问返回 $([int]$okResp.StatusCode)，请检查 .env 中的 API_ACCESS_TOKEN" -ForegroundColor Red
        exit 1
    }
}
$http.Dispose()

Write-Host ""
Write-Host "✅容器启动完成并通过校验！" -ForegroundColor Green
Write-Host "查看日志命令：docker logs $containerName"
Write-Host "访问文档地址：http://127.0.0.1:8000/docs"
