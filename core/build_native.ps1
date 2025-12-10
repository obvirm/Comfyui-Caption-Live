#!/usr/bin/env pwsh
# Build script for Caption Engine native desktop
# Usage: .\build_native.ps1 [-Release] [-Vulkan] [-CUDA] [-Clean]

param(
    [switch]$Release,
    [switch]$Vulkan,
    [switch]$CUDA,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$BuildDir = "build"
$BuildType = if ($Release) { "Release" } else { "Debug" }

Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host "  Caption Engine Native Build" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Clean if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "`n[1/4] Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Configure CMake options
$CMakeArgs = @(
    "-DBUILD_WASM=OFF",
    "-DBUILD_PYTHON=ON",
    "-DBUILD_TESTS=OFF",
    "-DENABLE_UNIFIED_PIPELINE=ON",
    "-DCMAKE_BUILD_TYPE=$BuildType"
)

# GPU backends
if ($Vulkan) {
    $CMakeArgs += "-DENABLE_VULKAN=ON"
    Write-Host "[GPU] Vulkan enabled" -ForegroundColor Green
}

if ($CUDA) {
    $CMakeArgs += "-DENABLE_CUDA=ON"
    Write-Host "[GPU] CUDA enabled" -ForegroundColor Green
}

Write-Host "`n[2/4] Configuring CMake ($BuildType)..." -ForegroundColor Yellow
Push-Location $BuildDir
try {
    & cmake .. $CMakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
    
    Write-Host "`n[3/4] Building..." -ForegroundColor Yellow
    & cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    
    Write-Host "`n[4/4] Copying Python modules..." -ForegroundColor Yellow
    $PydFiles = Get-ChildItem -Path "." -Recurse -Filter "*.pyd"
    foreach ($pyd in $PydFiles) {
        $dest = Join-Path ".." ".." $pyd.Name
        Copy-Item $pyd.FullName $dest -Force
        Write-Host "  -> $($pyd.Name)" -ForegroundColor DarkGray
    }
}
finally {
    Pop-Location
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "  Output: core/$BuildDir" -ForegroundColor DarkGray
Write-Host ""
