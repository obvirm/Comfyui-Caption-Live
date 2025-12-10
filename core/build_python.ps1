# Python Extension Build Script

Write-Host "🐍 Building C++ Caption Engine for Python..." -ForegroundColor Cyan

# Ensure we are in the script's directory
Set-Location $PSScriptRoot

# Create build directory
$buildDir = "build_python"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $buildDir

# Run cmake for Native Python build
Write-Host "📦 Configuring CMake..." -ForegroundColor Green
cmake .. -DBUILD_WASM=OFF -DBUILD_PYTHON=ON -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ CMake configuration failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Build
Write-Host "🔨 Building Python module..." -ForegroundColor Green
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Copy output to root directory
# Note: CMake output might be in Release folder on Windows
$pydFiles = Get-ChildItem -Path "$buildDir" -Recurse -Filter "*.pyd"
if ($pydFiles.Count -gt 0) {
    $src = $pydFiles[0].FullName
    Copy-Item $src "..\caption_engine_py.pyd" -Force
    Write-Host "✅ Copied: $src -> caption_engine_py.pyd" -ForegroundColor Green
}
else {
    Write-Host "❌ Could not find output .pyd file in $buildDir" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Python build complete!" -ForegroundColor Cyan
