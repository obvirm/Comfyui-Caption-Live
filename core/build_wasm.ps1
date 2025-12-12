# Emscripten WASM Build Script
# Run this after emsdk is installed and activated

Write-Host "Building C++ Caption Engine for WebAssembly..." -ForegroundColor Cyan

# Check if emcc is available
$emcc = Get-Command emcc -ErrorAction SilentlyContinue
if (-not $emcc) {
    Write-Host "emcc not found. Please run:" -ForegroundColor Red
    Write-Host "   emsdk install latest" -ForegroundColor Yellow
    Write-Host "   emsdk activate latest" -ForegroundColor Yellow
    Write-Host "   emsdk_env.bat" -ForegroundColor Yellow
    exit 1
}

# Ensure we are in the script's directory
Set-Location $PSScriptRoot

# Create build directory
$buildDir = "build_wasm"
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

Push-Location $buildDir

# Run cmake with Emscripten toolchain
Write-Host "Configuring CMake with Emscripten..." -ForegroundColor Green
emcmake cmake .. -DBUILD_WASM=ON -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Build
Write-Host "Building WASM module..." -ForegroundColor Green
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Copy output to web directory
$outputDir = Join-Path ".." "web" "wasm"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$wasmFiles = @(
    (Join-Path $buildDir "bin" "caption_engine_wasm.js"),
    (Join-Path $buildDir "bin" "caption_engine_wasm.wasm"),
    (Join-Path $buildDir "bin" "caption_engine_wasm.data")
)

foreach ($file in $wasmFiles) {
    if (Test-Path $file) {
        Copy-Item $file $outputDir -Force
        Write-Host "Copied: $file" -ForegroundColor Green
    }
}

Write-Host "WASM build complete!" -ForegroundColor Cyan
Write-Host "Output directory: $outputDir" -ForegroundColor White
