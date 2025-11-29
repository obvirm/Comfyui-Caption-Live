import sys
import subprocess
import os
import platform
import shutil
import urllib.request

# --- CONFIGURATION ---
GITHUB_REPO = "YOUR_USERNAME/YOUR_REPO" # TODO: Update this!
VERSION = "0.1.0"
# ---------------------

def check_command(command):
    return shutil.which(command) is not None

def run_command(command, cwd=None):
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        return False

def get_platform_tag():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return "win_amd64"
    elif system == "linux":
        return "manylinux_2_17_x86_64.manylinux2014_x86_64" # Standard for maturin
    elif system == "darwin":
        if machine == "arm64":
            return "macosx_11_0_arm64"
        else:
            return "macosx_10_7_x86_64"
    return None

def get_python_tag():
    # e.g., cp310
    v = sys.version_info
    return f"cp{v.major}{v.minor}"

def download_wheel(repo, version, py_tag, platform_tag):
    # rust_caption-0.1.0-cp310-cp310-win_amd64.whl
    filename = f"rust_caption-{version}-{py_tag}-{py_tag}-{platform_tag}.whl"
    url = f"https://github.com/{repo}/releases/download/v{version}/{filename}"
    
    print(f"🔍 Looking for wheel: {url}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        return filename
    except Exception as e:
        print(f"⚠️ Wheel not found or download failed: {e}")
        return None

def main():
    print("### Caption Live (Rust) Installer ###")
    
    rust_dir = os.path.join(os.path.dirname(__file__), "rust_caption_src")
    
    # 1. Try Download Wheel
    platform_tag = get_platform_tag()
    py_tag = get_python_tag()
    
    wheel_file = None
    if platform_tag and "YOUR_USERNAME" not in GITHUB_REPO:
        print("☁️ Attempting to download pre-compiled binary...")
        wheel_file = download_wheel(GITHUB_REPO, VERSION, py_tag, platform_tag)
        
    if wheel_file:
        print(f"📦 Installing {wheel_file}...")
        if run_command(f"{sys.executable} -m pip install {wheel_file}"):
            print("\n🎉 Success! Installed from binary.")
            os.remove(wheel_file)
            return

    # 2. Fallback to Compilation
    print("\n🔨 Compiling from source (Fallback)...")
    
    if not check_command("cargo"):
        print("❌ Error: Rust (cargo) is not installed.")
        print("   Please install Rust: https://rustup.rs/")
        sys.exit(1)

    # Check for maturin
    try:
        import maturin
    except ImportError:
        print("📦 Installing maturin...")
        run_command(f"{sys.executable} -m pip install maturin")

    print("🚀 Building with maturin...")
    cmd = "maturin develop --release"
    if run_command(cmd, cwd=rust_dir):
        print("\n🎉 Success! Compiled and installed locally.")
    else:
        print("\n❌ Compilation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
