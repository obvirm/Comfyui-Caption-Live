import sys
import subprocess
import os
import platform
import shutil
import urllib.request
import glob

def check_command(command):
    return shutil.which(command) is not None

def run_command(command, cwd=None, env=None):
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        subprocess.check_call(command, shell=True, cwd=cwd, env=full_env)
        return True
    except subprocess.CalledProcessError:
        return False

def copy_icu_datafile():
    """Copy icudtl.dat to project folder. __init__.py will set ICU_DATA env var."""
    node_dir = os.path.dirname(__file__)
    dest_path = os.path.join(node_dir, "icudtl.dat")
    
    # Already exists
    if os.path.exists(dest_path):
        print(f"✅ ICU datafile exists: {dest_path}")
        return True
    
    # Find from build output
    for src in glob.glob(os.path.join(node_dir, "target", "release", "build", "skia-bindings-*", "out", "skia", "icudtl.dat")):
        if os.path.exists(src):
            try:
                shutil.copy2(src, dest_path)
                print(f"✅ Copied ICU datafile to: {dest_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed: {e}")
    
    print("⚠️ ICU datafile not found - text rendering may have issues")
    return False

def main():
    print("### Caption Live (Rust) Installer - THE HARD WAY ###")
    
    rust_dir = os.path.join(os.path.dirname(__file__), "backend_bridge")
    
    # ENFORCE BUILD FROM SOURCE
    build_env = {
        "SKIA_BUILD_FROM_SOURCE": "1",
        "SKIA_NINJA_COMMAND": "ninja"
    }
    
    print("⚠️  FORCE BUILD FROM SOURCE ACTIVE")
    print("⏳ Expect 30-60 minutes build time. Requires LLVM & Visual Studio.")

    if not check_command("cargo"):
        print("❌ Error: Rust (cargo) is not installed.")
        sys.exit(1)

    try:
        import maturin
    except ImportError:
        print("📦 Installing maturin...")
        subprocess.check_call(f"{sys.executable} -m pip install maturin", shell=True)

    print("🚀 Building with maturin...")
    # Gunakan release build
    cmd = "maturin develop --release"
    if run_command(cmd, cwd=rust_dir, env=build_env):
        print("\n🎉 Success! Compiled and installed locally.")
        
        # Copy ICU datafile for text rendering
        print("\n📋 Setting up ICU datafile...")
        copy_icu_datafile()
    else:
        print("\n❌ Compilation failed.")
        print("💡 Check your LLVM/Clang and Visual Studio installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()

