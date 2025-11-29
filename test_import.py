import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print(f"Python Version: {sys.version}")
print(f"Current Dir: {os.getcwd()}")

try:
    import rust_caption
    print("SUCCESS: Imported rust_caption!")
except ImportError as e:
    print(f"ERROR: Failed to import. Details:\n{e}")
except Exception as e:
    print(f"FATAL: Unexpected error:\n{e}")

