import sys
import os

sys.path.append(os.getcwd())

try:
    import caption_live
    print("Import successful")
    print(caption_live.NODE_CLASS_MAPPINGS)
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
