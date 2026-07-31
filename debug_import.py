
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Sys path: {sys.path}")

try:
    import ai2thor
    print(f"ai2thor imported: {ai2thor.__version__}")
except ImportError as e:
    print(f"ai2thor import failed: {e}")

try:
    import env.thor_env
    print("env.thor_env imported")
except ImportError as e:
    print(f"env.thor_env import failed: {e}")
except Exception as e:
    print(f"env.thor_env failed with other error: {e}")
