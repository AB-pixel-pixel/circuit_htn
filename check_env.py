
import sys
import os

# Assume current directory is ALFRED_ROOT
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'gen'))

try:
    import ai2thor
    print("ai2thor imported successfully")
except ImportError:
    print("ai2thor NOT found")

try:
    import gen.constants as constants
    print("gen.constants imported successfully")
except ImportError as e:
    print(f"gen.constants NOT found: {e}")

try:
    from gen.env.thor_env import ThorEnv
    print("ThorEnv imported successfully")
except ImportError as e:
    print(f"ThorEnv NOT found: {e}")
