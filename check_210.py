
import ai2thor.controller
import os
import sys

# Override HOME
os.environ['HOME'] = os.getcwd()
os.environ['AI2THOR_HOME'] = os.path.join(os.getcwd(), 'ai2thor_cache')
if not os.path.exists(os.environ['AI2THOR_HOME']):
    os.makedirs(os.environ['AI2THOR_HOME'])

print("Starting controller... (v2)")
import traceback
try:
    c = ai2thor.controller.Controller(download_only=False)
    # c.start() # Deprecated in 5.0.0
    print("Controller started.")
    c.reset('FloorPlan302')
    print("Reset to FloorPlan302 done.")
    print("Objects:", len(c.last_event.metadata['objects']))
    all_types = set(o['objectType'] for o in c.last_event.metadata['objects'])
    print(f"Types: {sorted(list(all_types))}")
except Exception as e:
    traceback.print_exc()
