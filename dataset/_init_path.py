import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

add_path(os.path.join(this_dir, '..'))
# add_path(os.path.join(this_dir, '..', 'dataset'))
# add_path(os.path.join(this_dir, '..', 'networks'))
# add_path(os.path.join(this_dir, '..', 'utils'))