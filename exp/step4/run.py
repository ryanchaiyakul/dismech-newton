import os, runpy, sys
sys.path.insert(0, os.path.dirname(__file__))
import probe  # noqa
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
