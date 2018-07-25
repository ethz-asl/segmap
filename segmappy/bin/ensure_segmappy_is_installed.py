from __future__ import print_function

from builtins import input
import sys, os

try:
    import segmappy
except ImportError:
    keys = input(
        "Failing to import segmappy. Would you like to temporarily add it to PYTHONPATH? [Y/n]: "
    )
    assert isinstance(keys, str)
    if keys not in ["n", "no", "No", "N"]:
        parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, parentdir)
        try:
            import segmappy
        except OSError:
            print("Error: Could not add {} to PYTHONPATH.".format(parentdir))
            raise
        else:
            print(
                "{} temporarily added to PYTHONPATH, import successful.".format(
                    parentdir
                )
            )
