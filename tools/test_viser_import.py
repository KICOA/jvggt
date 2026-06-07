import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jvggt.viser_wrapper import viser_wrapper

print("import ok", viser_wrapper)
