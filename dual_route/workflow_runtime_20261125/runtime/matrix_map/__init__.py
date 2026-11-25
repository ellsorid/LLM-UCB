"""Compatibility loader; all SOM implementations live in archive/SOM/matrix_map."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[3] / 'SOM' / 'matrix_map')]
if not Path(__path__[0]).is_dir():
    raise ImportError('SOM/matrix_map is missing from the archive root')
