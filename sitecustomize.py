import os
from pathlib import Path


if 'MPLCONFIGDIR' not in os.environ:
    mpl_config_dir = Path(__file__).resolve().parent / '.mplconfig'
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_config_dir)
