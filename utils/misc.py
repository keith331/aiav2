from pathlib import Path
import logging

logger = logging.getLogger(__name__)

INIT_PATH = Path(__file__).resolve().parents[1]

def get_inital_path() -> Path:

    return INIT_PATH

def get_approjx_path(fname) -> Path:

    return Path(INIT_PATH, fname)

def get_image_dir(images='images'):
    
    return Path(INIT_PATH, images)

def get_fname_string(fpath):

    return Path(fpath).name

def get_working_dir():
    
    Path(INIT_PATH/'use_data').mkdir(parents=True, exist_ok=True)
    return Path(INIT_PATH, 'use_data')

def create_save_dir(save_path=None):

    if save_path == None:
        p = Path('c:/save_data')
        p.mkdir(parents=True, exist_ok=True)

