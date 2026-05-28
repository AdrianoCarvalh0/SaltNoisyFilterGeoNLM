import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from typing import Any, Union
import os


PathLike = Union[str, Path]

def save_results_to_xlsx(records, output_dir, filename='results.xlsx'):
    """
    Save records to an Excel file built from an output directory and filename.
    """
    df = pd.DataFrame(records)

    def to_builtin(x):
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer):  return int(x)
        if isinstance(x, np.bool_):    return bool(x)
        return x

    df = df.applymap(to_builtin)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    df.to_excel(file_path, index=False)
    print(f"File saved to: {file_path}")
    return file_path

def save_pickle(obj: Any, output_dir: PathLike, filename: str = "object.pkl") -> Path:
    """
    Save a Python object to a pickle file built from an explicit output directory and filename.
    Creates parent directories if they don't exist.

    Parameters
    ----------
    obj : Any
        Python object to serialize.
    output_dir : str | Path
        Target directory where the pickle will be saved.
    filename : str
        Output filename (default: 'object.pkl').

    Returns
    -------
    Path
        Full path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with file_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path


def load_pickle(output_dir: PathLike, filename) -> Any:
    """
    Load a Python object from a pickle file using an explicit output directory and filename.

    Parameters
    ----------
    output_dir : str | Path
        Directory where the pickle is located.
    filename : str
        Filename to load (default: 'object.pkl').

    Returns
    -------
    Any
        Deserialized Python object.
    """
    file_path = Path(output_dir) / filename
    with file_path.open("rb") as f:
        return pickle.load(f)


def read_directories(directory, img=None, exclude_json=None):
    # Get a list of filenames in the specified directory
    filenames = []
    for filename in os.listdir(directory):
        if img is not None:
            # If 'img' is provided, filter filenames containing it
            if img in filename:   
                filenames.append(filename)
        elif exclude_json is not None:
            filenames.append(filename.replace('.json',''))     
        else:
            filenames.append(filename)    
    return filenames


def is_low_noise_or(h, sigma, h_lim=60.0, sigma_lim=10.0):
    """
    Determines whether an image is classified as 'low noise'
    based on either of two independent noise indicators.

    Parameters
    ----------
    h : float
        The NLM parameter (h) estimated from the image.
        Lower values typically indicate less noise.
    sigma : float
        The estimated noise standard deviation (σ) from the image.
    h_lim : float, optional
        Threshold for the NLM parameter to be considered 'low noise'.
        Default is 60.0.
    sigma_lim : float, optional
        Threshold for the estimated sigma to be considered 'low noise'.
        Default is 10.0.

    Returns
    -------
    bool
        True if either `h` or `sigma` indicates low noise,
        i.e., (h < h_lim) OR (sigma < sigma_lim).
        This makes the function more sensitive (inclusive)
        compared to using a strict AND condition.
    """
    return (h < h_lim) or (sigma < sigma_lim)


def get_multiplier(h, sigma, h_lim=60.0, sigma_lim=10.0):
    """
    Computes the adaptive multiplier for the denoising parameter
    based on the noise level classification.

    Parameters
    ----------
    h : float
        The NLM parameter (h) estimated from the image.
    sigma : float
        The estimated noise standard deviation (σ) from the image.
    h_lim : float, optional
        Threshold for h used in the low-noise classification.
    sigma_lim : float, optional
        Threshold for σ used in the low-noise classification.

    Returns
    -------
    float
        1.40  → if the image is classified as 'low noise'
        1.55  → otherwise (moderate or high noise)

    Notes
    -----
    The rule follows the logic:
        - Use a smaller multiplier (1.40) when either `h` or `σ` is low.
        - Use the default multiplier (1.55) otherwise.
    This setup achieved ~98% correct classification in calibration tests
    with no false positives on moderate/high noise images.
    """
    return 1.40 if is_low_noise_or(h, sigma, h_lim, sigma_lim) else 1.55


def ensure_output_dirs(base_output_dir):
    """
    Ensure that all required output directories exist.

    Parameters
    ----------
    base_output_dir : str or Path
        Base directory where experiment outputs are stored.
    """
    base = Path(base_output_dir)

    for subdir in ['NLM', 'GEONLM', 'BM3D', 'results']:
        (base / subdir).mkdir(parents=True, exist_ok=True)
