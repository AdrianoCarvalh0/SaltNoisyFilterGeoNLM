import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Gaussian_low import generate_gaussian_experiment_low
from functions.Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the low-noise experiment results
    root_dir_output_low = Path('/workspace/data/output/set12/low_noisy/test')

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/general_images')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_output_low)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_low': str(root_dir_output_low),
        'dir_images_general': str(dir_images_general),

        # Output folders for each filtering method
        'dir_out_nlm': str(root_dir_output_low / 'NLM'),
        'dir_out_geonlm': str(root_dir_output_low / 'GEONLM'),
        'dir_out_bm3d': str(root_dir_output_low / 'BM3D'),
        'dir_out_results': str(root_dir_output_low / 'results'),

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_nlm_output_low': 'array_nln_low_filtereds.pkl',
        'name_pickle_results_gnlm_bm3d_output_low':
            'results_gaussian_gnlm_bm3d_low.pkl',
        'name_results_xlsx_nlm_gnlm_bm3d_output_low':
            'gnlm_bm3d_low_filtereds.xlsx',

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
        'nn': 10,      # Number of nearest neighbors
    }

    # Execute the low-noise Gaussian experiment
    generate_gaussian_experiment_low(parameters)
