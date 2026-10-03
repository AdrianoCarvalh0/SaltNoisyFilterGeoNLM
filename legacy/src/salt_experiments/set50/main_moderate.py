import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Salt_moderate import generate_salt_experiment_moderate


if __name__ == '__main__':

    # Base output directory for the moderate salt-and-pepper noise experiment results
    root_dir_output_moderate = Path('/workspace/data/output/set50/salt_pepper_moderate/full_512')

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/set50')

    # Ensure required output directories exist
    for subdir in ['NLM', 'GEONLM', 'MEDIAN', 'ASWMF', 'results']:
        (root_dir_output_moderate / subdir).mkdir(parents=True, exist_ok=True)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input images and saving results
        'root_dir_output_moderate': str(root_dir_output_moderate),
        'dir_images_set50': str(dir_images_general),

        # Output folders for each filtering method
        'dir_out_nlm': str(root_dir_output_moderate / 'NLM'),
        'dir_out_geonlm': str(root_dir_output_moderate / 'GEONLM'),
        'dir_out_median': str(root_dir_output_moderate / 'MEDIAN'),
        'dir_out_aswmf': str(root_dir_output_moderate / 'ASWMF'),
        'dir_out_results': str(root_dir_output_moderate / 'results'),

        # Filenames for serialized results
        'name_pickle_nlm_output_moderate': 'array_nlm_salt_pepper_moderate_filtereds.pkl',
        'name_pickle_results_gnlm_median_aswmf_output_moderate':
            'results_salt_pepper_gnlm_median_aswmf_moderate.pkl',
        'name_results_xlsx_nlm_gnlm_median_aswmf_output_moderate':
            'gnlm_median_aswmf_salt_pepper_moderate_filtereds.xlsx',

        # Salt-and-pepper noise parameter
        # 0.03 means 3% total corrupted pixels:
        # 1.5% salt + 1.5% pepper
        'salt_pepper_density': 0.03,

        # Median filter baseline for salt-and-pepper noise.
        # Use 3x3 for moderate-density impulse noise to avoid unnecessary smoothing.
        'median_size': 3,

        # ASWMF parameters from Thanh et al. (2020): 7x7 window and weights 1,1,10.
        'aswmf_radius': 3,
        'aswmf_weight_diag_1': 1.0,
        'aswmf_weight_diag_2': 1.0,
        'aswmf_weight_other': 10.0,

        # Full-size run: keep the original Set50 image size.
        'resize_shape': None,
        'print_metrics': True,
        'verbose_internal': True,

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius
        'alpha': 0.5,  # Geometric weight for GEO-NLM
        'nn': 10,      # Number of nearest neighbors
    }

    # Execute the moderate salt-and-pepper noise experiment
    generate_salt_experiment_moderate(parameters)
