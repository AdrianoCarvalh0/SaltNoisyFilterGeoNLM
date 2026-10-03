import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Salt_medium import generate_salt_experiment_medium


if __name__ == '__main__':

    # Base output directory for the medium salt-and-pepper noise experiment results
    root_dir_output_medium = Path('/workspace/data/output/set50/salt_pepper_medium/full_512')

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/set50')

    # Ensure required output directories exist
    for subdir in ['NLM', 'GEONLM', 'MEDIAN', 'ASWMF', 'results']:
        (root_dir_output_medium / subdir).mkdir(parents=True, exist_ok=True)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input images and saving results
        'root_dir_output_medium': str(root_dir_output_medium),
        'dir_images_set50': str(dir_images_general),

        # Output folders for each filtering method
        'dir_out_nlm': str(root_dir_output_medium / 'NLM'),
        'dir_out_geonlm': str(root_dir_output_medium / 'GEONLM'),
        'dir_out_median': str(root_dir_output_medium / 'MEDIAN'),
        'dir_out_aswmf': str(root_dir_output_medium / 'ASWMF'),
        'dir_out_results': str(root_dir_output_medium / 'results'),

        # Filenames for serialized results
        'name_pickle_nlm_output_medium': 'array_nlm_salt_pepper_medium_filtereds.pkl',
        'name_pickle_results_gnlm_median_aswmf_output_medium':
            'results_salt_pepper_gnlm_median_aswmf_medium.pkl',
        'name_results_xlsx_nlm_gnlm_median_aswmf_output_medium':
            'gnlm_median_aswmf_salt_pepper_medium_filtereds.xlsx',

        # Salt-and-pepper noise parameter
        # 0.05 means 5% total corrupted pixels:
        # 2.5% salt + 2.5% pepper
        'salt_pepper_density': 0.05,

        # Median filter baseline for salt-and-pepper noise.
        # Use 3x3 for medium-density impulse noise to avoid unnecessary smoothing.
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

    # Execute the medium salt-and-pepper noise experiment
    generate_salt_experiment_medium(parameters)
