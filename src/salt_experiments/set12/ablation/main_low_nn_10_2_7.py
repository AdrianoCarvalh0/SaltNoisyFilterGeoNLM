import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gaussian_experiments.set12.ablation.Gaussian_low import generate_gaussian_experiment_low
from Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the low-noise experiment results
    root_dir_nn10 = Path('/workspace/data/output/set12/ablation/low_noisy/nn_10')    
   
    dir_out_results = root_dir_nn10 / 'results'

    # Directory containing the input images used in the experiment
    dir_images_set12 = Path('/workspace/data/input/set12')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_nn10)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_nn10': str(root_dir_nn10),
        'dir_images_set12': str(dir_images_set12),

        'dir_out_results': str(dir_out_results),

        # Output folders for each filtering method
        'dir_out_geonlm': str(root_dir_nn10 / 'GEONLM'),

        # Filenames for serialized results (pickle/XLSX)      
        'name_pickle_results_gnlm_output_low': 'array_gnlm_low_filtereds.pkl',   
        'name_results_xlsx_gnlm_output_low': 'gnlm_gnlm_low_filtereds.xlsx',
        'pickle_results_summary_low': '/workspace/data/output/set12/low_noisy/results/array_nln_low_filtereds.pkl',

        # Algorithmic parameters used internally by the experiment
        'f': 2,        # Patch radius
        't': 7,        # Search window radius
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
        'nn': 10,      # Number of nearest neighbors
    }

    # Execute the low-noise Gaussian experiment
    generate_gaussian_experiment_low(parameters)
