import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from Gaussian_experiment import generate_gaussian_experiment


from functions.Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the low-noise experiment results
    root_dir_dataset_experiment = Path('/workspace/data/output/set12/dataset_experiment')    
   
    dir_out_results = root_dir_dataset_experiment / 'results'

    # Directory containing the input images used in the experiment
    dir_images_set12 = Path('/workspace/data/input/set12')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_dataset_experiment)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_dataset_experiment': str(root_dir_dataset_experiment),
        'dir_images_set12': str(dir_images_set12),

        'dir_out_results': str(dir_out_results),

        # Output folders for each filtering method
        'dir_out_geonlm': str(root_dir_dataset_experiment / 'GEONLM'),
        'dir_out_nlm': str(root_dir_dataset_experiment / 'NLM'),
      


        # Filenames for serialized results (pickle/XLSX)      
        'name_pickle_results_nlm_output': 'array_nlm_filtereds.pkl',  
        'name_pickle_results_gnlm_output': 'array_gnlm_filtereds.pkl',   
        'name_results_xlsx_gnlm_output': 'gnlm_gnlm_filtereds.xlsx',
        'pickle_regional_experiment_summary': '/workspace/data/output/set12/dataset_experiment/results/regional_experiment_dataset.pkl',

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
        'nn': 10,      # Number of nearest neighbors
    }

    # Execute the low-noise Gaussian experiment
    generate_gaussian_experiment(parameters)
