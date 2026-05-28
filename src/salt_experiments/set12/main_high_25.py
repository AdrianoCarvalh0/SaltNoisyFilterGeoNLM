from Gaussian_high_25 import generate_gaussian_experiment_high

if __name__ == '__main__':

    # Base output directory for the high-noise experiment results
    root_dir_output_high = f'/workspace/data/output/set12/high_noisy_25/test'

    # Directory containing the general input images used in the experiment
    dir_images_general = f'/workspace/data/input/general_images'
      
    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_high': root_dir_output_high,
        'dir_images_general': dir_images_general,

        # Output folders for each filtering method
        'dir_out_nlm': f'{root_dir_output_high}/NLM',
        'dir_out_geonlm': f'{root_dir_output_high}/GEONLM',
        'dir_out_bm3d': f'{root_dir_output_high}/BM3D',
        'dir_out_results': f'{root_dir_output_high}/results',

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_nlm_output_high': 'array_nln_high_filtereds.pkl',
        'name_pickle_results_gnlm_bm3d_output_high': 'results_gaussian_gnlm_bm3d_high.pkl',
        'name_results_xlsx_nlm_gnlm_bm3d_output_high': 'gnlm_bm3d_high_filtereds.xlsx',

        # Algorithmic parameters used internally by the experiment
        'f': 4,      # Patch radius
        't': 7,      # Search window radius
        'alpha': 0.5, # Geometric weight (for GEO-NLM)
        'nn': 10,     # Number of nearest neighbors / similar patches
    }

    # Execute the high-noise Gaussian experiment
    generate_gaussian_experiment_high(parameters)
