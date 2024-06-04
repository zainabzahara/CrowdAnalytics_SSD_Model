import os
import scipy.io
import pandas as pd

# Set the path to the .mat files
mat_path = r"H:\Zainab Zahara\HackerSpace\ML_Project1\part_A_final\train_data\ground_truth"
# Set the path to the .csv files
csv_path = 'ground_truth_csv'
# Loop over all .mat files in the directory
for mat_file in os.listdir(mat_path):
    # Check if the file is a .mat file
    if mat_file.endswith('.mat'):
        # Load the .mat file
        mat_data = scipy.io.loadmat(os.path.join(mat_path, mat_file))
        print("printing data of mat file:", mat_data.keys())
        # Extract the name of the variable containing the ground truth data
        gt_var_names = [k for k in mat_data.keys() if k == 'image_info']
        if gt_var_names:
            gt_var_name = gt_var_names[0]
        else:
            print(f"No variable found in {mat_file} that contains the ground truth data")
            continue
        # Extract the ground truth data from the .mat file
        gt_data = mat_data[gt_var_name]

        # Create a pandas DataFrame from the ground truth data
        gt_df = pd.DataFrame(gt_data)

        # Save the DataFrame as a .csv file
        csv_file = os.path.splitext(mat_file)[0] + '.csv'
        csv_path_file = os.path.join(csv_path, csv_file)
        gt_df.to_csv(csv_path_file, index=False)