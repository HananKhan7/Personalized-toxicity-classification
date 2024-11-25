import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import spearmanr

directory = ''  # Insert model output path here

### Evaluation 
# (F1, MSI, and spearman rank correlation)
def evaluation(datapath: str):
    # Load the dataset
    with open(f"{datapath}.json", 'r') as file:
        data = json.load(file)

    # Initialize lists to store results
    results = []
    true_labels_all = []
    predictions_all = []

    # Define valid ratings
    valid_ratings = {"0", "1", "2", "3", "4"}

    # Iterate over each annotator
    for annotator in data:
        annotator_id = annotator['annotator_id']
        ratings = annotator['ratings']

        # Extract model predictions and true scores, ignoring invalid predictions
        true_scores = []
        model_predictions = []

        for rating in ratings:
            model_prediction = rating['model_prediction']
            toxic_score = rating['toxic_score']

            if model_prediction in valid_ratings:
                model_predictions.append(int(model_prediction))
                true_scores.append(toxic_score)

        # Append to the global list for micro-average calculations
        true_labels_all.extend(true_scores)
        predictions_all.extend(model_predictions)

        # Calculate metrics if there are valid predictions
        if model_predictions:
            accuracy = accuracy_score(true_scores, model_predictions)
            f1 = f1_score(true_scores, model_predictions, average='weighted')
            mse = mean_squared_error(true_scores, model_predictions)

            # Calculate Spearman rank correlation if there are enough data points
            if len(true_scores) > 1:
                spearman_corr, p_value = spearmanr(true_scores, model_predictions)
            else:
                spearman_corr, p_value = np.nan, np.nan

            results.append({
                "Annotator ID": annotator_id,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "MSE": mse,
                "Spearman Rank Correlation": spearman_corr,
                "Spearman Rank p_value": p_value
            })

    # Calculate micro averages using all combined true labels and predictions
    micro_accuracy = accuracy_score(true_labels_all, predictions_all)
    micro_f1 = f1_score(true_labels_all, predictions_all, average='weighted')
    micro_mse = mean_squared_error(true_labels_all, predictions_all)
    if len(true_labels_all) > 1:
        micro_spearman_corr, micro_p_value = spearmanr(true_labels_all, predictions_all)
    else:
        micro_spearman_corr, micro_p_value = np.nan, np.nan

    # Append micro-average results for global view
    results.append({
        "Annotator ID": "Micro Average",
        "Accuracy": micro_accuracy,
        "F1 Score": micro_f1,
        "MSE": micro_mse,
        "Spearman Rank Correlation": micro_spearman_corr,
        "Spearman Rank p_value": micro_p_value
    })

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{datapath}.xlsx", index=False)

#####################

def list_json_files(directory):
    # List to store the names of JSON files
    json_files = []

    # Iterate through the files in the directory
    for file in os.listdir(directory):
        # Check if the file ends with .json
        if file.endswith(".json"):
            json_files.append(os.path.splitext(file)[0])

    return json_files

directory_path = directory

# Get the list of JSON files
json_files = list_json_files(directory_path)

for file in tqdm(json_files, desc="Evaluating output"):
    evaluation(f"{directory_path}/{file}")

### Scores Aggregation

metrics = ['Accuracy', 'F1 Score', 'MSE', 'Spearman Rank Correlation', 'Spearman Rank p_value']
# List all files in the directory and filter out only the .xlsx files
xlsx_files = [file for file in os.listdir(directory) if file.endswith('.xlsx')]

# Initialize a dictionary to store results for all files
all_results = {}

# Process each .xlsx file
for file_name in xlsx_files:
    file_path = os.path.join(directory, file_name)
    # Load all sheets into a dictionary of dataframes
    all_data = {sheet_name: pd.read_excel(file_path, sheet_name=sheet_name) for sheet_name in pd.ExcelFile(file_path).sheet_names}

    # Create a dictionary to store results for the current file
    file_results = {}

    for sheet_name, data in all_data.items():
        # Calculate macro averages (mean across annotators)
        macro_averages = data[metrics].mean()

        # Extract the already computed micro averages
        micro_results = data.loc[data['Annotator ID'] == 'Micro Average', metrics].iloc[0]

        # Store the results
        file_results[sheet_name] = {
            'Macro Accuracy': macro_averages['Accuracy'],
            'Micro Accuracy': micro_results['Accuracy'],
            'Macro F1 Score': macro_averages['F1 Score'],
            'Micro F1 Score': micro_results['F1 Score'],
            'Macro MSE': macro_averages['MSE'],
            'Micro MSE': micro_results['MSE'],
            'Macro Spearman Rank Correlation': macro_averages['Spearman Rank Correlation'],
            'Micro Spearman Rank Correlation': micro_results['Spearman Rank Correlation'],
            'Macro Spearman Rank P_value': macro_averages['Spearman Rank p_value'],
            'Micro Spearman Rank P_value': micro_results['Spearman Rank p_value'],
        }

    # Add the results of the current file to the overall results
    all_results[file_name] = pd.DataFrame(file_results)

df_output = pd.concat(all_results, axis=1)

# Reordering and naming adjustment remains unchanged
def naming_function(path:str):
    with open(f"evaluation/{path}.json", 'r') as file:
        data_sample = json.load(file)
        user_profile_traits = len(data_sample[0]['selected_profile_traits'])
        output_string = f"{path}_with_{user_profile_traits}_profile_traits"
    return output_string

rows_order = [
    'Macro Accuracy',
    'Micro Accuracy',
    'Macro F1 Score',
    'Micro F1 Score',
    'Macro MSE',
    'Micro MSE',
    'Macro Spearman Rank Correlation',
    'Micro Spearman Rank Correlation',
    'Macro Spearman Rank P_value',
    'Micro Spearman Rank P_value',
]
# Reorder the rows of the dataframe according to the desired order
ordered_df = df_output.loc[rows_order]
col_map = {}
for column in ordered_df.columns:
    if "experiment" in column[0]:
        substring = column[0].split(".xlsx")[0]
        col_map[column[0]] = naming_function(substring)

# Create a new MultiIndex with the renamed columns
new_columns = [(col_map.get(file, file), sheet) for file, sheet in ordered_df.columns]
 
#Assign the new MultiIndex to the dataframe
ordered_df.columns = pd.MultiIndex.from_tuples(new_columns)

# Save output file
ordered_df.to_excel(f'{directory}/average_scores.xlsx')
