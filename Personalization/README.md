# Personalization Experiment for Toxicity Prediction

This project focuses on personalizing toxicity prediction experiments using three different models: Mistral Nemo Instruct, Llama 3.1 Instruct, and Gemma 2-it. It processes annotated data, applies different personalization approaches, and evaluates the performance of the models based on the generated predictions.

## Table of Contents

- [Personalization Experiment for Toxicity Prediction](#personalization-experiment-for-toxicity-prediction)
  - [Table of Contents](#table-of-contents)
    - [1. Overview](#1-overview)
    - [2. Data Loading](#2-data-loading)
    - [3. Data Preprocessing](#3-data-preprocessing)
    - [4. Processing Annotators and Batching](#4-processing-annotators-and-batching)
    - [5. Prediction Generation](#5-prediction-generation)
    - [6. Evaluation](#6-evaluation)
    - [Conclusion](#conclusion)

---

### 1. Overview

The **Personalization** module aims to customize the toxicity prediction process by leveraging data from multiple annotators with distinct profiles. The experiment uses three language models: **Mistral-Nemo-Instruct-2407**, **Llama-3.1-8B-Instruct**, and **Gemma-2-9b-it**, each trained and evaluated on toxicity prediction tasks. The core of the module focuses on tailoring the model's predictions based on annotator profiles, such as gender, race, education, and more, which allows for studying how different demographic factors influence toxicity ratings.

This approach applies various experimental configurations such as zero-shot and multi-profile experiments, and customizes predictions using both primary and secondary annotator data. A Hugging Face model is used for toxicity classification, with results being evaluated based on multiple metrics (e.g., accuracy, F1 score).

---

### 2. Data Loading

The dataset used in this experiment is loaded through the `load_dataset` function from the Hugging Face library, where it is expected to be in JSON format. The dataset includes a set of preprocessed and filtered comments, each annotated with a toxicity score. This dataset forms the foundation for training and testing the model.

Key actions during data loading:
- **Loading Data:** The preprocessed dataset (`datasets/preprocessed_filtered_dataset.json`) is read and stored into memory.
- **Context:** The dataset includes both primary and secondary annotators' profiles, comments, and toxicity labels. This structure is crucial for applying different personalization techniques to the model predictions.
- **Output:** A dataset object containing the processed data, ready for model evaluation and experimentation.

---

### 3. Data Preprocessing

Data preprocessing prepares the dataset for model input by performing several key transformations. This involves extracting and organizing data to facilitate efficient model training and evaluation.

Key actions include:
- **Feature Extraction:** Relevant features like the comment text and associated toxicity scores are extracted. These will be used to train the model and to generate predictions.
- **Profile Structuring:** Categorical attributes such as gender, race, and education level are structured into usable profile data, which is integral for the personalization experiments.
- **Data Splitting:** The dataset is split into training and testing subsets. Training data, typically associated with the primary annotator's profile, is used to fine-tune predictions, while test data is used for model evaluation.

The final preprocessed dataset is organized by annotator and ready for use in the subsequent steps of processing, annotation, and prediction generation.

---

### 4. Processing Annotators and Batching

Given the large volume of data and the need to process it efficiently, the `create_annotator_output` function splits the dataset into smaller batches. This function handles the batching process and annotator-specific data preparation, ensuring that resource constraints, such as memory limitations, are managed.

Key actions:
- **Batch Processing:** Data is split into batches of a specified size (`batch_size`), such as 5, to ensure that the system handles each batch independently, minimizing memory usage.
- **Annotator Handling:** The function processes each annotator's profile, both primary and secondary, with the training and test data being passed to the model in manageable chunks.
- **Efficiency:** The use of batching reduces memory consumption, allowing the experiment to scale and process large datasets effectively.

The output is a series of JSON files that contain the predictions for each annotator, with metadata indicating the model's performance on each batch.

---

### 5. Prediction Generation

The **prediction generation** process is where the actual toxicity classifications are made by the model. Based on the experiment type and the annotator profiles, a prompt is dynamically constructed and fed into the **Gemma-2-9b-it** model (or another model depending on configuration). The model then generates a toxicity score for each comment.

Key actions:
- **Dynamic Prompt Creation:** Depending on the experiment type (e.g., "Zero-shot Baseline," "Profile Only," or "Multi-profile"), the function generates different prompt structures. For example, in the "Profile Only" experiment, the model receives inputs that include the primary annotator's profile and the comments, while in "Multi-profile" experiments, multiple annotator profiles are included for a more complex prediction.
- **Model Inference:** Using the constructed prompts, the model generates toxicity ratings based on the comments. This is performed in a batch-wise manner to handle larger datasets efficiently.
- **Model Configuration:** The model is loaded with specific settings, such as 8-bit quantization and CUDA support for faster GPU inference. The model's device configuration (either CPU or GPU) is dynamically selected based on availability, ensuring optimal performance.
- **Output:** The predictions are saved alongside the original comments and true toxicity labels, forming the basis for evaluation in later steps.

---

### 6. Evaluation

After generating predictions, the model's performance is assessed using several key metrics: **Accuracy**, **F1 Score**, **Mean Squared Error (MSE)**, and **Spearman Rank Correlation**. The evaluation process is designed to provide both per-annotator metrics as well as micro-average metrics for the entire dataset, which are then saved for further analysis.

The **evaluation** function takes in a file path to a JSON file containing the model's predictions and true toxicity scores. The function iterates over each annotator's predictions and computes individual metrics. Here's a detailed breakdown of the process:

1. **Loading Data:**
   The function first loads the data from a specified JSON file. Each annotator's data, including their ratings, predictions, and true toxicity scores, is parsed from the file. This data is used for both per-annotator evaluation and the global micro-average evaluation.

2. **Metrics Calculation:**
   For each annotator, the function extracts the true toxicity scores and model predictions. It ensures that only valid ratings (i.e., scores between "0" and "4") are considered in the evaluation. If valid predictions are available, the following metrics are computed:
   
   - **Accuracy**: The proportion of correct predictions compared to the total number of predictions.
   - **F1 Score**: The weighted F1 score, which balances precision and recall, is calculated for each annotator.
   - **Mean Squared Error (MSE)**: The average squared difference between the true toxicity scores and the model predictions.
   - **Spearman Rank Correlation**: A statistical measure of the monotonic relationship between true toxicity scores and model predictions. If there are more than one data point, the Spearman correlation is computed along with its associated p-value.

   These metrics provide a comprehensive evaluation of how well the model performs for each annotator, allowing for a detailed comparison across different annotators.

3. **Micro-Averages Calculation:**
   In addition to the per-annotator metrics, the function calculates **micro-averages**, which aggregate the results across all annotators. This allows for a global performance overview, reflecting the overall accuracy, F1 score, MSE, and Spearman rank correlation for the entire dataset.

   - **Micro Accuracy**: The overall accuracy across all predictions in the dataset.
   - **Micro F1 Score**: The weighted F1 score across all predictions.
   - **Micro MSE**: The MSE calculated across all true toxicity scores and model predictions.
   - **Micro Spearman Rank Correlation**: The Spearman correlation for the entire dataset.

4. **Results Storage:**
   The calculated metrics for each annotator, as well as the micro-average results, are stored in a **DataFrame**. This data frame is then saved to an **Excel file** (e.g., `output_filename.xlsx`) for easy access and further analysis.

5. **Evaluation Loop:**
   The function is applied to all relevant files in the specified directory. For each JSON file, the `evaluation` function is executed, generating macro and micro-average results.

6. **Scores Aggregation and Final Output:**
Scores Aggregation and Final Output: After evaluating all files, the results are aggregated. A summary table is created for each experiment, which includes both macro and micro averages of the evaluation metrics. This aggregated table allows for a comparative analysis of the models' performance across different configurations.

The final scores are organized in a DataFrame, with the following metrics:

   - **Macro Accuracy**, **Macro F1 Score**, **Macro MSE**, **Macro Spearman Rank Correlation**: These represent the mean of the individual annotator scores.
   - **Micro Accuracy**, **Micro F1 Score**, **Micro MSE**, **Micro Spearman Rank Correlation**: These represent the overall performance for the entire dataset.

The DataFrame is saved as an Excel file, and the results are further refined with multi-index column names to clearly indicate the experiment configuration and the specific scores associated with each model's performance.

The final aggregated results are saved in a file like average_scores.xlsx for easy interpretation and comparison across different experiments.


---

### Conclusion

This repository provides a framework to personalize toxicity prediction tasks by using different annotator profiles and experimenting with various model configurations. The final evaluation allows for comparison between models and the effects of different personalization techniques on prediction accuracy.

For further details, please refer to the specific scripts and experiment configurations used for each model.
