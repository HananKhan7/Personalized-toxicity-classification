# Optimization for Toxicity Classification

The **Optimisation** folder focuses on finding the best combination of user profile traits to optimize the model's toxicity classification results. It consists of several experiments aimed at identifying the most impactful traits for improving the model's predictions. These experiments involve using different optimization techniques to explore and refine combinations of traits.

## Table of Contents

- [Optimization for Toxicity Classification](#optimization-for-toxicity-classification)
  - [Table of Contents](#table-of-contents)
    - [1. Overview](#1-overview)
    - [2. Experiments](#2-experiments)
      - [2.1 Random Search Optimization](#21-random-search-optimization)
      - [2.2 Hill Climbing Optimization](#22-hill-climbing-optimization)
      - [2.3 5-Trait Random Search Optimization](#23-5-trait-random-search-optimization)
    - [3. Key Scripts](#3-key-scripts)
    - [Conclusion](#conclusion)

---

### 1. Overview

The goal of this module is to identify the best set of user profile traits for predicting toxicity in text. By performing multiple experiments using optimization methods, we aim to enhance the model's ability to predict toxicity by fine-tuning the combination of traits that influence the outcome.

---

### 2. Experiments

#### 2.1 Random Search Optimization

The **Random Search Optimization** experiment is designed to explore different combinations of user profile traits randomly. The random search method evaluates multiple trait configurations to identify which combinations of traits lead to the best performance for toxicity prediction. This approach provides a broad exploration of possible combinations and is often effective in identifying good models without exhaustive tuning.

- **Goal:** Identify the best combination of user profile traits for predicting toxicity.
- **Method:** Randomly sample different trait combinations for model evaluation.
- **Result:** The best performing trait combinations for toxicity prediction.

#### 2.2 Hill Climbing Optimization

Building on the results of the random search, the **Hill Climbing Optimization** experiment iterates on the results by refining the search for optimal traits. The method improves upon the best-performing configurations by making small changes to the trait combinations and selecting the ones that perform best. This approach helps fine-tune the results obtained from random search, progressively improving the model.

- **Goal:** Refine the search for optimal traits based on the results of the random search.
- **Method:** Iteratively improve the best configurations found in the random search by making small adjustments.
- **Result:** A refined set of traits that produce improved toxicity predictions.

#### 2.3 5-Trait Random Search Optimization

The **5-Trait Random Search Optimization** experiment takes a more focused approach by testing combinations of exactly five traits at a time. This experiment evaluates the impact of different five-trait combinations on the model's ability to predict toxicity. The approach narrows down the search space and allows for more targeted optimization of trait sets.

- **Goal:** Identify the most effective combination of exactly five traits for toxicity prediction.
- **Method:** Randomly sample combinations of five traits for evaluation.
- **Result:** The best-performing five-trait combinations for toxicity prediction.

---

### 3. Key Scripts

The **Random Search Optimization** script is a key part of the first experiment. Below is an overview of the key components of this script:

1. **Libraries and Parameters**:
   - The script imports necessary libraries such as `json`, `torch`, `random`, and `transformers` for model handling and random search operations.
   - Parameters such as batch size, number of experiments, and model configurations are defined at the start of the script.

2. **Model Loading**:
   - The script loads a pre-trained model from Hugging Face, configuring it for quantization and device mapping.

3. **Data Preparation**:
   - The dataset of annotated comments is loaded, and core and optional user profile traits are defined.

4. **Optimization Functions**:
   - A series of functions are defined to:
     - Randomly select traits.
     - Generate profile strings for users.
     - Predict toxicity based on selected traits.
     - Process comments in batches and collect results.

5. **Experiment Execution**:
   - The experiment is run for a set number of iterations, with random trait combinations selected for each iteration.
   - For each combination, toxicity predictions are made using the model, and the results are saved.


The **Hill Climbing Optimization** script is another key component of the optimization experiments, providing an alternative optimization method. Below is an overview of the key components of this script:

1. **Libraries and Parameters**:
   - Same as in **Random Search Optimization**. The script uses similar libraries for model handling and optimization operations but with the inclusion of hill climbing-specific logic for optimizing the trait combinations.

2. **Model Loading**:
   - A pre-trained Hugging Face model is loaded along with its tokenizer. The model is configured for efficient memory usage (8-bit quantization).
   - The device is determined based on available hardware, defaulting to CUDA if a compatible GPU is available.

3. **Data Preparation**:
   - The script loads a dataset containing annotated comments. 
   - It defines various user profile traits and trait descriptions, including boolean traits with customized descriptions for profile generation.

4. **Optimization Functions**:
   - The script defines a function to generate profile strings, which incorporates both boolean and other traits.
   - Another function (`predict_toxicity`) is used to predict the toxicity of comments using the model, based on a generated profile for each annotator.

5. **Hill Climbing Process**:
   - The algorithm starts with an initial set of traits and iteratively selects new trait combinations based on performance.
   - For each iteration, the script processes the test dataset in batches, making toxicity predictions and evaluating the model's performance using metrics like accuracy, F1 score, mean squared error, and Spearman correlation.
   - The algorithm includes a mechanism for early stopping if no performance improvement is observed after a defined number of iterations.

6. **Experiment Execution**:
   - For each iteration, the script updates the selected traits and performs toxicity prediction on batches of comments.
   - It tracks the best-performing trait combination and stops early if no significant improvement is seen after a threshold number of iterations.


The **5-Trait Optimization** script is the third experiment, where we focus on optimizing a fixed set of 5 traits for annotator profiles. Below is an overview of the key components of this script:

1. **Libraries and Parameters**:
   - Same as in **Random Search Optimization**. The script utilizes the same libraries, including `json`, `torch`, `random`, and `transformers`, for handling the model and data.

2. **Model Loading**:
   - The script loads a pre-trained model from Hugging Face using 8-bit quantization for efficient memory usage. The model is configured to run on available hardware (CUDA if available).
   - The tokenizer is also loaded for text preprocessing, with padding adjustments to ensure proper tokenization.

3. **Data Preparation**:
   - The dataset of annotated comments is loaded, and user profile traits are defined, including a complete list of traits from general demographic characteristics to behavior-related traits.
   - A function is provided to select exactly 5 unique traits from this list for each experiment.

4. **Optimization Functions**:
   - A key function, `select_five_unique_traits`, randomly selects 5 unique traits for each experiment iteration, ensuring that previously selected combinations are not repeated.
   - Another function, `generate_profile_string`, constructs a string representation of the profile based on the selected traits, which will be used in the toxicity prediction process.

5. **Predicting Toxicity**:
   - The script uses the `predict_toxicity` function to evaluate the toxicity of a list of comments. The function prepares a prompt for the model by combining the profile string, example comments, and secondary annotators' profiles for context.
   - The model generates toxicity ratings for each comment, based on the profile information of the annotators.

6. **Experiment Execution**:
   - The script runs multiple experiments (`num_experiments`), selecting different combinations of 5 traits for each experiment. 
   - In each experiment, a batch of annotators is processed, and their toxicity ratings for test comments are predicted. The results are saved into JSON files for further analysis.
   - The experiment is repeated for the specified number of iterations (`num_experiments`), with each iteration using a new set of 5 traits.

This script focuses on optimizing the performance by selecting and testing different combinations of 5 traits for each experiment, ultimately evaluating the effect on the accuracy of the toxicity predictions.


---

### Conclusion

The **Optimisation** folder contains a series of experiments designed to fine-tune the model’s ability to predict toxicity by selecting the most impactful user profile traits. By leveraging methods such as random search, hill climbing optimization, and focused 5-trait testing, this module aims to identify the optimal combination of traits for accurate toxicity classification.

For further details, refer to the specific scripts for each optimization experiment.

---
