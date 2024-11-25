# Profile-in-Context Learning For Personalized Toxicity Classification

This repository contains the code used to conduct experiments for the thesis *"Profile-in-Context Learning for Personalized Toxicity Classification"*. It is organized into three main sections: **Data**, **Personalization**, and **Optimization**. Each section corresponds to a key phase in the workflow, from data processing to model evaluation and optimization. Below is an overview of the structure and functionality of each folder and its associated scripts.


> Each of the subfolders also contain their own `README.md` file, which provides a more detailed description of the specific tasks involved.
---

## **1. Data Folder**

The `Data` folder handles the essential tasks of **data cleaning, preprocessing, and restructuring**. The key steps in this phase include:

- **Cleaning and formatting** the raw dataset to remove any inconsistencies.
- **Restructuring the data** to align with the input requirements of the models used in subsequent experiments.
- **Creating datasets** that define the annotator profiles and their corresponding labels for toxicity classification tasks.

Once the data is preprocessed, it is ready for use in the **Personalization** experiments that follow.



---

## **2. Personalization Folder**

The `Personalization` folder focuses on **personalizing** the model to evaluate how different profile traits affect the toxicity classification task. This section includes the following:

- **Personalization Experiment for Three Models**: The scripts in this folder fine-tune three models by adjusting their behavior based on varying user profiles.
- **Evaluation of Results**: After training, the models are evaluated based on their performance with different user profiles, providing insights into the impact of profile personalization on toxicity predictions.

The results from this folder feed directly into the **Optimisation** phase.

---

## **3. Optimisation Folder**

The `Optimisation` folder focuses on finding the best combination of user profile traits to optimize the model's toxicity classification results. It consists of several experiments:

1. **Random Search Optimization**: 
   - A random search approach is used to explore different combinations of user profile traits, aiming to identify the best traits for predicting toxicity.

2. **Hill Climbing Optimization**: 
   - Building on the results of the random search, the hill climbing optimization method refines the search for the optimal traits by iteratively improving upon the best-performing configurations.

3. **5-Trait Random Search Optimization**: 
   - The final experiment tests combinations of exactly five traits at a time to find the most effective set of traits for toxicity prediction.

---