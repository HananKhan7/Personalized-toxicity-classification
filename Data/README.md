# **Data Preprocessing and Structuring for Personalization**

The `Data` folder is responsible for data loading, processing, cleaning, and preparing the dataset for model personalization and optimization. This folder performs various tasks, including fixing the dataset format, assigning unique annotator IDs, splitting the data into training and test sets, and cleaning the comments for further analysis. The folder is organized into four parts:

---

## **Part 1: Load Data, Fix Format, Assign Unique Annotator IDs**

In this part, the raw dataset is loaded and cleaned:

1. **Loading the Data**: 
   - The raw dataset is read from the specified path and converted into a JSON format suitable for processing.
   - Since the dataset contains multiple root elements, it is wrapped into a single list of JSON objects for proper parsing.

2. **Assigning Unique Annotator IDs**:
   - Each annotator is assigned a unique ID based on their profile, which consists of a set of 17 characteristics. These include factors like gender, race, education, political affiliation, and others.
   - A new field `annotator_id` is added to each rating to link it to a unique annotator.

The processed data is saved as `processed_dataset.json`.

---

## **Part 2: Split Data into Training and Test Sets**

This section splits the dataset into training and test sets:

1. **Group Annotations by Annotator**: 
   - Annotations are grouped by the unique `annotator_id`, and each annotator's ratings are limited to a maximum of 20 annotations.
   
2. **Balance Training Set**:
   - A function ensures the training set is balanced across different toxicity scores (from 0 to 4). A random selection is made to avoid bias towards any score.
   
3. **Split Data**:
   - After balancing, the data is split into training and test datasets for each annotator, ensuring that each dataset contains unique comments.

The resulting data, which includes training and test sets, is saved as `annotator_datasets.json`.

---

## **Part 3: Add Additional Annotator Information**

In this part, additional information about annotators is added:

1. **Identify Secondary Annotators**:
   - The primary annotator's dataset is analyzed, and secondary annotators are identified based on the overlap of annotated comments in both their training and test datasets.
   
2. **Link Secondary Annotators**:
   - For each annotator, a list of secondary annotators is created, i.e., annotators who have annotated a common set of comments.

The data with added secondary annotator information is saved as `updated_annotator_datasets.json`.

---

## **Part 4: Clean/Preprocess Data**

This part focuses on cleaning the comment text:

1. **Comment Cleaning**:
   - Special characters, URLs, user mentions, and hashtags are removed from the comments to ensure that the text is clean and consistent for further analysis.
   
2. **Processing the Dataset**:
   - The dataset is cleaned for both the training and test datasets.
   
The cleaned data is saved as `preprocessed_filtered_dataset.json`.

---

### **Summary**

The `Data` folder contains the following key steps:

1. **Data Loading & Annotator ID Assignment**: Load and format the dataset, then assign unique annotator IDs based on profile characteristics.
2. **Data Splitting**: Split the data into training and test sets, ensuring balance across toxicity scores.
3. **Annotator Information**: Add secondary annotators based on shared annotations.
4. **Data Cleaning**: Clean comments by removing unwanted characters and content.

By the end of this process, the data is ready for use in the **Personalization** phase, ensuring it is both structured and cleaned for model experimentation.
