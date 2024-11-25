import re
import json
from collections import defaultdict
from tqdm import tqdm
import random

# Parameters
dataset_path = ""  # Insert dataset path here

# PART 1: LOAD DATA, FIX FORMAT, ASSIGN UNIQUE ANNOTATORS ID's

def load_data(file_path: str):
    # Read the file as a single string
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Since the file has multiple root elements, wrap them in a list
    # This assumes each root object is separated by a newline
    fixed_content = '[' + file_content.replace('}\n{', '},\n{') + ']'

    # Parse the JSON string into a Python object
    data = json.loads(fixed_content)
    return data

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def assign_unique_annotator_ids(data):
    unique_annotators = {}
    current_id = 0
    
    # Define the keys for the 17 characteristics
    keys = [
        "gender", "race", "technology_impact", "uses_media_social", "uses_media_news",
        "uses_media_video", "uses_media_forums", "personally_seen_toxic_content",
        "personally_been_target", "identify_as_transgender", "toxic_comments_problem",
        "education", "age_range", "lgbtq_status", "political_affilation", "is_parent",
        "religion_important"
    ]
    
    for comment in data:
        if "ratings" in comment:
            for rating in comment["ratings"]:
                # Create a tuple of the annotator characteristics
                annotator_profile = tuple(rating.get(key) for key in keys if key in rating)
                
                # Check if the annotator is already assigned an ID
                if annotator_profile not in unique_annotators:
                    unique_annotators[annotator_profile] = current_id
                    rating["annotator_id"] = current_id
                    current_id += 1
                else:
                    rating["annotator_id"] = unique_annotators[annotator_profile]
                    
    return data

# Load, process, and save data from Part 1
raw_data = load_data(dataset_path)  # Load the dataset
processed_data = assign_unique_annotator_ids(raw_data)  # Assign unique annotator IDs
save_data(processed_data, "processed_dataset.json")  # Save the processed data

# PART 2: SPLIT DATA INTO TRAINING AND TEST

# Initialize a dictionary to store annotations by annotator
annotations_by_annotator = defaultdict(list)

# Extract annotations and group them by annotator_id
for entry in processed_data:
    comment = entry['comment']
    comment_id = entry['comment_id']
    for rating in entry['ratings']:
        annotator_id = rating['annotator_id']
        annotation = {
            'comment': comment,
            'comment_id': comment_id,
            'toxic_score': rating['toxic_score']
        }
        annotations_by_annotator[annotator_id].append(annotation)

# Limit to the first 20 annotations per annotator
for annotator_id in annotations_by_annotator:
    annotations_by_annotator[annotator_id] = annotations_by_annotator[annotator_id][:20]

# Fields to exclude from the annotator profile
exclude_fields = {
    "toxic_score",
    "is_profane",
    "is_threat",
    "is_identity_attack",
    "is_insult",
    "is_sexual_harassment",
    "fine_to_see_online",
    "remove_from_online",
}

# Prepare the final dataset
final_dataset = []

# Function to balance the training dataset and ensure uniqueness
def balance_training_set(annotations):
    score_buckets = {i: [] for i in range(5)}
    for annotation in annotations:
        score_buckets[annotation['toxic_score']].append(annotation)
    
    training_set = set()
    for score in range(5):
        if score_buckets[score]:
            chosen_annotation = random.choice(score_buckets[score])
            training_set.add(chosen_annotation['comment_id'])
        else:
            chosen_annotation = random.choice(annotations)
            training_set.add(chosen_annotation['comment_id'])
    
    return [annotation for annotation in annotations if annotation['comment_id'] in training_set]

# Use tqdm to show progress
for annotator_id, annotations in tqdm(annotations_by_annotator.items(), desc="Processing annotators"):
    # Extract the annotator profile
    annotator_profile = next(
        (rating for entry in processed_data for rating in entry['ratings'] if rating['annotator_id'] == annotator_id)
    )
    
    annotator_profile_filtered = {
        key: annotator_profile[key] for key in annotator_profile if key not in exclude_fields and key != 'annotator_id'
    }

    # Shuffle annotations to randomize the selection
    random.shuffle(annotations)

    # Balance the training set and ensure uniqueness
    train_dataset = balance_training_set(annotations)

    # Remaining annotations for the test dataset
    train_comment_ids = {annotation['comment_id'] for annotation in train_dataset}
    test_dataset = [annotation for annotation in annotations if annotation['comment_id'] not in train_comment_ids]

    # Create the annotator data entry
    annotator_data = {
        "annotator_id": annotator_id,
        "annotator_profile": [annotator_profile_filtered],
        "test_dataset": test_dataset,
        "train_dataset": train_dataset
    }

    final_dataset.append(annotator_data)

# Save the final dataset to a single JSON file
save_data(final_dataset, "annotator_datasets.json")

# PART 3: ADD ADDITIONAL ANNOTATOR INFORMATION

# Initialize the updated data structure
updated_data = []

# Iterate over each annotator in the dataset with a progress bar
for i, annotator in enumerate(tqdm(final_dataset, desc='Processing annotators')):
    annotator_id = annotator['annotator_id']
    primary_profile = annotator['annotator_profile'][0]
    primary_train_dataset = annotator['train_dataset']
    
    # Find secondary annotators who have annotated all N_shots comments
    secondary_annotators = []
    primary_train_comment_ids = {entry['comment_id'] for entry in primary_train_dataset}
    for secondary_annotator in final_dataset:
        if secondary_annotator['annotator_id'] != annotator_id and len(secondary_annotators) < 4:
            secondary_train_comments = {entry['comment_id']: entry for entry in secondary_annotator['train_dataset']}
            secondary_test_comments = {entry['comment_id']: entry for entry in secondary_annotator['test_dataset']}
            combined_secondary_comments = {**secondary_train_comments, **secondary_test_comments}
            if primary_train_comment_ids.issubset(combined_secondary_comments.keys()):
                secondary_annotators.append(secondary_annotator['annotator_id'])

    # Add secondary annotators to the annotator data
    annotator['secondary_annotators'] = secondary_annotators
    updated_data.append(annotator)

# Save the updated data to a JSON file
save_data(updated_data, "updated_annotator_datasets.json")

# PART 4: CLEAN/PREPROCESS DATA

# Function to clean a single comment
def clean_comment(comment):
    # Remove special characters, URLs, mentions (@user), and hashtags
    comment = re.sub(r'http\S+|@\S+|#\S+|[^a-zA-Z0-9\s]', '', comment)
    
    # Return the cleaned comment
    return comment

# Function to process and clean the dataset
def clean_dataset(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Iterate over the dataset and clean each comment in both test_dataset and train_dataset
    cleaned_data = []
    for entry in data:
        cleaned_entry = entry.copy()  # Copy original entry
        
        # Clean test_dataset comments
        for comment in cleaned_entry['test_dataset']:
            comment['comment'] = clean_comment(comment['comment'])
        
        # Clean train_dataset comments
        for comment in cleaned_entry['train_dataset']:
            comment['comment'] = clean_comment(comment['comment'])

        cleaned_data.append(cleaned_entry)

    # Save the cleaned dataset as a new JSON file
    save_data(cleaned_data, output_file)

# Clean the dataset and save
clean_dataset("updated_annotator_datasets.json", "preprocessed_filtered_dataset.json")
