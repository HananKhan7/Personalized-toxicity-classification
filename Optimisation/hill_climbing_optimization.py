# Importing the libraries
import json
import torch
import random
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from scipy.stats import spearmanr

# Parameters
access_token = ''
cache_dir = "Mistral_Nemo_Instruct/DP-checkpoints"
N_shots = 5
batch_size = 7
max_annotators = 0
num_iterations = 30  # Number of hill climbing iterations
initial_traits = []  # Initial traits based on random search result

output_path = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Hugging Face model checkpoint
checkpoint = "mistralai/Mistral-Nemo-Instruct-2407"

# Load model and tokenizer from Hugging Face
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map = 'auto',
    quantization_config=quantization_config,
    cache_dir=cache_dir,
    use_auth_token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                        )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load dataset
data = load_dataset("json", data_files='datasets/updated_annotator_datasets.json')['train']

# Define all user profile traits
all_traits = [
    'education', 'age_range', 'lgbtq_status', 'race', 'toxic_comments_problem',
    'gender', 'political_affilation', 'is_parent', 'religion_important', 'technology_impact',
    'uses_media_social', 'uses_media_news', 'uses_media_video', 'uses_media_forums',
    'personally_seen_toxic_content', 'personally_been_target', 'identify_as_transgender'
]

# Define trait descriptions
trait_descriptions = {
    'education': 'education level',
    'age_range': 'age range',
    'lgbtq_status': 'LGBTQ status',
    'race': 'race',
    'toxic_comments_problem': 'They find toxic comments to be',
    'gender': 'gender',
    'political_affilation': 'political affiliation',
    'is_parent': 'parental status',
    'religion_important': 'importance of religion',
    'technology_impact': 'impact of technology',
}

# Define the detailed descriptions for boolean traits
boolean_trait_descriptions = {
    'uses_media_social': lambda profile: "They use social media" if profile.get('uses_media_social') else "They do not use social media",
    'uses_media_news': lambda profile: "They use news from media" if profile.get('uses_media_news') else "They do not use news from media",
    'uses_media_video': lambda profile: "They use video from media" if profile.get('uses_media_video') else "They do not use video from media",
    'uses_media_forums': lambda profile: "They use forums" if profile.get('uses_media_forums') else "They do not use forums",
    'identify_as_transgender': lambda profile: "They identify as transgender" if profile.get('identify_as_transgender') else "They do not identify as transgender",
    'personally_seen_toxic_content': lambda profile: "They have personally seen toxic content" if profile.get('personally_seen_toxic_content') else "They have personally not seen toxic content",
    'personally_been_target': lambda profile: "They have personally been a target of toxicity" if profile.get('personally_been_target') else "They have personally not been a target of toxicity"
}

# Function to generate profile strings
def generate_profile_string(profile, selected_traits):
    profile_parts = []
    for trait in selected_traits:
        if trait in boolean_trait_descriptions:
            profile_parts.append(boolean_trait_descriptions[trait](profile))
        else:
            profile_parts.append(f"{trait_descriptions[trait]}: '{profile[trait]}'")
    return ", ".join(profile_parts)

# Function to predict toxicity using the model
def predict_toxicity(comments, primary_profile, secondary_profiles, primary_train_data, secondary_train_data_list, selected_traits):
    """
    This function uses the Hugging Face model to predict the toxicity rating of a batch of comments.
    It constructs a prompt with the primary annotator's profile, example comments, and secondary annotators' profiles.
    The function then processes the model output and returns the predicted toxicity ratings.

    Parameters:
    comments (list): List of comments for which the toxicity ratings need to be predicted.
    primary_profile (dict): The profile of the primary annotator.
    secondary_profiles (list): A list of profiles of secondary annotators.
    primary_train_data (list): A list of example comments annotated by the primary annotator.
    secondary_train_data_list (list): A list of lists, where each inner list contains example comments annotated by a secondary annotator.

    Returns:
    list: List of predicted toxicity ratings of the comments.
    """
    # Construct primary profile details and example texts
    example_texts_primary = "\n".join([f'Text: "{entry["comment"]}"\nRating: {entry["toxic_score"]}' for entry in primary_train_data])

    # Construct secondary examples texts
    additional_examples_texts = ""
    annotator_labels = ["B", "C", "D", "E"]

    for idx, (secondary_profile, secondary_train_data) in enumerate(zip(secondary_profiles, secondary_train_data_list)):
        additional_examples_text = "\n".join([f'- Text: "{entry["comment"]}"\n    Rating: {entry["toxic_score"]}' for entry in secondary_train_data])
        
        additional_examples_texts += f"""[INST]For context, here are the same comments rated by a {annotator_labels[idx]} person with a different profile:\n Their profile: {generate_profile_string(secondary_profile, selected_traits)} \nexamples: {additional_examples_text}[/INST]\n"""

    # Create the prompt for each comment in the batch
    prompts = []
    primary_profile_string = generate_profile_string(primary_profile, selected_traits)
    for comment in comments:
        prompt = f"""
        <s>
        [INST]
        Given a text, how would a person 'A' of {primary_profile_string}, 
        rate the degree of toxicity in the given text, considering:
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some text examples, rated by person 'A': 
        {example_texts_primary} [/INST]
        {additional_examples_texts}
        [INST]Now classify the following text based on person 'A'.
        Text: {comment} [INST]</s>
        [INST] Toxicity:
        """
        prompts.append(prompt)

    # Calculate lengths of prompts
    prompt_lengths = [len(prompt) for prompt in prompts]
    # Set max_length to the maximum length found
    length = max(prompt_lengths)
    inputs = tokenizer(prompts, return_tensors="pt",
                   padding=True, truncation=True,
                    return_token_type_ids=False,
                    max_length=length
                    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = [output.split()[-1] for output in decoded_outputs]

    return results

# Function to calculate performance metrics
def calculate_performance_metrics(predictions, true_scores):
    """
    This function calculates the performance metrics of the model predictions.

    Parameters:
    predictions (list): List of model predictions.
    true_scores (list): List of true toxicity scores.

    Returns:
    dict: Dictionary containing accuracy, F1 score, and MSE.
    """
    # Filter predictions to include only valid integers in the range 0-4
    valid_predictions = []
    valid_true_scores = []
    for pred, score in zip(predictions, true_scores):
        try:
            pred_int = int(pred)
            if pred_int in [0, 1, 2, 3, 4]:
                valid_predictions.append(pred_int)
                valid_true_scores.append(int(score))
        except ValueError:
            continue  # Skip any invalid predictions

    if not valid_predictions:  # If no valid predictions remain
        raise ValueError("No valid predictions were generated.")

    accuracy = accuracy_score(valid_true_scores, valid_predictions)
    # Calculate F1 score
    f1 = f1_score(valid_true_scores, valid_predictions, average='weighted')
    
    # Calculate MSE
    mse = mean_squared_error(valid_true_scores, valid_predictions)
    
    # Calculate Spearman rank correlation
    spearman_corr, _ = spearmanr(valid_predictions, valid_true_scores)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'mse': mse,
        'spearman_corr': spearman_corr
    }

# Function to create annotator output
def create_annotator_output(dataset, max_annotators, N_shots, output_path, batch_size, num_iterations, initial_traits):
    """
    This function processes a dataset of comments and ratings, and generates a JSON file containing the model's predictions and annotator ratings.

    Parameters:
    dataset (list): A list of dictionaries, where each dictionary represents an annotator and their datasets. Each annotator dictionary should have the following keys: 'annotator_id', 'annotator_profile', 'test_dataset', 'train_dataset', and 'secondary_annotators'.
    output_path (str): The path where the generated JSON file will be saved.
    max_annotators (int): The maximum number of unique annotators to consider.
    N_shots (int): The number of shots to use as training data in prompt, for each annotator.
    batch_size (int): The number of comments to process in a single batch.
    num_iterations (int): The number of hill climbing iterations to conduct.
    initial_traits (list): Initial set of traits to start with.

    Returns:
    None
    """

    # Create a dictionary to quickly access annotators by their ID
    annotator_dict = {annotator['annotator_id']: annotator for annotator in dataset}
    
    # Initialize a set to keep track of selected traits combinations
    selected_traits_set = set()

    # Start with the initial set of traits
    current_traits = initial_traits
    selected_traits_set.add(tuple(current_traits))

    best_performance = None
    best_traits = current_traits
    best_iteration = 0  # Variable to track the best iteration number

    no_improvement_counter = 0  # Counter for tracking consecutive iterations without improvement
    improvement_threshold = 10  # Stop if no improvement is found within these many iterations

    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration+1}/{num_iterations}")

        # Initialize an empty dictionary to store the annotator data
        annotator_data = {}

        # Initialize the tqdm progress bar for annotators
        annotator_progress = tqdm(total=max_annotators, desc=f"Processing annotators for iteration {iteration+1}/{num_iterations}")

        # Iterate over each annotator in the dataset
        for i, annotator in enumerate(dataset):
            if i >= max_annotators:
                break

            annotator_id = annotator['annotator_id']
            primary_profile = annotator['annotator_profile'][0]
            test_dataset = annotator['test_dataset']
            
            # Select N_shots examples and retain only 'comment' and 'toxic_score'
            primary_train_dataset = [
                {k: entry[k] for k in ('comment', 'toxic_score')}
                for entry in annotator['train_dataset'][:N_shots]
            ]
            
            secondary_annotator_ids = annotator['secondary_annotators']

            # Retrieve secondary annotators' profiles and training data
            secondary_profiles = []
            secondary_train_data_list = []
            primary_train_comment_ids = {entry['comment_id'] for entry in annotator['train_dataset'][:N_shots]}
            
            for secondary_annotator_id in secondary_annotator_ids:
                secondary_annotator = annotator_dict[secondary_annotator_id]
                secondary_profiles.append(secondary_annotator['annotator_profile'][0])
                combined_secondary_data = [
                    {k: entry[k] for k in ('comment', 'toxic_score')}
                    for entry in secondary_annotator['train_dataset'] + secondary_annotator['test_dataset']
                    if entry['comment_id'] in primary_train_comment_ids
                ]
                secondary_train_data_list.append(combined_secondary_data)

            # Filter the primary profile to include only the selected traits
            filtered_primary_profile = {k: primary_profile[k] for k in current_traits}

            # Add annotator data
            annotator_data[annotator_id] = {
                "annotator_id": annotator_id,
                "selected_profile_traits": filtered_primary_profile,
                "ratings": []
            }

            # Process the test dataset in batches
            for batch_start in range(0, len(test_dataset), batch_size):
                batch_end = batch_start + batch_size
                batch = test_dataset[batch_start:batch_end]

                comments = [entry['comment'] for entry in batch]
                comment_ids = [entry['comment_id'] for entry in batch]
                true_scores = [entry['toxic_score'] for entry in batch]

                # Get the model's predictions for the batch
                predictions = predict_toxicity(comments, primary_profile, secondary_profiles, primary_train_dataset, secondary_train_data_list, current_traits)

                # Store the predictions
                for comment_id, comment, prediction, true_score in zip(comment_ids, comments, predictions, true_scores):
                    rating_data = {
                        "comment_id": comment_id,
                        "comment": comment,
                        "model_prediction": prediction,
                        "toxic_score": true_score
                    }
                    annotator_data[annotator_id]['ratings'].append(rating_data)

            annotator_progress.update(1)

        annotator_progress.close()

        # Save the collected data to a JSON file
        iteration_output_path = os.path.join(output_path, f'iteration_{iteration+1}.json')
        with open(iteration_output_path, 'w', encoding='utf-8') as f:
            json.dump(list(annotator_data.values()), f, indent=4)

        # Calculate performance metrics
        all_predictions = []
        all_true_scores = []
        for annotator_id, annotator in annotator_data.items():
            for rating in annotator['ratings']:
                all_predictions.append(rating['model_prediction'])
                all_true_scores.append(rating['toxic_score'])

        performance_metrics = calculate_performance_metrics(all_predictions, all_true_scores)
        print(f"Iteration {iteration+1} performance: {performance_metrics}")

        # Check if this iteration gives better performance
        if best_performance is None or performance_metrics['accuracy'] > best_performance['accuracy']:
            best_performance = performance_metrics
            best_traits = current_traits
            best_iteration = iteration + 1  # Update the best iteration number
            print(f"New best traits found: {best_traits} with performance: {best_performance}")
            no_improvement_counter = 0  # Reset the counter if improvement is found
        else:
            no_improvement_counter += 1  # Increment the counter if no improvement is found

        # Print the current best iteration number after each iteration
        print(f"Best traits found in iteration: {best_iteration}")

        # Stop the loop if no improvement is found within the threshold
        if no_improvement_counter >= improvement_threshold:
            print(f"Stopping early after {iteration + 1} iterations due to no improvement for {improvement_threshold} consecutive iterations.")
            break

        # Try to modify the current traits for the next iteration
        new_traits = current_traits.copy()
        if random.random() > 0.5 and len(all_traits) > 0:
            # Add a random trait
            trait_to_add = random.choice(all_traits)
            if trait_to_add not in new_traits:
                new_traits.append(trait_to_add)
        else:
            # Remove a random trait
            if len(new_traits) > 1:  # Ensure at least one trait remains
                trait_to_remove = random.choice(new_traits)
                new_traits.remove(trait_to_remove)

        # Ensure the new traits combination is unique
        if tuple(new_traits) not in selected_traits_set:
            current_traits = new_traits
            selected_traits_set.add(tuple(new_traits))
        else:
            print("Duplicate trait combination found, skipping this iteration")

    print(f"Best traits: {best_traits} with performance: {best_performance} found in iteration: {best_iteration}")

# Process the data and create the output files for each iteration
create_annotator_output(data, max_annotators=max_annotators, N_shots=N_shots, output_path=output_path, batch_size=batch_size, num_iterations=num_iterations, initial_traits=initial_traits)
