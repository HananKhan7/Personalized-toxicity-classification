# Importing the libraries
import json
import torch
import random
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Parameters
access_token = ''
cache_dir = "Mistral_Nemo_Instruct/DP-checkpoints"
N_shots = 5
batch_size = 7
max_annotators = 0
num_experiments = 50  # Number of experiments
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
    device_map='auto',
    quantization_config=quantization_config,
    cache_dir=cache_dir,
    use_auth_token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load dataset
data = load_dataset("json", data_files='datasets/updated_annotator_datasets.json')['train']

# Define the complete list of user profile traits
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

# Function to select exactly 5 unique traits
def select_five_unique_traits(selected_traits_set):
    while True:
        selected_traits = tuple(random.sample(all_traits, 5))
        if selected_traits not in selected_traits_set:
            selected_traits_set.add(selected_traits)
            return list(selected_traits)

# Function to generate profile strings
def generate_profile_string(profile, selected_traits):
    profile_parts = []
    for trait in selected_traits:
        if trait in boolean_trait_descriptions:
            profile_parts.append(boolean_trait_descriptions[trait](profile))
        else:
            profile_parts.append(f"{trait_descriptions.get(trait, trait)}: '{profile[trait]}'")
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

# Function to create annotator output
def create_annotator_output(dataset, max_annotators, N_shots, output_path, batch_size, num_experiments):
    annotator_dict = {annotator['annotator_id']: annotator for annotator in dataset}
    selected_traits_set = set()

    for experiment in range(num_experiments):
        selected_traits = select_five_unique_traits(selected_traits_set)

        annotator_data = {}
        annotator_progress = tqdm(total=max_annotators, desc=f"Processing annotators for experiment {experiment+1}/{num_experiments}")

        for i, annotator in enumerate(dataset):
            if i >= max_annotators:
                break

            annotator_id = annotator['annotator_id']
            primary_profile = annotator['annotator_profile'][0]
            test_dataset = annotator['test_dataset']
            
            primary_train_dataset = [
                {k: entry[k] for k in ('comment', 'toxic_score')}
                for entry in annotator['train_dataset'][:N_shots]
            ]
            
            secondary_annotator_ids = annotator['secondary_annotators']
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

            filtered_primary_profile = {k: primary_profile[k] for k in selected_traits}

            annotator_data[annotator_id] = {
                "annotator_id": annotator_id,
                "selected_profile_traits": filtered_primary_profile,
                "ratings": []
            }

            for batch_start in range(0, len(test_dataset), batch_size):
                batch_end = batch_start + batch_size
                batch = test_dataset[batch_start:batch_end]

                comments = [entry['comment'] for entry in batch]
                comment_ids = [entry['comment_id'] for entry in batch]
                true_scores = [entry['toxic_score'] for entry in batch]

                predictions = predict_toxicity(comments, primary_profile, secondary_profiles, primary_train_dataset, secondary_train_data_list, selected_traits)

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

        experiment_output_path = os.path.join(output_path, f'experiment_{experiment+1}.json')
        with open(experiment_output_path, 'w', encoding='utf-8') as f:
            json.dump(list(annotator_data.values()), f, indent=4)

# Process the data and create the output files for each experiment
create_annotator_output(data, max_annotators=max_annotators, N_shots=N_shots, output_path=output_path, batch_size=batch_size, num_experiments=num_experiments)
