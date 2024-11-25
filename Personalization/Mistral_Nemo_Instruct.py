# Import necessary libraries
import json, os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Parameters
access_token = ''
cache_dir = "Mistral_Nemo_Instruct/DP-checkpoints"
N_shots = 5  # Default value for N_shots; will change based on experiments
start_id = 0  # Starting annotator ID for processing. Processing the data in splits to cater for resource constraints.
end_id = 10000  # Ending annotator ID for processing
batch_size = 5
output_path = 'outputs/personalization_by_profile/full_run/Nemo_multi_annotators_5_shots/'
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
    device_map={ "": device },  # Specify the device explicitly
    quantization_config=quantization_config,
    cache_dir=cache_dir,
    use_auth_token=access_token
)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load dataset
data = load_dataset("json", data_files='datasets/filtered_dataset.json')['train']

# Function to predict toxicity using the model
def predict_toxicity(experiment, comments, primary_profile, secondary_profiles, primary_train_data, secondary_train_data_list):
    """
    This function generates a prompt based on the experiment type and primary/secondary annotator profiles, 
    and then uses the Hugging Face model to predict the toxicity rating of a batch of comments.
    
    Parameters:
    experiment (str): The type of experiment being conducted (e.g., 'Zero-shot Baseline', 'Profile Only', etc.).
    comments (list): A list of comments to classify for toxicity.
    primary_profile (dict): Profile of the primary annotator (e.g., gender, race, education).
    secondary_profiles (list): List of profiles for secondary annotators.
    primary_train_data (list): Training data from the primary annotator's annotations.
    secondary_train_data_list (list): List of training data from secondary annotators.
    
    Returns:
    list: Predicted toxicity ratings for the batch of comments.
    """
    # Construct primary profile details and example texts
    example_texts_primary = "\n".join([f'Text: "{entry["comment"]}"\nRating: {entry["toxic_score"]}' for entry in primary_train_data])
        
    # Construct examples for secondary annotator
    secondary_example_texts_B = "\n".join([f'- Text: "{entry["comment"]}"\n    Toxicity: {entry["toxic_score"]}' for entry in secondary_train_data_list[0]])

    # Construct secondary examples texts
    additional_examples_texts = ""
    annotator_labels = ["B", "C", "D", "E"]
    
    for idx, (secondary_profile, secondary_train_data) in enumerate(zip(secondary_profiles, secondary_train_data_list)):
        additional_examples_text = "\n".join([f'- Text: "{entry["comment"]}"\n    Rating: {entry["toxic_score"]}' for entry in secondary_train_data])
        
        additional_examples_texts += f"""[INST]For context, here are the same comments rated by a {annotator_labels[idx]} person with a different profile:\n Their profile: gender: '{secondary_profile['gender']}', 
        race: '{secondary_profile['race']}', age range: '{secondary_profile['age_range']}', 
        education level: '{secondary_profile['education']}', political affiliation: '{secondary_profile['political_affilation']}', 
        LGBTQ status: '{secondary_profile['lgbtq_status']}', parental status: '{secondary_profile['is_parent']}', 
        importance of religion: '{secondary_profile['religion_important']}', impact of technology: '{secondary_profile['technology_impact']}', 
        {'They use social media' if secondary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if secondary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if secondary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if secondary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if secondary_profile['uses_media_forums'] else 'They do not identify as transgender'},
        - They find toxic comments to be {secondary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if secondary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if secondary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        \nexamples: {additional_examples_text}[/INST]\n"""

    # Create the prompt based on the experiment type
    if experiment == 'Zero-shot Baseline':
        prompts = [f"""
        <s> [INST] You are an analysis bot designed to assess text toxicity. Your task is to categorize the toxicity level of a given text into one of the predefined categories. The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the numerical rating without including the word "Rating/Toxicity". Do not provide explanations, empty strings or notes. [/INST] 
        [INST]Text: {comment}[/INST]
        [INST] Toxicity: [/INST]</s>
        """ for comment in comments]

    elif experiment == 'Five-shot Baseline':
        prompts = [f"""
        <s> [INST] You are an analysis bot designed to assess text toxicity. Your task is to categorize the toxicity level of a given text into one of the predefined categories.
        The predefined categories are: 0 - not toxic, 1 - slightly toxic, 2 - moderately toxic, 3 - very toxic, 4 - extremely toxic.
        Only respond with the numerical rating without including the word "Toxicity/Rating". Do not provide explanations, empty strings or notes. [/INST] 
        [INST]Examples:
         - Text: "they have nothing nice to say"
         Rating: 0
        
         - Text: "Nothing but a bunch of lame-ass"
         Rating: 1

         - Text: "OH MY GOD I JUST DEALT WITH THE BIGGEST SHITLORD"
         Rating: 2

         - Text: 'Anna got all that ass and don’t know how to use it'
         Rating: 3

         - Text: "Jews are seen as snakes and thieving folks who should never be trusted"
         Rating: 4 [/INST]

        [INST]Text: {comment}[/INST]
        [INST] Toxicity: [/INST]</s>
        """
        for comment in comments]

    elif experiment == 'Profile Only':
        prompts = [f"""
        <s>
        [INST]
        Given a text, how would a person 'A' of gender: '{primary_profile['gender']}', 
        race: '{primary_profile['race']}', age range: '{primary_profile['age_range']}', 
        education level: '{primary_profile['education']}', political affiliation: '{primary_profile['political_affilation']}', 
        LGBTQ status: '{primary_profile['lgbtq_status']}', parental status: '{primary_profile['is_parent']}', 
        importance of religion: '{primary_profile['religion_important']}', impact of technology: '{primary_profile['technology_impact']}', 
        {'They use social media' if primary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if primary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if primary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if primary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if primary_profile['uses_media_forums'] else 'They do not identify as transgender'}, 
        rate the degree of toxicity in the given text, considering:
        - They find toxic comments to be {primary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if primary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if primary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Now classify the following text based on person 'A'.
        Text: {comment} </s>
        [INST] Toxicity:
        """ for comment in comments]
    elif experiment == 'Annotators Five-shot':
        prompts = [f"""
        <s>
        [INST]
        Given a text, rate the degree of toxicity in the text. 
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic. 
        Only respond with the numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some examples of how previous comments were rated:
        {example_texts_primary}
        Now classify the following text:
        Text: {comment} </s>
        [INST]Toxicity:
        """ for comment in comments]

    elif experiment == 'Profile + Five-shot':
        prompts = [f"""
        <s>
        Given a text, how would a person 'A' of gender: '{primary_profile['gender']}', 
        race: '{primary_profile['race']}', age range: '{primary_profile['age_range']}', 
        education level: '{primary_profile['education']}', political affiliation: '{primary_profile['political_affilation']}', 
        LGBTQ status: '{primary_profile['lgbtq_status']}', parental status: '{primary_profile['is_parent']}', 
        importance of religion: '{primary_profile['religion_important']}', impact of technology: '{primary_profile['technology_impact']}', 
        {'They use social media' if primary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if primary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if primary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if primary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if primary_profile['uses_media_forums'] else 'They do not identify as transgender'}, 
        rate the degree of toxicity in the given text, considering:
        - They find toxic comments to be {primary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if primary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if primary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some text examples, rated by person 'A':
        {example_texts_primary}
        Now classify the following text based on person 'A'.
        Text: {comment} </s>
        [INST] Toxicity:
        """ for comment in comments]
    elif experiment == 'Cross-profile':
        prompts = [f"""
        <s>
        [INST]
        Given a text, how would a person 'A' of gender: '{primary_profile['gender']}', 
        race: '{primary_profile['race']}', age range: '{primary_profile['age_range']}', 
        education level: '{primary_profile['education']}', political affiliation: '{primary_profile['political_affilation']}', 
        LGBTQ status: '{primary_profile['lgbtq_status']}', parental status: '{primary_profile['is_parent']}', 
        importance of religion: '{primary_profile['religion_important']}', impact of technology: '{primary_profile['technology_impact']}', 
        {'They use social media' if primary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if primary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if primary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if primary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if primary_profile['uses_media_forums'] else 'They do not identify as transgender'}, 
        rate the degree of toxicity in the given text, considering:
        - They find toxic comments to be {primary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if primary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if primary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some text examples, rated by person 'A':
        {example_texts_primary}
        For context, here are the same comments rated by a person 'B' with gender: '{secondary_profiles[0]['gender']}', 
        race: '{secondary_profiles[0]['race']}', age range: '{secondary_profiles[0]['age_range']}', 
        education level: '{secondary_profiles[0]['education']}', political affiliation: '{secondary_profiles[0]['political_affilation']}', 
        LGBTQ status: '{secondary_profiles[0]['lgbtq_status']}', parental status: '{secondary_profiles[0]['is_parent']}', 
        importance of religion: '{secondary_profiles[0]['religion_important']}', impact of technology: '{secondary_profiles[0]['technology_impact']}', 
        {'They use social media' if secondary_profiles[0]['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if secondary_profiles[0]['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if secondary_profiles[0]['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if secondary_profiles[0]['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if secondary_profiles[0]['uses_media_forums'] else 'They do not identify as transgender'},
        - They find toxic comments to be {secondary_profiles[0]['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if secondary_profiles[0]['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if secondary_profiles[0]['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        Examples:
        {secondary_example_texts_B} [/INST]
        [INST]Now classify the following text based on person 'A'.
        Text: {comment} [/INST] </s>
        [INST]Toxicity:
        """ for comment in comments]
    
    elif experiment == 'Multi-profile':
        prompts = [f"""
        <s>
        [INST]
        Given a text, how would a person 'A' of gender: '{primary_profile['gender']}', 
        race: '{primary_profile['race']}', age range: '{primary_profile['age_range']}', 
        education level: '{primary_profile['education']}', political affiliation: '{primary_profile['political_affilation']}', 
        LGBTQ status: '{primary_profile['lgbtq_status']}', parental status: '{primary_profile['is_parent']}', 
        importance of religion: '{primary_profile['religion_important']}', impact of technology: '{primary_profile['technology_impact']}', 
        {'They use social media' if primary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if primary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if primary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if primary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if primary_profile['uses_media_forums'] else 'They do not identify as transgender'}, 
        rate the degree of toxicity in the given text, considering:
        - They find toxic comments to be {primary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if primary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if primary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some text examples, rated by person 'A': 
        {example_texts_primary} [/INST]
        {additional_examples_texts}
        [INST]Now classify the following text based on person 'A'.
        Text: {comment} [INST] </s>
        [INST] Toxicity:
        """ for comment in comments]

    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt",
                   padding=True, truncation=False,
                    return_token_type_ids=False,
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

# Function to create annotator output with multiple experiments
def create_annotator_output(dataset, start_id, end_id, output_path, batch_size, experiments):
    """
    This function processes a dataset of annotators over a specified range (from start_id to end_id) 
    and generates a JSON file containing toxicity predictions based on different experiment variations.
    
    Parameters:
    dataset (list): A list of dictionaries representing annotators and their data.
    start_id (int): The starting annotator ID for processing.
    end_id (int): The ending annotator ID for processing.
    output_path (str): Path where the generated JSON file is saved.
    batch_size (int): The batch size to process the comments.
    experiments (list): A list of experiment variations to run (e.g., ['Zero-shot Baseline', 'Five-shot Baseline']).
    
    Returns:
    None: The function saves the results in a JSON file.
    """
    annotator_dict = {annotator['annotator_id']: annotator for annotator in dataset}

    # Iterate over each experiment
    for experiment in experiments:
        print(f"\nConducting {experiment} classification\n")

        annotator_data = {}

        annotator_progress = tqdm(total=end_id - start_id, desc=f"Processing annotators for {experiment}")

        for i, annotator in enumerate(dataset):
            annotator_id = annotator['annotator_id']

            if annotator_id < start_id or annotator_id > end_id:
                continue

            primary_profile = annotator['annotator_profile'][0]
            test_dataset = annotator['test_dataset']
            primary_train_dataset = [{k: entry[k] for k in ('comment', 'toxic_score')} for entry in annotator['train_dataset'][:N_shots]]

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

            annotator_data[annotator_id] = {
                "annotator_id": annotator_id,
                "annotator_profile": primary_profile,
                "ratings": []
            }

            # Process test dataset in batches
            for batch_start in range(0, len(test_dataset), batch_size):
                batch_end = batch_start + batch_size
                batch = test_dataset[batch_start:batch_end]
                comments = [entry['comment'] for entry in batch]
                comment_ids = [entry['comment_id'] for entry in batch]
                true_scores = [entry['toxic_score'] for entry in batch]

                predictions = predict_toxicity(experiment, comments, primary_profile, secondary_profiles, primary_train_dataset, secondary_train_data_list)

                for comment_id, comment, prediction, true_score in zip(comment_ids, comments, predictions, true_scores):
                    rating_data = {
                        "comment_id": comment_id,
                        "comment": comment,
                        "model_prediction": prediction,
                        "toxic_score": true_score
                    }
                    annotator_data[annotator_id]['ratings'].append(rating_data)

                torch.cuda.empty_cache()

            annotator_progress.update(1)

        annotator_progress.close()

        output_file = f"{output_path}Nemo_toxicity_classification_{experiment}.json"
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(list(annotator_data.values()))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)

#### Main

# Variable for experiments
experiments = [
    # 'Zero-shot Baseline',
#     'Five-shot Baseline',
#     'Profile Only',
#     'Annotators Five-shot',
#     'Profile + Five-shot',
#     'Cross-profile',
    'Multi-profile'
]


# Process the data for the specified range of annotators
create_annotator_output(
    dataset=data,
    start_id=start_id,
    end_id=end_id,
    output_path=output_path,
    batch_size=batch_size,
    experiments=experiments
)