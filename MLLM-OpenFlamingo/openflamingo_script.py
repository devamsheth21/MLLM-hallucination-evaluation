from open_flamingo import create_model_and_transforms
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import re


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)

# grab model checkpoint from huggingface hub
## model download is this below, you can specify where to download the checkpoint by modifying the path in 2nd argument.

## checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
## model.load_state_dict(torch.load(path), strict=False)


# Load the dataset with the `large_random_1k` subset
dataset = load_dataset('poloclub/diffusiondb', '2m_first_1k')

my_1k_data = dataset['train']


## pipeline functions 


def convert_question_to_statement(question):
    # Use regular expression to extract the part starting from "the" to the end
    match = re.search(r'the.*\?', question)
    
    if match:
        # Replace the question mark with "is"
        statement_part = match.group().replace('?', 'is').strip()

        # Construct the statement
        statement = statement_part[0].upper() + statement_part[1:]

        return statement

    return question


def generate_text_from_image_and_question(image, question):
    # Step 1: Load images (Replace this with your own image loading code)
    query_image = image

    # Step 2: Preprocessing images
    vision_x = [image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    # Step 3: Preprocessing text
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image> " + convert_question_to_statement(question)],
        return_tensors="pt",
    )

    # Step 4: Generate text
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    return tokenizer.decode(generated_text[0])

# Example usage:

#image = my_1k_data['image'][5]
#question = "5. What is the style of the portrait?"

#result_text = generate_text_from_image_and_question(image, question)
#print("Generated text: ", result_text)
