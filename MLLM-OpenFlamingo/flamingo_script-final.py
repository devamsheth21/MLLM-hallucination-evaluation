from open_flamingo import create_model_and_transforms
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoTokenizer
import torch
import json
import transformers
from PIL import Image
import requests
import re
import requests
import cv2
import io
import os
from functools import partial
import pickle
import random
import albumentations as A
import sys



torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

local_path = "/scratch/dmsheth1/nlp/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def image_augmenter(transform_id):
    id_prefixes = ['blur','brightness','contrast']
    id_suffixes = ['1','2','3','4','5']
    if transform_id == id_prefixes[0] + id_suffixes[0]:
      transform = A.Blur(p=1, blur_limit = 3)
    if transform_id == id_prefixes[0] + id_suffixes[1]:
      transform = A.Blur(p=1, blur_limit = 4)
    if transform_id == id_prefixes[0] + id_suffixes[2]:
      transform = A.Blur(p=1, blur_limit = 5)
    if transform_id == id_prefixes[0] + id_suffixes[3]:
      transform = A.Blur(p=1, blur_limit = 6)
    if transform_id == id_prefixes[0] + id_suffixes[4]:
      transform = A.Blur(p=1, blur_limit = 7)
    
    if transform_id == id_prefixes[1] + id_suffixes[0]:
      transform = A.RandomBrightness(p=1, limit = 0.2)
    if transform_id == id_prefixes[1] + id_suffixes[1]:
      transform = A.RandomBrightness(p=1, limit = 0.3)
    if transform_id == id_prefixes[1] + id_suffixes[2]:
      transform = A.RandomBrightness(p=1, limit = 0.3)
    if transform_id == id_prefixes[1] + id_suffixes[3]:
      transform = A.RandomBrightness(p=1, limit = 0.4)
    if transform_id == id_prefixes[1] + id_suffixes[4]:
      transform = A.RandomBrightness(p=1, limit = 0.5)
    
    if transform_id == id_prefixes[1] + id_suffixes[0]:
      transform = A.RandomContrast(p=1, limit = 0.2)
    if transform_id == id_prefixes[1] + id_suffixes[1]:
      transform = A.RandomContrast(p=1, limit = 0.3)
    if transform_id == id_prefixes[1] + id_suffixes[2]:
      transform = A.RandomContrast(p=1, limit = 0.3)
    if transform_id == id_prefixes[1] + id_suffixes[3]:
      transform = A.RandomContrast(p=1, limit = 0.4)
    if transform_id == id_prefixes[1] + id_suffixes[4]:
      transform = A.RandomContrast(p=1, limit = 0.5)

    return transform


samp = """I will give you a prompt. Generate 6 to 7 pairs of questions and answers from this prompt. You can include question about what attributes mentioned in the prompt such as shape, size, color etc. Think of these questions in such a way that their answers complete the description of the prompt and can be used to make an image through the features described in it. As an example, look at this prompt and the question answer pair generated for it:
Prompt: a little girl embroidery, fine art, muted background, calm colore. oil paiting by jan vermeer, masterpiese Question-Answer pair to be generated: Q. What is the object? A. The object is an embroidery of a little girl Q. What is the background color? A. The background is muted  Q.What type of painting is it? A. It is an oil painting. Q.Who is the painter? A. Jan Vermeer is the painter Generate similar question answer pairs for the following prompt:"""


# grab model checkpoint from huggingface hub
## model download is this below, you can specify where to download the checkpoint by modifying the path in 2nd argument.

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path,map_location=device), strict=False)

split_start = int(sys.argv[1])
split_end = int(sys.argv[2])

print("split_start: " + str(split_start), "split_end: " + str(split_end))

score_aggregator = []

# for i in range(0,10):
#    with open(local_path+"score_"+str(split_start+(i*1000))+"_"+str(split_end+(i*1000)), "rb") as fp:   # Unpickling
#        score_aggregator += pickle.load(fp)

with open("/scratch/dmsheth1/nlp/MLLM-hallucination-evaluation/Final_scripts/score_10k", "rb") as fp:   # Unpickling
    score_aggregator += pickle.load(fp)


mean = np.mean(score_aggregator)
std_dev = np.std(score_aggregator)
upperbound = mean + std_dev
lowerbound = mean - std_dev
print(lowerbound,upperbound)

# Initialize the final output dictionary
final_output = {}

dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k', split="train[" + str(split_start) + ":" + str(split_end)+"]")
# dataset = dataset.skip(split_start)
dataset = dataset.to_iterable_dataset()
json_file_path = "/scratch/dmsheth1/nlp/MLLM-hallucination-evaluation/QnA-Data/"+"QA_data" +str(split_start) +"_" + str(split_end) +".json"
with open(json_file_path, 'r') as json_file:
    final_output = json.load(json_file)

#transfor id to select transformation of format "<Transform type><Transform Amount>" based on above id_prefixes, id_suffixes in image_augmenter()
transform_id = 'blur1'

image_transform = image_augmenter(transform_id)

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


final = {}
for key, value in final_output.items():
    index1 = int(key) - split_start  # Adjust the index based on split_start
    data = next(iter(dataset.skip(index1).take(1)))
    print(data['prompt'])
   
    image = data['image']
  
    questions = value['qa_pairs'].keys()

    final[key] = value
    
  
    for question in questions:
        generated_text = generate_text_from_image_and_question(image, question)

        if 'answers_flamingo' not in final[key]:
            final[key]['answers_flamingo'] = {}
        final[key]['answers_flamingo'][question] = generated_text
        
        if list(final_output.keys())[0] == key:
          print(generated_text)
        # break
    # break

    image_data = np.array(image)
    image_BGR = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    augmented_image = image_transform(image=image_BGR)['image']
    augment_index = key + transform_id
    final[augment_index] = final[key]

    # Loop over each question for the augmented image
    for question in questions:
        generated_text = generate_text_from_image_and_question(Image.fromarray(augmented_image), question)
        final[augment_index]['answers_flamingo'][question] = generated_text
        
        # print(generated_text)
        # break



    modified_data = {}

    for key, value in final.items():
        modified_data[key] = {
            'qa_pairs': value['qa_pairs'],
            'answers_flamingo': value['answers_flamingo']
        }

    file_path = "modified_data"+str(split_start)+"_"+str(split_end)+".json"

    with open(local_path+file_path, 'w') as json_file:
        json.dump(modified_data, json_file, indent=4)

    # print(f"The modified data has been saved to {file_path}")


# Example usage:

#image = my_1k_data['image'][5]
#question = "5. What is the style of the portrait?"

#result_text = generate_text_from_image_and_question(image, question)
#print("Generated text: ", result_text)
