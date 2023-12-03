import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import pickle
import random
from transformers import AutoTokenizer,InstructBlipProcessor, InstructBlipForConditionalGeneration
import transformers
import torch
import json
import re
import requests
import cv2
import io
import albumentations as A

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

local_path = "/scratch/ajoshi72/"

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

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

split_start = 5000
split_end = 6000

print("split_start: " + str(split_start), "split_end: " + str(split_end))

# Initialize the final output dictionary
final_output = {}
# split_instruction = "train[" + str(split_start) + ":" + str(split_end) + "]"

dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k', split="train", streaming=True)
dataset = dataset.skip(split_start)

json_file_path = local_path+"QA_data" +str(split_start) +"_" + str(split_end) +".json"
with open(json_file_path, 'r') as json_file:
    final_output = json.load(json_file)

#transfor id to select transformation of format "<Transform type><Transform Amount>" based on above id_prefixes, id_suffixes in image_augmenter()
transform_id = 'blur5'

image_transform = image_augmenter(transform_id)

final_blip_output = {}

### This is single question code
### working
for key, value in final_output.items():
    # image = images[key]
    index1 = int(key) + split_start  # Adjust the index based on split_start
    data = next(iter(dataset.skip(index1).take(1)))
    # data = dataset[index1]
    image = data['image']
    # display(image)
    questions = value['qa_pairs'].keys()
    # print(questions)
    final_blip_output[key] = value

    # Loop over each question for the current image
    for question in questions:
        string = question
        # prompt = "answer this question about the image: "
        inputs = processor(images=image, text=string, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        
        if 'answers_blip' not in final_blip_output[key]:
            final_blip_output[key]['answers_blip'] = {}
        final_blip_output[key]['answers_blip'][question] = generated_text
        break

    image_data = np.array(image)
    image_BGR = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    augmented_image = image_transform(image=image_BGR)['image']
    augment_index = key + transform_id
    final_blip_output[augment_index] = final_blip_output[key]

    # Loop over each question for the augmented image
    for question in questions:
        string = question
        # prompt = "answer this question about the image: "
        inputs = processor(images=Image.fromarray(augmented_image), text=string, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        final_blip_output[augment_index]['answers_blip'][question] = generated_text
        break
    
    # Here we have finished processing 1000 images and their augmentations and put them in final_output
    # Print the updated final_output
    print(final_blip_output)

    modified_data = {}

    for key, value in final_blip_output.items():
        modified_data[key] = {
            'qa_pairs': value['qa_pairs'],
            'answers_blip': value['answers_blip']
        }

    file_path = "modified_data"+str(split_start)+"_"+str(split_end)+".json"

    with open(local_path+file_path, 'w') as json_file:
        json.dump(modified_data, json_file, indent=4)

    print(f"The modified data has been saved to {file_path}")