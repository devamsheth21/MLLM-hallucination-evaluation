import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.multimodal import clip_score
import numpy as np
from PIL import Image
from datasets import load_dataset
from functools import partial
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

# def calculate_clip_score(images, prompts):
#     # images_int = (images * 255).astype("uint8")
#     # permute = [2, 0, 1]
#     imageRGB = torch.from_numpy(images)
#     clip_score = clip_score_fn(imageRGB.permute(2,0,1), prompts).detach()
#     # clip_score = clip_score_fn(imageRGB, prompts).detach()
#     return round(float(clip_score), 4)


# def savelist(clipsc,start,end):
#     with open(local_path+"score_"+str(start)+"_"+str(end), "wb+") as fp:   #Pickling
#         pickle.dump(clipsc, fp)

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

samp = """I will give you a prompt. Generate 6 to 7 pairs of questions and answers from this prompt. You can include question about what attributes mentioned in the prompt such as shape, size, color etc. Think of these questions in such a way that their answers complete the description of the prompt and can be used to make an image through the features described in it. As an example, look at this prompt and the question answer pair generated for it:
Prompt: a little girl embroidery, fine art, muted background, calm colore. oil paiting by jan vermeer, masterpiese Question-Answer pair to be generated: Q. What is the object? A. The object is an embroidery of a little girl Q. What is the background color? A. The background is muted  Q.What type of painting is it? A. It is an oil painting. Q.Who is the painter? A. Jan Vermeer is the painter Generate similar question answer pairs for the following prompt:"""

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

split_start = 3000
split_end = 4000

print("split_start: " + str(split_start), "split_end: " + str(split_end))

score_aggregator = []

# for i in range(0,10):
#    with open(local_path+"score_"+str(split_start+(i*1000))+"_"+str(split_end+(i*1000)), "rb") as fp:   # Unpickling
#        score_aggregator += pickle.load(fp)

with open(local_path+"score_10k", "rb") as fp:   # Unpickling
    score_aggregator += pickle.load(fp)

# with open(local_path+"score_"+str(6000)+"_"+str(10000), "rb") as fp:   # Unpickling
#     score_aggregator += pickle.load(fp)

mean = np.mean(score_aggregator)
std_dev = np.std(score_aggregator)
upperbound = mean + std_dev
lowerbound = mean - std_dev
print(lowerbound,upperbound)

# Initialize the final output dictionary
final_output = {}
# split_instruction = "train[" + str(split_start) + ":" + str(split_end) + "]"

dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k', split="train", streaming=True)
dataset = dataset.skip(split_start)

images = {}
# prompts = dataset['prompt']

# for index in range(len(my_1k_data['image'])):
for index,data in enumerate(dataset):
    s = score_aggregator[split_start:split_end][index]
    if s<= upperbound and s>=lowerbound:
        images[str(index+split_start)] = data['image']
        sequences = pipeline(
            samp + data['prompt'] + "Question-Answer pair to be generated:",
            do_sample=False,
            top_k=50,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=1e-05,
            top_p=1,
            repetition_penalty=1.0,
            length_penalty=1
        )
    
        text = sequences[0]['generated_text']
    
        input_text = text.split('Question-Answer pair to be generated:')
        
        qa_dict = {}
        # qa_list = []
        for i in range(2, len(input_text), 2):
            questions = re.findall(r'Q\..*?\?', input_text[i])
            answers = re.findall(r'A\..*?(?=(?:Q\.|$))', input_text[i])
    
            for question, answer in zip(questions, answers):
                # newdict = {} #new dict for each qa
                # newdict[question.strip()] = answer.strip()
                # qa_list.append(newdict)
                qa_dict[question.strip()[3:]] = answer.strip()[3:]
                
        #Actual index of the image in the dataset is index+split_start
        final_output[str(index+split_start)] = {
            # 'image': images[index],
            'prompt': data['prompt'],
            # 'qa_pairs': qa_list
            'qa_pairs': qa_dict
            
        }

print(final_output)

file_path = "QA_data"+str(split_start)+"_"+str(split_end)+".json"

with open(local_path+file_path, 'w') as json_file:
    json.dump(final_output, json_file, indent=4)

print(f"The QnA data has been saved to {file_path}")

#transfor id to select transformation of format "<Transform type><Transform Amount>" based on above id_prefixes, id_suffixes in image_augmenter()
transform_id = 'blur5'

image_transform = image_augmenter(transform_id)

final_blip_output = {}

### This is single question code
### working
for key, value in final_output.items():
    image = images[key]
    # display(image)
    questions = value['qa_pairs'].keys()
    # print(questions)
    final_blip_output[key] = value

    # Loop over each question for the current image
    for question in questions:
        string = question
        print(string)
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
        print(generated_text)

        
        if 'answers_blip' not in final_blip_output[key]:
            final_blip_output[key]['answers_blip'] = {}
        final_blip_output[key]['answers_blip'][question] = generated_text

    image_data = np.array(image)
    image_BGR = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    augmented_image = image_transform(image=image_BGR)['image']
    augment_index = key + transform_id
    final_blip_output[augment_index] = final_blip_output[key]

    # Loop over each question for the augmented image
    for question in questions:
        string = question
        print(string)
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
        print(generated_text)
        final_blip_output[augment_index]['answers_blip'][question] = generated_text
    
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
