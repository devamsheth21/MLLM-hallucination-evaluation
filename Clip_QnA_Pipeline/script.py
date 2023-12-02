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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = load_dataset('poloclub/diffusiondb', '2m_first_1k')
my_1k_data = dataset['train']

with open("/home/ajoshi72/score_1000", "rb") as fp:   # Unpickling
   b = pickle.load(fp)

mean = np.mean(b)
std_dev = np.std(b)
upperbound = mean + std_dev
lowerbound = mean - std_dev
print(lowerbound,upperbound)


model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

samp = """I will give you a prompt. Generate 6 to 7 pairs of questions and answers from this prompt. You can include question about what attributes mentioned in the prompt such as shape, size, color etc. Think of these questions in such a way that their answers complete the description of the prompt and can be used to make an image through the features described in it. As an example, look at this prompt and the question answer pair generated for it:
Prompt: a little girl embroidery, fine art, muted background, calm colore. oil paiting by jan vermeer, masterpiese Question-Answer pair to be generated: Q. What is the object? A. The object is an embroidery of a little girl Q. What is the background color? A. The background is muted  Q.What type of painting is it? A. It is an oil painting. Q.Who is the painter? A. Jan Vermeer is the painter Generate similar question answer pairs for the following prompt:"""

# Initialize the final output dictionary
final_output = {}

# for index in range(len(my_1k_data['image'])):
for index,s in enumerate(b):
    if s<= upperbound and s>=lowerbound:
        sequences = pipeline(
            samp + my_1k_data['prompt'][index] + "Question-Answer pair to be generated:",
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
                
    
        final_output[index] = {
            'image': my_1k_data['image'][index],
            'prompt': my_1k_data['prompt'][index],
            # 'qa_pairs': qa_list
            'qa_pairs': qa_dict
            
        }
        break
print(final_output)

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

### This is single question code
### working
for key, value in final_output.items():
    image = value['image']
    # display(image)
    questions = value['qa_pairs'].keys()
    # print(questions)

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

        
        if 'answers_blip' not in value:
            value['answers_blip'] = {}
        value['answers_blip'][question] = generated_text

# Print the updated final_output
print(final_output)

### This is multi question code
### not working
# for key, value in final_output.items():
#     image = value['image']
#     # display(image)
#     questions = value['qa_pairs'].keys()
#     # print(questions)

#     # Loop over each question for the current image
#     # for question in questions:
#     #     string = question
#     string = " ".join(questions)
    
#     print(string)
#     prompt = "answer these questions about the image: "
#     inputs = processor(images=image, text=prompt+string, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         do_sample=False,
#         num_beams=5,
#         max_length=256,
#         min_length=1,
#         top_p=0.9,
#         repetition_penalty=1.5,
#         length_penalty=1.0,
#         temperature=1,
#     )

#     generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
#     print(generated_text)

    
#     if 'answers_blip' not in value:
#         value['answers_blip'] = {}
#     value['answers_blip'][question] = generated_text

# # Print the updated final_output
# print(final_output)