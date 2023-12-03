import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.multimodal import clip_score
import numpy as np
from PIL import Image
from datasets import load_dataset
from functools import partial
import pickle
import random
import transformers
import torch
import json



torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
local_path = '"/scratch/dmsheth1/nlp/clipscores/'
# dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k', split="train[:1000]"v
dataset = load_dataset("poloclub/diffusiondb", '2m_first_10k', split="train", streaming=True)
dataset = dataset.skip(6000)
pickle_size = 6000
# with open(local_path+"score_"+pickle_size, "rb") as fp:   # Unpickling
#    b = pickle.load(fp)
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    # images_int = (images * 255).astype("uint8")
    # permute = [2, 0, 1]
    imageRGB = torch.from_numpy(images)
    clip_score = clip_score_fn(imageRGB.permute(2,0,1), prompts).detach()
    # clip_score = clip_score_fn(imageRGB, prompts).detach()
    return round(float(clip_score), 4)

# print(dataset['train'])

def savelist(clipsc,t):
    with open("/scratch/dmsheth1/nlp/clipscores/score_"+str(6000)+"-"+str(6000+t), "wb") as fp:   #Pickling
        pickle.dump(clipsc, fp)

clip_scores = []
save_point = 2000
# clip_scores = b
for i,example in enumerate(dataset):
    if i%save_point==0:
        print(i+6000)
        savelist(clip_scores,i+6000)
    clip_scores.append(calculate_clip_score(np.array(example['image']), example['prompt']))
savelist(clip_scores,i)
b = clip_scores

mean = np.mean(b)
std_dev = np.std(b)
upperbound = mean + std_dev
lowerbound = mean - std_dev
print(lowerbound,upperbound)
