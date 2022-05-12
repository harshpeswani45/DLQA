import QA_1
import QA_1_list
import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
import datetime 
import torch
import spacy
import csv
import json
import random
import torch
import pandas as pd
import numpy as np
import time
import datetime
import random
import os
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.metrics import classification_report

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import ast
import os
import glob


os.environ['CUDA_VISIBLE_DEVICES']="3"



# QA_1.initialize(name_and_model)
# QA_1_list.initialize(name_and_model)

product_name=""

# tokenizer1_1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
# model1_1 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

# tokenizer1_2 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
# model1_2 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-nl-en")

# tokenizer2_1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
# model2_1 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# tokenizer2_2 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
# model2_2 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

# tokenizer3_1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
# model3_1 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# tokenizer3_2 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
# model3_2 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# model1_1=model1_1.to("cuda")
# model1_2=model1_2.to("cuda")

# model2_1=model2_1.to("cuda")
# model2_2=model2_2.to("cuda")

# model3_1=model3_1.to("cuda")
# model3_2=model3_2.to("cuda")

device="cuda"

def initialize(model_no):
    global product_name
    QA_1.initialize(str(model_no))
    QA_1_list.initialize(str(model_no))
    manual_path='./manual_json/'+str(model_no)+'.json'
    with open(manual_path) as f:
        data = json.load(f)
    try:
        product_name=data['book']['bookinfo']['productname'].lower()
    except:
        pass
    
class CausalClassification:
  def __init__(self, model_path, model_class=AutoModelForSequenceClassification, tokenizer_class=AutoTokenizer):
    self.model = model_class.from_pretrained(model_path)
    self.tokenizer = tokenizer_class.from_pretrained(model_path)

    # Copy the model to the GPU.
    self.model.to(device)
  
  def predict(self, text1, text2):
    encoded_dict = self.tokenizer.encode_plus(
                        text1,                      # Sentence to encode.
                        text2,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 100,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    self.model.eval()
    with torch.no_grad():
        preds = self.model(input_ids.to(device), token_type_ids=None, 
                          attention_mask=attention_mask.to(device))
    logits = preds[0].detach().cpu().numpy()[0]
    
    logits=torch.softmax(torch.tensor(logits),0)
    return logits[1].item()


model_path='/home/development/harshp/LG_Work/Answer_Selection/Model_1'

clf = CausalClassification(model_path)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
#print(tokenizer)
#print()

#"/home/development/harshp/LG_Work/T5_MashQA/Unified_Model"
model=T5ForConditionalGeneration.from_pretrained("/home/development/harshp/LG_Work/T5_MashQA/Unified_Model")
#print(model)
# Read File

# df=pd.read_csv('./Dishwasher_Operation.csv')
# questions=df['Question']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
# list_ans=QA_1_list.generate_passage(query)

def t5_answer(con):
    #con=question+" \\n "+context
    encoding = tokenizer.encode_plus(con, return_tensors="pt",max_length=512, truncation=True)
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    #print(tokenizer.decode(encoding['input_ids'].squeeze()))
    output = beam_search_decoding (input_ids, attention_masks)
    return str(output)

def beam_search_decoding (inp_ids,attn_mask):
    beam_output = model.generate(input_ids=inp_ids,
                                     attention_mask=attn_mask,
                                     max_length=512,
                                     num_beams=1,
                                   num_return_sequences=1,
#                                        no_repeat_ngram_size=2,

                                   early_stopping=True
                                   )
    Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
    return [Question.strip().capitalize() for Question in Questions]

# def generate_answer(query):
    
#     descriptive_ans=QA_1.generate_passage(query)
    
#     Context=query+" \\n "+descriptive_ans
#     #print(Context)
#     res = ast.literal_eval(t5_answer(Context))
#     #print(res)
#     return res[0], descriptive_ans
 
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


def generate_queries(query):
    res=[query]
    
    sentence1=[query]
    encoded1=tokenizer1_1.batch_encode_plus(sentence1,return_tensors="pt",padding=True)
    encoded1=encoded1.to("cuda")
    ret1=model1_1.generate(**encoded1)
    sentence2=[tokenizer1_1.decode(t, skip_special_tokens=True) for t in ret1]
    
    encoded2=tokenizer1_2.batch_encode_plus(sentence2,return_tensors="pt",padding=True)
    encoded2=encoded2.to("cuda")
    ret2=model1_2.generate(**encoded2)
    tmp=[tokenizer1_2.decode(t, skip_special_tokens=True) for t in ret2]
    res.append(tmp[0])
    
    sentence1=[query]
    encoded1=tokenizer2_1.batch_encode_plus(sentence1,return_tensors="pt",padding=True)
    encoded1=encoded1.to("cuda")
    ret1=model2_1.generate(**encoded1)
    sentence2=[tokenizer2_1.decode(t, skip_special_tokens=True) for t in ret1]
    
    encoded2=tokenizer2_2.batch_encode_plus(sentence2,return_tensors="pt",padding=True)
    encoded2=encoded2.to("cuda")
    ret2=model2_2.generate(**encoded2)
    tmp=[tokenizer2_2.decode(t, skip_special_tokens=True) for t in ret2]
    res.append(tmp[0])
    
    sentence1=[query]
    encoded1=tokenizer3_1.batch_encode_plus(sentence1,return_tensors="pt",padding=True)
    encoded1=encoded1.to("cuda")
    ret1=model3_1.generate(**encoded1)
    sentence2=[tokenizer3_1.decode(t, skip_special_tokens=True) for t in ret1]
    
    encoded2=tokenizer3_2.batch_encode_plus(sentence2,return_tensors="pt",padding=True)
    encoded2=encoded2.to("cuda")
    ret2=model3_2.generate(**encoded2)
    tmp=[tokenizer3_2.decode(t, skip_special_tokens=True) for t in ret2]
    res.append(tmp[0])
    
    return res


def generate_answer(query):
    
    #queries=generate_queries(query)
    #print(query)
    query=query.lower()
#     query=query.replace("microwave","")
#     query=query.replace("oven","")
    
    for product in product_name.split():
        query=query.replace(product,"")
        
    #print(query)
    descriptive_answers=dict()
    list_answers=dict()
    queries=[query]
    for questions in queries:
        can_descriptive_ans=QA_1.generate_passage(questions).strip()
        can_list_ans=QA_1_list.generate_passage(questions).strip()
        
        if can_descriptive_ans in descriptive_answers:
            descriptive_answers[can_descriptive_ans]+=1
        else:
            descriptive_answers[can_descriptive_ans]=1
            
        if can_list_ans in list_answers:
            list_answers[can_list_ans]+=1
        else:
            list_answers[can_list_ans]=1
    
    freq_desc=0
    descriptive_ans=""
    
    for answer in descriptive_answers.keys():
        if descriptive_answers[answer.strip()]>freq_desc:
            freq_desc=descriptive_answers[answer.strip()]
            descriptive_ans=answer.strip()
            
    freq_list=0
    list_ans=""
    
    for answer in list_answers.keys():
        if list_answers[answer.strip()]>=freq_list:
            freq_list=list_answers[answer.strip()]
            list_ans=answer.strip()
    
     
    result_1=0
    result_2=0
    
    for question in queries:
#         cross_inp_1=[[question,descriptive_ans.strip()]]
#         cross_inp_2=[[question,list_ans.strip()]]

        result_1+=clf.predict(question,descriptive_ans.strip())
        result_2+=clf.predict(question,list_ans.strip())
    
   # print(list_ans)
    list_ans=list_ans.replace('summary,','')
    list_ans=list_ans.replace('summary','')
    list_ans=list_ans.replace('note,','')
    list_ans=list_ans.replace('note','')
    list_ans=list_ans.replace('caution,','')
    list_ans=list_ans.replace('caution','')
    list_ans=list_ans.replace('procedure,','')
    list_ans=list_ans.replace('procedure','')
    list_ans=list_ans.replace('warning,','')
    list_ans=list_ans.replace('warning','')
    list_ans=list_ans.replace('figure,','')
    list_ans=list_ans.replace('figure','')
    list_ans=list_ans.replace('variablelist,','')
    list_ans=list_ans.replace('variablelist','')
    
    list_ans=list_ans.replace('figure_1,','')
    list_ans=list_ans.replace('figure_1','')
    
    list_ans=list_ans.replace('figure_2,','')
    list_ans=list_ans.replace('figure_2','')
    
    list_ans=list_ans.replace('figure_3,','')
    list_ans=list_ans.replace('figure_3','')
    
    list_ans=list_ans.replace('figure_4,','')
    list_ans=list_ans.replace('figure_4','')
    
   # print(list_ans)
    
    
    Context1=query+" \\n "+descriptive_ans
    res1 = ast.literal_eval(t5_answer(Context1))
    
    Context2=query+" \\n "+list_ans
    res2 = ast.literal_eval(t5_answer(Context2))
    
    if list_ans.find(descriptive_ans.strip())!=-1:
        result_2=0
        result_1=-1
    
    return list_ans,descriptive_ans
    #if result_1 >= result_2:
        
        #ans=Bert_Classifier.clf.predict(query)
    #    return [res1[0],list_ans,descriptive_ans,res2[0],res1[0]],queries
        
    #else:
        
    #    return [list_ans,list_ans,descriptive_ans,res2[0],res1[0]],queries

    

    

# generated_answer=list()
# #print(generate_answer(questions[0]))
# for question in questions:
#     generated_answer.append(generate_answer(question)[0][0])

            
        

# QA_1.initialize(name_and_model)
# QA_1_list.initialize(name_and_model)

# for _ in range(len(generated_answer)-len(expected_answer)):
#     expected_answer.append("")
    
# bleu_scores=list()
    
# candidate=list()

# for answer in generated_answer:
#  #   print(answer)
#     candidate.append(answer.lower().split())
    
# reference=list()

# for answer in expected_answer:
#     reference.append([answer.lower().split()])

# for i in range(len(expected_answer)):
#     bleu_scores.append(sentence_bleu(reference[i], candidate[i]))
# score=corpus_bleu(reference, candidate)



# questions.append("Corpus BLEU Score")
# expected_answer.append(str(score))
# generated_answer.append("")
# bleu_scores.append("")

# generated_answer=list()
# prev_model=""
# curr_model=""

# for i in range(len(model_no)):
#     flag=True
#     print(model_no[i])
#     print(questions[i])
#     for file in glob.glob('./manuals/*'):
#         #print(model_no[i])
#         if file.split('/')[-1].strip()==model_no[i]:
#             flag=False
#             curr_model=(file.split('/')[-1] + '.')[:-1]
#             if not prev_model==curr_model:
#                 QA_1.initialize(str(model_no[i]))
#                 QA_1_list.initialize(str(model_no[i]))
#             manual_path='./manual_json/'+str(model_no[i])+'.json'
#             with open(manual_path) as f:
#                 data = json.load(f)
#             product_name=data['book']['bookinfo']['productname'].lower()
#             generated_answer.append(generate_answer(questions[i])[0][0])
#             prev_model=(curr_model + '.')[:-1]
#     if flag:
#         generated_answer.append("User manual does not exist")
            
# for question in questions:
    
