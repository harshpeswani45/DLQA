import json
import re
import copy
import sys
import os
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
os.environ['CUDA_VISIBLE_DEVICES']="3"

if len(sys.argv)!=3:
    print("please enter all the arguments")
    exit()

manual_path=sys.argv[1]
name_and_model=sys.argv[2]

if not os.path.isdir(manual_path):
    try:
        os.mkdir("./manuals/"+name_and_model)
    except:
        print("Cannot make the directory")
        exit()

with open(manual_path) as f:
  data = json.load(f)

model_name = 'msmarco-distilbert-base-v3'
bi_encoder = SentenceTransformer(model_name)


passages=dict()
parent=list()
cnt=0

def preparing_passages(data):
    global passages,parent,cnt
    if(data is None):
        return False

    if(len(data)==0):
        return True
   
    if(not isinstance(data,str)):
        for elem in data.keys():
            # Remove stars and hash of elem
            #Stars Removed
            elem1 = elem.replace('*', '<')
            elem1=re.sub('<[^<]+<', '', elem1)
            elem1=re.sub('<[^<]*<', '', elem1)


            elem1=re.sub('#[^#]+#', '', elem1)
            elem1=re.sub('#[^#]*#', '', elem1)
            if not elem1.find("table")==-1:
                continue

            #It ends here
            if elem1.find("step")==-1:
                parent.append(elem1)
            else:
                parent.append("")
            check=preparing_passages(data[elem])
            if(check):
                tmp=parent.pop()
                passages[tmp]=copy.deepcopy(parent)
                parent.append(tmp)
            parent.pop()
    else:
        parent.append(data)
        #check=preparing_passages(data[elem])
        #if(check):
        tmp=parent.pop()
        passages[tmp]=copy.deepcopy(parent)
        # parent.append(tmp)
    return True


ret_data=preparing_passages(data)

with open("./manuals/"+name_and_model+"/sentence_to_parent.json", 'w') as json_file:
  json.dump(passages, json_file)

data=passages
data_1=list()
data_1=list(data.keys())
data_2=list(data.values())
#print(type(data_1))

parent_to_sentence=dict()

for i in range(len(data_2)):
    #parent=list()
    if len(data_2[i])>0:
        parent=' : '.join(data_2[i])
        if parent in parent_to_sentence:
            parent_to_sentence[parent].append(data_1[i])
        else:
            parent_to_sentence[parent]=[data_1[i]]

with open("./manuals/"+name_and_model+"/parent_to_sentence.json", 'w') as json_file:
  json.dump(parent_to_sentence, json_file)
f.close()

data=passages
data_1=list()
data_1=list(data.keys())
data_2=list(data.values())
#print(type(data_1))
ret_lists=list()
for i in range(len(data_1)):
  ret_lists.append(" : ".join(data_2[i]+[data_1[i]]))
with open("./manuals/"+name_and_model+"/paragraphs.json", 'w') as json_file:
  
  json.dump(ret_lists, json_file)
# Closing file
f.close()
#print(len(ret_lists))
corpus_embeddings = bi_encoder.encode(ret_lists, convert_to_tensor=True, show_progress_bar=True)
torch.save(corpus_embeddings,"./manuals/"+name_and_model+"/embeddings_desc.pt")

