import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch


import string
from tqdm.autonotebook import tqdm
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES']="3"

if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'msmarco-distilbert-base-v3'
bi_encoder = SentenceTransformer(model_name)
top_k = 100     #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

#lg_manual_filepath = '/content/drive/MyDrive/Data/[MFL71467502]2231_/output_folder/manual_final.json'

# if not os.path.exists(wikipedia_filepath):
#     util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)
passages=list()
corpus_embeddings=""
sentence_to_parents=""
parent_to_sentence=""

def initialize(name_and_model):
    global passages, corpus_embeddings, sentence_to_parents, parent_to_sentence
    with open("./manuals/"+name_and_model+'/paragraphs.json') as f:
      passages = json.load(f)
    # with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    #     for line in fIn:
    #         data = json.loads(line.strip())
    #         passages.extend(data['paragraphs'])

    #We limit the number of passages so that the encoding is faster.
    #Just uncomment the next line to encode the complete corpus of ~500+ paragraphs
    print("Passages:", len(passages))


    # To speed things up, pre-computed embeddings are downloaded.
    # The provided file encoded the passages with the model 'msmarco-distilbert-base-v2'
    # if model_name == 'msmarco-distilbert-base-v2':
    #   embeddings_filepath = 'simplewiki-2020-11-01-msmarco-distilbert-base-v2.pt'
    #   if not os.path.exists(embeddings_filepath):
    #       util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01-msmarco-distilbert-base-v2.pt', embeddings_filepath)

      # corpus_embeddings = torch.load(embeddings_filepath)
      # corpus_embeddings = corpus_embeddings.float() #Convert embedding file to float
      # if torch.cuda.is_available():
      #   corpus_embeddings = corpus_embeddings.to('cuda')
       #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    embeddings_filepath="./manuals/"+name_and_model+'/embeddings_desc.pt'
    corpus_embeddings = torch.load(embeddings_filepath)
    corpus_embeddings = corpus_embeddings.float() #Convert embedding file to float
    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to('cuda')
    #corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

    f = open("./manuals/"+name_and_model+'/sentence_to_parent.json',)

    # returns JSON object as 
    # a dictionary
    sentence_to_parents = json.load(f)

    f = open("./manuals/"+name_and_model+'/parent_to_sentence.json',)

    # returns JSON object as 
    # a dictionary
    parent_to_sentence = json.load(f)


    #This function will search all wikipedia articles for passages that
    #answer the query
def search(query):
  #print("Input question:", query)

   # print("Top-5 lexical search (BM25) hits")
  # for hit in bm25_hits[0:5]:
  #     print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

  ##### Sematic Search #####
  #Encode the query using the bi-encoder and find potentially relevant passages
  question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
  question_embedding = question_embedding.cuda() 
  # top_k=500
  hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
  hits = hits[0]  # Get the hits for the first query

  ##### Re-Ranking #####
  #Now, score all retrieved passages with the cross_encoder
  cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
  cross_scores = cross_encoder.predict(cross_inp)

  #Sort results by the cross-encoder scores
  for idx in range(len(cross_scores)):
      hits[idx]['cross-score'] = cross_scores[idx]


  #Output of top-5 hitt
  # print("Top-5 Bi-Encoder Retrieval hits")
  hits = sorted(hits, key=lambda x: x['score'], reverse=True)
  # for hit in hits[0:5]:
  #     print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

  # print("Top-5 Cross-Encoder Re-ranker hits")
  hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
  # for hit in hits[0:10]:
  #     print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
  return hits[0:20]

def extract_possible_answers(hits):
  possible_answers=list()
  for hit in hits:
    # if(hit['cross-score']<=0):
    #   break
    passage=passages[hit['corpus_id']]
    parent=""
    sentences=passage.split(':')
    
    for sentence in sentences:
      flag=False
      if(len(sentence.split())>=6):
        flag=True
        if(parent+sentence not in possible_answers):
          possible_answers.append(parent+sentence)
      
      

      if(len(sentence.split())>=2 and len(sentence.split())<6 ):
        if(parent+sentence not in possible_answers):
          possible_answers.append(parent+sentence)
      if not flag:
        parent+=sentence+" : "
        
  return possible_answers

def encoding_answers(possible_answers,query):
  answers_embeddings = bi_encoder.encode(possible_answers, convert_to_tensor=True)
  question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
  question_embedding = question_embedding.cuda() 
  top_k=500
  # print(corpus_embeddings.shape)
  # print(answers_embeddings.shape)
  hits = util.semantic_search(question_embedding, answers_embeddings, top_k=top_k)
  hits = hits[0]  # Get the hits for the first query

  ##### Re-Ranking #####
  #Now, score all retrieved passages with the cross_encoder
  cross_inp = [[query, possible_answers[hit['corpus_id']]] for hit in hits]
  cross_scores = cross_encoder.predict(cross_inp)

  #Sort results by the cross-encoder scores
  for idx in range(len(cross_scores)):
      hits[idx]['cross-score'] = cross_scores[idx]

  #Output of top-5 hitt
  # print("Top-5 Bi-Encoder Retrieval hits")
  hits = sorted(hits, key=lambda x: x['score'], reverse=True)
  # for hit in hits[0:5]:
  #     print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

  # print("Top-5 Cross-Encoder Re-ranker hits")
  hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
  return hits

def extracting_answers(answers):
  answer=answers[0].split(':')
  return answer[-1].strip()
  #return answers

# def forming_passage(hits):
#   i=0
#   passage=""
#   for hit in hits:
#     if hit['cross-score']>0 or True:
#       sentence_1=passages[hit['corpus_id']].split(':')[:-1]
#       sentence=passages[hit['corpus_id']].split(':')[-1]
#       sentence_1=str(sentence_1).strip()
#       sentence=str(sentence).strip()
#       if sentence_1 in list(parent_to_sentence.keys()):
#         passage+=sentence+" \n "+' '.join(parent_to_sentence[sentence_1])
#         i+=1
#         continue
#       parent_list=sentence_to_parents[sentence]
#       if  len(parent_list)>0:
#         parent=' : '.join(parent_list)
#         passage+=' '.join(parent_to_sentence[parent])
#     else:
#       break
#     i+=1
#     if i==2:
#       break
#   return passage

def forming_passage(hits,passages):
    passage=""
    hit=hits[0]
    sentence=passages[hit['corpus_id']].split(':')[-1]
    return sentence
        

def generate_passage(query):
  hits=search(query = query)
  possible_answer=extract_possible_answers(hits)
  if len(possible_answer)==0:
    return "N"
  hits_1=encoding_answers(possible_answer,query = query)
  return forming_passage(hits_1,possible_answer)