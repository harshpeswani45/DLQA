import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
import re


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
    with open("./manuals/"+name_and_model+'/headings.json') as f:
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
    embeddings_filepath="./manuals/"+name_and_model+'/embeddings_list.pt'
    corpus_embeddings = torch.load(embeddings_filepath)
    corpus_embeddings = corpus_embeddings.float() #Convert embedding file to float
    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to('cuda')
    f = open("./manuals/"+name_and_model+'/sentence_to_parent_list.json',)

    # returns JSON object as 
    # a dictionary
    sentence_to_parents = json.load(f)

    f = open("./manuals/"+name_and_model+'/parent_to_sentence_list.json',)

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
  #question_embedding = question_embedding.cuda() REMOVE THIS
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

# def extract_possible_answers(hits):
#   possible_answers=list()
#   for hit in hits:
#     if(hit['cross-score']<=0):
#       break
#     passage=passages[hit['corpus_id']]
#     parent=""
#     sentences=passage.split(':')
    
#     for sentence in sentences:
#       flag=False
#       if(len(sentence.split())>=6):
#         flag=True
#         if(parent+sentence not in possible_answers):
#           possible_answers.append(parent+sentence)
      
      

#       if(len(sentence.split())>=2 and len(sentence.split())<6 ):
#         if(parent+sentence not in possible_answers):
#           possible_answers.append(parent+sentence)
#       if not flag:
#         parent+=sentence+" : "
        
#   return possible_answers

# def encoding_answers(possible_answers,query):
#   answers_embeddings = bi_encoder.encode(possible_answers, convert_to_tensor=True)
#   question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
#   question_embedding = question_embedding.cuda()
#   top_k=500
#   # print(corpus_embeddings.shape)
#   # print(answers_embeddings.shape)
#   hits = util.semantic_search(question_embedding, answers_embeddings, top_k=top_k)
#   hits = hits[0]  # Get the hits for the first query

#   ##### Re-Ranking #####
#   #Now, score all retrieved passages with the cross_encoder
#   cross_inp = [[query, possible_answers[hit['corpus_id']]] for hit in hits]
#   cross_scores = cross_encoder.predict(cross_inp)

#   #Sort results by the cross-encoder scores
#   for idx in range(len(cross_scores)):
#       hits[idx]['cross-score'] = cross_scores[idx]

#   #Output of top-5 hitt
#   # print("Top-5 Bi-Encoder Retrieval hits")
#   hits = sorted(hits, key=lambda x: x['score'], reverse=True)
#   # for hit in hits[0:5]:
#   #     print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

#   # print("Top-5 Cross-Encoder Re-ranker hits")
#   hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
#   return hits

# def check(hits):
#   passage_1=""
#   passage_2=""
#   hit=hits[0]
#   sentence_1=passages[hit['corpus_id']].split(':')[-1]
#   sentence_2=passages[hit['corpus_id']].split(':')[:-1]
#   sentence_2=sentence_2[-1]

#   small_check=passages[hit['corpus_id']].split(':')[-2]
  
#   if len(small_check.split()) <=4 :
#     return False
  
#   sentence_1=str(sentence_1).strip()
#   sentence_2=str(sentence_2).strip()
#   if sentence_1 not in sentence_to_parents:
#     return False
  
#   parent_list_2=[]
#   parent_list_1=[]
#   if sentence_1 in sentence_to_parents: 
#       parent_list_1=sentence_to_parents[sentence_1]
    
#   if sentence_2 in sentence_to_parents:
#       parent_list_2=sentence_to_parents[sentence_2]

#   # print(sentence_1)
#   # print(sentence_2)

#   if  len(parent_list_2)>0:
#     parent=' : '.join(parent_list_2)
#     if len(parent_to_sentence[parent])<=2:
#       return False
#     cnt_5=0
    
#     for sentence in parent_to_sentence[parent]:
#       if len(sentence.split())>5:
#         cnt_5+=1
    
#     if len(parent_to_sentence[parent])-cnt_5<=2:
#       return True
#     else:
#       return False





# def check_which_answer(hits,query):
#   sentence_1=passages[hits[0]['corpus_id']].split(':')[:-1]
#   sentence_2=passages[hits[0]['corpus_id']].split(':')[-1]
#   # print(passages[hits[0]['corpus_id']])
#   sentence_1=' : '.join(sentence_1)
  
#   cross_inp_1=[[query,sentence_1]]
#   cross_score_1=cross_encoder.predict(cross_inp_1)

#   cross_inp_2=[[query,sentence_2]]
#   cross_score_2=cross_encoder.predict(cross_inp_2)
# #   print(sentence_1)
# #   print(sentence_2)
# #   print()
# #   print(cross_score_1)
# #   print(cross_score_2)

#   if cross_score_1[0]>cross_score_2[0]:
#     return False
#   else:
#     return True

# def extracting_answers(answers):
#   answer=answers[0].split(':')
#   return answer[-1].strip()
#   #return answers

# def forming_passage(hits,passa=0):
#   if passa==0: 
#     i=0
#     passage=""
#     for hit in hits:
#       if hit['cross-score']>0 or True:
#         sentence_1=passages[hit['corpus_id']].split(':')[:-1]
#         sentence=passages[hit['corpus_id']].split(':')[-1]
#         sentence_1=str(sentence_1).strip()
#         sentence=str(sentence).strip()
#         # if sentence_1 in list(parent_to_sentence.keys()):
#         #   passage+=sentence+" \n "+' '.join(parent_to_sentence[sentence_1])
#         #   i+=1
#         #   continue
#         if sentence not in sentence_to_parents:
#             return sentence
#         parent_list=sentence_to_parents[sentence]
#         if  len(parent_list)>0:
#           parent=' : '.join(parent_list)
#           #j=1
#           for elem in parent_to_sentence[parent]:
#             passage+=elem+" "
#             #j+=1
#       else:
#         break
#       i+=1
#       if i==1 or len(passage.split())>1000:
#         break
#       # if i!=1:
#       #   passage+=' '
#     return passage

#   else:
#     i=0
#     passage=""
#     for hit in hits:
#       if hit['cross-score']>0 or True:
#         sentence_1=passa.split(':')[:-1]
#         sentence=passa.split(':')[-1]
#         sentence_1=str(sentence_1).strip()
#         sentence=str(sentence).strip()
#         # if sentence_1 in list(parent_to_sentence.keys()):
#         #   passage+=sentence+" \n "+' '.join(parent_to_sentence[sentence_1])
#         #   i+=1
#         #   continue
#         # print(sentence)
#         # print(passa)
#         parent_list=[]
#         if sentence in sentence_to_parents: 
#             parent_list=sentence_to_parents[sentence]
#         if  len(parent_list)>0:
#           parent=' : '.join(parent_list)
#           #j=1
#           for elem in parent_to_sentence[parent]:
#             passage+=elem+" "
#             #j+=1
#       else:
#         break
#       i+=1
#       if i==1 or len(passage.split())>1000:
#         break
#       # if i!=1:
#       #   passage+=' '
#     return passage





# # def forming_passage(hits,passages):
# #     passage=""
# #     hit=hits[0]
# #     sentence=passages[hit['corpus_id']].split(':')[-1]
# #     return sentence
# def forming_answer(hits,query):
   
#   if len(passages[hits[0]['corpus_id']].split(':')[-1].split()) <=4:
#     if check_which_answer(hits,query):
#       if passages[hits[0]['corpus_id']] in list(parent_to_sentence.keys()):
#           if passages[hits[0]['corpus_id']] in parent_to_sentence: 
#               passage=' '.join(parent_to_sentence[passages[hits[0]['corpus_id']]])
#               return passage
          
      

#   if check(hits):
    
#     hit = hits[0]
#     sentence_1=passages[hit['corpus_id']].split(':')[:-1]
#     sentence_2=passages[hit['corpus_id']].split(':')[-2:]
#     sentence_1=' : '.join(sentence_1)
#     #sentence_2=str(sentence_2).strip()
#     cross_inp_1 = [[query, sentence_1]]
#     cross_scores_1 = cross_encoder.predict(cross_inp_1)
#     sentence_2=' : '.join(sentence_2)
#     cross_inp_2 = [[query, sentence_2]]
#     cross_scores_2 = cross_encoder.predict(cross_inp_2)
# #     print(sentence_1)
# #     print(sentence_2)

# #     print(cross_scores_1)
# #     print(cross_scores_2)
#     if cross_scores_1[0] > cross_scores_2[0]:
#       dummy=dict()
#       dummy['cross-score']=1
#       return forming_passage([dummy],sentence_1)
#     else:
#       dummy=dict()
#       dummy['cross-score']=1
#       return forming_passage(hits)

#   else:
#     return forming_passage(hits)

# # def forming_answer(hits,query):
# #   return passages[hits[0]['corpus_id']].split(':')[-1]


def give_candidate(sentence):
  sentence=sentence.strip()
  #print(sentence)
  if str(sentence) in parent_to_sentence:
   
    if len(parent_to_sentence[sentence]) > 2:
      return parent_to_sentence[sentence]
    else:
      #ret=[sentence.split(':')[-1]+"  \n"]
      ret=list()
      for child in parent_to_sentence[sentence]:
        #print(sentence+" : "+child)
        ret.append(give_candidate(sentence+" : "+child))
      #print(ret)
      if ["NO ANSWER"] in ret:
        return parent_to_sentence[sentence]
      return ret
  
  else:

    for elem in list(parent_to_sentence.keys()):
      try:
        if re.search(sentence,elem):
            if re.search("figure",elem):
              return parent_to_sentence[elem]
      except:
        return ["NO ANSWER"]
    return ["NO ANSWER"]

def Candidates_no_dis(paragraph):
  r_head=""
  sentence_lst=paragraph.split(':')
  #HEADING 1
  try:
    if sentence_lst[-1]=='':
      sentence_lst=sentence_lst[:-1]
      sentence=':'.join(sentence_lst)
      sentence+=":"
    
    else:
      sentence=":".join(sentence_lst)
  except:
    sentence=":".join(sentence_lst)

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]
  candidate_1=heading+give_candidate(sentence)
  sentence=sentence.strip()
  sentence_1=sentence+" : "+"summary"

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]

  candidate_5=[r_head+" : "]+give_candidate(sentence_1)
  sentence=sentence.strip()
  sentence_2=sentence+" : "+"procedure"

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]

  candidate_6=[r_head+" : "]+give_candidate(sentence_2)

  sentence_3=sentence+" : "+"calloutlist"
  # print(sentence)
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]

  candidate_9=[r_head+" : "]+give_candidate(sentence_3)
  #HEADING 2
  sentence_lst=sentence_lst[:-1]
  try:
    if sentence_lst[-1]=='':
      sentence_lst=sentence_lst[:-1]
      sentence=':'.join(sentence_lst)
      sentence+=":"
    
    else:
      sentence=":".join(sentence_lst)
  except:
    sentence=":".join(sentence_lst)
  
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]  
  # print(heading)
  # print(sentence)
  candidate_2=[r_head+" : "]+give_candidate(sentence)
  sentence=sentence.strip()

  sentence_1=sentence+" : "+"summary"
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]
  candidate_7=[r_head+" : "]+give_candidate(sentence_1)
  sentence=sentence.strip()
  sentence_2=sentence+" : "+"procedure"
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]
  candidate_8=[r_head+" : "]+give_candidate(sentence_2)
  
  sentence_3=sentence+" : "+"calloutlist"

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
        pass
  heading=[r_head+" : "]

  candidate_10=[r_head+" : "]+give_candidate(sentence_3)
  #SUMMARY and PROCEDURE PART IN PARENT
  candidate_3=""
  if re.search("summary",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("summary",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
      else:
        pass
    heading=[r_head+" : "]
    candidate_3=[r_head+" : "]+give_candidate(sentence)

  candidate_4=""
  if re.search("procedure",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("procedure",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    sentence+=":"
    #print(sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
      else:
        pass
    heading=[r_head+" : "]
    candidate_4=[r_head+" : "]+give_candidate(sentence)


  candidate_11=""
  # print("Hello"+paragraph)
  if re.search("calloutlist",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("calloutlist",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    #sentence+=":"
    # print("Hello"+sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
      else:
        pass
    heading=[r_head+" : "]
    candidate_11=[r_head+" : "]+give_candidate(sentence)

  #Checking In Childs

  # candidate_1,candidate_2,candidate_3,candidate_4,candidate_5,candidate_6,candidate_7,candidate_8

  return [candidate_1,candidate_2,candidate_3,candidate_4,candidate_5,candidate_6,candidate_7,candidate_8,candidate_9,candidate_10,candidate_11]

def Candidates(paragraph):
  paragraph=paragraph.strip()
  special_candidate_1,special_candidate_2,special_candidate_3,special_candidate_4,special_candidate_5,special_candidate_6,special_candidate_7,special_candidate_8,special_candidate_9,special_candidate_10,special_candidate_11=["" for _ in range(11)]
  if re.search("figure",paragraph):
    special_candidate_1,special_candidate_2,special_candidate_3,special_candidate_4,special_candidate_5,special_candidate_6,special_candidate_7,special_candidate_8,special_candidate_9,special_candidate_10,special_candidate_11=Candidates_no_dis(paragraph)
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      if re.search("figure",elem):
        break
      
      tmp_sentence.append(elem)
    paragraph=':'.join(tmp_sentence)
  
  sentence_lst=paragraph.split(':')
  #HEADING 1
  try:
    if sentence_lst[-1]=='':
      sentence_lst=sentence_lst[:-1]
      sentence=':'.join(sentence_lst)
      sentence+=":"
    
    else:
      sentence=":".join(sentence_lst)
  except:
    sentence=":".join(sentence_lst)

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    else:
      pass
  heading=[r_head+" : "]
  candidate_1=heading+give_candidate(sentence)
  sentence=sentence.strip()
  sentence_1=sentence+" : "+"summary"

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
      
  heading=[r_head+" : "]

  candidate_5=[r_head+" : "]+give_candidate(sentence_1)
  sentence=sentence.strip()
  sentence_2=sentence+" : "+"procedure :"
  #print("Hello"+sentence_2)
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
     
  heading=[r_head+" : "]

  candidate_6=[r_head+" : "]+give_candidate(sentence_2)

  sentence_3=sentence+" : "+"calloutlist"
  # print(sentence)
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
      
  heading=[r_head+" : "]

  candidate_9=[r_head+" : "]+give_candidate(sentence_3)
  #HEADING 2
  sentence_lst=sentence_lst[:-1]
  
  if len(sentence_lst)!=0 and sentence_lst[-1]=='':
    sentence_lst=sentence_lst[:-1]
    sentence=':'.join(sentence_lst)
    sentence+=":"
  
  elif len(sentence_lst)!=0:
    sentence=":".join(sentence_lst)
  
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
  heading=[r_head+" : "]  
  # print(heading)
  
  candidate_2=[r_head+" : "]+give_candidate(sentence)
  sentence=sentence.strip()
  sentence_1=sentence+" : "+"summary"
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
  heading=[r_head+" : "]
  candidate_7=[r_head+" : "]+give_candidate(sentence_1)
  sentence=sentence.strip()
  sentence_2=sentence+" : "+"procedure :"
 
  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
  heading=[r_head+" : "]
  candidate_8=[r_head+" : "]+give_candidate(sentence_2)
  
  sentence_3=sentence+" : "+"calloutlist"

  if len(sentence.split(':')[-1].strip())!=0:
    r_head=sentence.split(':')[-1]
  else:
    if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
  heading=[r_head+" : "]

  candidate_10=[r_head+" : "]+give_candidate(sentence_3)
  #SUMMARY and PROCEDURE PART IN PARENT
  candidate_3=""
  if re.search("summary",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("summary",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    heading=[r_head+" : "]
    candidate_3=[r_head+" : "]+give_candidate(sentence)

  candidate_4=""
  if re.search("procedure",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("procedure",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    sentence+=":"
    #print(sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    heading=[r_head+" : "]
    candidate_4=[r_head+" : "]+give_candidate(sentence)


  candidate_11=""
  # print("Hello"+paragraph)
  if re.search("calloutlist",paragraph):
    tmp_sentence=list()
    for elem in paragraph.split(':'):
      tmp_sentence.append(elem)

      if(re.search("calloutlist",elem)):
        break
    
    sentence=':'.join(tmp_sentence)
    #sentence+=":"
    # print("Hello"+sentence)
    if len(sentence.split(':')[-1].strip())!=0:
      r_head=sentence.split(':')[-1]
    else:
      if len(sentence.split(':')) >=2:
        r_head=sentence.split(':')[-2]
    heading=[r_head+" : "]
    candidate_11=[r_head+" : "]+give_candidate(sentence)

  #Checking In Childs

  # candidate_1,candidate_2,candidate_3,candidate_4,candidate_5,candidate_6,candidate_7,candidate_8

  return candidate_1,candidate_2,candidate_3,candidate_4,candidate_5,candidate_6,candidate_7,candidate_8,candidate_9,candidate_10,candidate_11,special_candidate_1,special_candidate_2,special_candidate_3,special_candidate_4,special_candidate_5,special_candidate_6,special_candidate_7,special_candidate_8,special_candidate_9,special_candidate_10,special_candidate_11

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def prepare_cross_input(candidates,is_rec=0):
  inputs=list()
  for candidate in candidates:
    if type(candidate)==str and len(candidate.strip())!=0:
      inputs.append(candidate)
      continue
    if "NO ANSWER" in candidate:
      continue
    if len(candidate)==0:
      continue
    if is_rec==0:
      if type(candidate[0])==str:
        inp=candidate[0]+" "
      else:
        candidate[0]=flatten(candidate[0])
        inp=candidate[0][0]+" "
    else:
      inp=""
    for i in range(len(candidate)):
      
      #print(candidate[i])
      #harhs_exit()
      if i==0 and is_rec==0:
        continue
      if type(candidate[i])!=str:
        candidate[i] = flatten(candidate[i])
        if len(prepare_cross_input(candidate[i],1))>0:
          inp+=prepare_cross_input(candidate[i],1)[0]
        continue
      if i!=len(candidate)-1:
        # if type(candidate[i])=="str":
        inp+=candidate[i]+", "
        # else:
          
        #   #print(prepare_cross_input([candidate[i]],1))
        #   inp+=prepare_cross_input([candidate[i]],1)[0]
      else:
        # if type(candidate[i])=="str":
        inp+=candidate[i]
        # else:
        #   inp+=prepare_cross_input([candidate[i]],1)[0]
    inputs.append(inp)
  
  return inputs

def give_scores(candidates):

  cross_inp = [[query, candidate] for candidate in candidates]
  cross_scores = cross_encoder.predict(cross_inp)
  candidates_ranked=list()
  i=0
  for candidate in candidates:
    tmp=dict()
    tmp['sentence']=candidate
    tmp['cross-score']=cross_scores[i]
    i+=1
    candidates_ranked.append(tmp)
  # print(candidates_ranked)
  answers = sorted(candidates_ranked,key= lambda x : x['cross-score'],reverse=True)
  return answers

def choose_one(answers):
  return answers[0]['sentence'].strip()

def some_modification(candidates,hit):
  passage=passages[hit['corpus_id']]
  #print(passage)
  if re.search('calloutlist',passage):
    replace_word="calloutlist"
    passage_lst=passage.split(':')
    for i in range(len(passage_lst)):
      if passage_lst[i].strip()=='calloutlist':
        j=i-1
        while j>=0:
          if passage_lst[j].strip()!='figure' and passage_lst[j].strip()!='summary' and passage_lst[j].strip()!='Method' and passage_lst[j].strip()!='' and passage_lst[j].strip()!='procedure':
            replace_word=passage_lst[j]
            break
          j=j-1
    ret_candidates=list()
    for elem in candidates:
      ret_candidates.append(elem.replace("calloutlist",replace_word))
    return ret_candidates
  else:
    return candidates


def give_answer(query):
  hits=search(query = query)
  to_send=passages[hits[0]['corpus_id']]
  #print(to_send)
  candidates=prepare_cross_input(Candidates(to_send))
  candidates=some_modification(candidates,hits[0])
# re_ranked=give_scores(candidates)
# # for answer in re_ranked:
# #   # print(answer)
# #   print(str(answer['cross-score'])+"  "+answer['sentence'])

# choose_one(re_ranked)
  re_ranked=give_scores(candidates)
  # for answer in re_ranked:
  #   # print(answer)
  #   print(str(answer['cross-score'])+"  "+answer['sentence'])

  return choose_one(re_ranked)



def generate_passage(query_1):
  global query
  query=query_1
  # answer=give_answer(query)
  # answer_lst=answer.split(':')
  # answer=answer_lst[1:]
     
  # answer=''.join(answer)
  # #print(answer)
  # #harsh_exit()  
  return give_answer(query)
