import json
import re
import copy
import sys
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
os.environ['CUDA_VISIBLE_DEVICES']="3"

if len(sys.argv)!=3:
    print("please enter all the arguments")
    exit()

manual_path=sys.argv[1]
name_and_model=sys.argv[2]

if not os.path.isdir("./manuals/"+name_and_model):
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
    # print(cnt)
    # cnt+=1
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

            #It ends here
            if elem1.find("step")==-1:
                parent.append(elem1)
            else:
                parent.append("")
            check=preparing_passages(data[elem])
            if(check):
                tmp=parent.pop()
                if tmp in passages:
                    passages[tmp].append(copy.deepcopy(parent))
                else:
                    passages[tmp]=[copy.deepcopy(parent)]
                parent.append(tmp)
            parent.pop()
    else:
        parent.append(data)
        #check=preparing_passages(data[elem])
        #if(check):
        tmp=parent.pop()
        if tmp in passages:
            passages[tmp].append(copy.deepcopy(parent))
        else:
            passages[tmp]=[copy.deepcopy(parent)]
        # parent.append(tmp)
    return True


def preparing_passages_cause_reason(data):
    global passages,parent,cnt
    # print(cnt)
    # cnt+=1
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

            #It ends here
            
            if elem1.find("troublelistentry")!=-1:
                item=""
                problem=""
                for e in data[elem]['problem'].keys(): 
                    problem=e
                
                for key in data[elem].keys():
                    answer=""
                    if key!='problem':
                        answer+="reason = "
                        tmp=""
                        for reason in data[elem][key]['reason']:
                            answer+=reason
                            tmp=reason
                        answer+=" \n solution = "
                        solutions=data[elem][key]['solution'] 
                        
                        for solution in solutions:
                            answer+=solution+" "
                            
                            
                            if solution not in passages:

                                passages[solution]=list()

                            passages[solution].append(tmp)
                            
                            
                        put_solution=answer 
                        if put_solution not in passages:
                            passages[put_solution]=list()
                        passages[put_solution].append(problem)

            check=preparing_passages_cause_reason(data[elem])
    return True

def preparing_passages_table(data):
    global passages,parent,cnt
    # print(cnt)
    # cnt+=1
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

            #It ends here
            
            if not elem1.find("row") == -1:
                if len(data[elem])==2:
                    try:
                        sentence=""
                        for key in data[elem]['entry_1']:
                            if key.find("colname")==-1:
                                key1 = key.replace('*', '<')
                                key1=re.sub('<[^<]+<', '', key1)
                                key1=re.sub('<[^<]*<', '', key1)


                                key1=re.sub('#[^#]+#', '', key1)
                                key1=re.sub('#[^#]*#', '', key1)
                                sentence+=key1+" " 
                        
                        for key in data[elem]['entry']:
                            if key.find("@")==-1:
                                if sentence not in passages:
                                    passages[sentence]=[[]]
                                key1 = key.replace('*', '<')
                                key1=re.sub('<[^<]+<', '', key1)
                                key1=re.sub('<[^<]*<', '', key1)


                                key1=re.sub('#[^#]+#', '', key1)
                                key1=re.sub('#[^#]*#', '', key1)
                                passages[sentence][0].append(key1)
                    except:
                        pass

            check=preparing_passages_table(data[elem])
    return True

def preparing_specification(data):
    global passages,parent,cnt
    # print(cnt)
    # cnt+=1
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

            #It ends here
            
            if not elem1.lower().find("specifications") == -1:
                
                try:
                    for key in data[elem]['variablelist'].keys():
                        #print(data[elem]['variablelist'][key].keys())
                        for value in data[elem]['variablelist'][key].keys():
                            
                            if len(value.split(':'))==2:
                                #print(value.split(':'))
                                sentence=value.split(':')[0]
                                sentence1 = sentence.replace('*', '<')
                                sentence1=re.sub('<[^<]+<', '', sentence1)
                                sentence1=re.sub('<[^<]*<', '', sentence1)


                                sentence1=re.sub('#[^#]+#', '', sentence1)
                                sentence1=re.sub('#[^#]*#', '', sentence1)
                                if value.split(':')[1] not in passages:
                                    passages[value.split(':')[1]]=[[]]
                                #print(sentence1+" "+value.split(':')[1])
                                passages[value.split(':')[1]][0].append(sentence1)
                except:
                    #print("not here")
                    pass
                                  
                
            check=preparing_specification(data[elem])
    return True

    

ret_data=preparing_passages(data)


parent=list()
cnt=0

#Call preparing_passages_cause_reason
ret_data=preparing_passages_cause_reason(data)

ret_data=preparing_passages_table(data)

ret_data=preparing_specification(data)
with open("./manuals/"+name_and_model+'/sentence_to_parent_list.json', 'w') as json_file:
  json.dump(passages, json_file)
#passages_json = json.dumps(ret_data)


data=passages

data_1=list()
data_1=list(data.keys())
data_2=list(data.values())
print(type(data_1))

parent_to_sentence=dict()

for i in range(len(data_2)):
    for j in range(len(data_2[i])):
        #parent=list()
    
        if len(data_2[i][j])>0:
            tmp=list()
            #print(type(data_2[i][j]))
            if not isinstance(data_2[i][j], str):
                for elem in data_2[i][j]:
                    tmp.append(elem)
                    
                for elem in data_2[i][j]:
                    if(len(elem.split())>6):
                        tmp.remove(elem)
                

                parent=' : '.join(tmp)
            else:
                parent=data_2[i][j]
            parent=parent.strip()
            if parent in parent_to_sentence:
                parent_to_sentence[parent].append(data_1[i])
            else:
                parent_to_sentence[parent]=[data_1[i]]

with open("./manuals/"+name_and_model+'/parent_to_sentence_list.json', 'w') as json_file:
  json.dump(parent_to_sentence, json_file)
f.close()

sentence_to_parent=passages
sentences=set()


child=list(sentence_to_parent.keys())
parents=list(sentence_to_parent.values())

for i in range(len(child)):
    
    for j in range(len(parents[i])):
        tmp=list()
        if not isinstance(parents[i][j],str):
            for elem in parents[i][j]:
                tmp.append(elem)
                
            for elem in parents[i][j]:
                if(len(elem.split())>6):
                    tmp.remove(elem)
            
            sentences.add(' : '.join(tmp))

        else:
            sentences.add(parents[i][j])



sentences_1=list()
for elem in sentences:
    sentences_1.append(elem)

with open("./manuals/"+name_and_model+'/headings.json', 'w') as json_file:
  json.dump(sentences_1, json_file)
#print(len(sentences_1))
corpus_embeddings = bi_encoder.encode(sentences_1, convert_to_tensor=True, show_progress_bar=True)
torch.save(corpus_embeddings,"./manuals/"+name_and_model+"/embeddings_list.pt")

