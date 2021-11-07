import pandas as pd
import json
from scipy import stats
import numpy as np
dataset_path  = '../../../input/data'
anns_file_path = dataset_path + '/' + 'train.json'

# Read annotations
new_dic={}
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

anns = dataset['annotations']
for i,v in enumerate(anns):
    each={}            
    each['id']    = (v['id'])
    each['area']      = v['area']
    each['image_id'] = v['image_id']
    each['category_id'] = v['category_id']
    each['segmentation']= v['segmentation']
    each['bbox']=v['bbox']
    each['iscrowd']=v['iscrowd']
    new_dic[i] = each
print(type(new_dic))
area_pd = pd.DataFrame(new_dic.values())
area_pd['zscore'] = stats.zscore(area_pd.area)
new_dic={}
for i,v in enumerate(anns):
    if np.abs(area_pd['zscore'][i])>1.2 and v['category_id']==1:
        each={}            
        each['id']    = (v['id'])
        each['area']      = v['area']
        each['image_id'] = v['image_id']
        each['category_id'] = v['category_id']
        each['segmentation']= v['segmentation']
        each['bbox']=v['bbox']
        each['iscrowd']=v['iscrowd']
        new_dic[i] = each
#tmp = pd.DataFrame(new_dic.values())
#print((dataset['annotations'][0]))
# for i in list(dataset.keys()):
#     if dataset[i] in new_dic:
#         del dataset[i]
print(len(area_pd['zscore']>0))
print(len(dataset['annotations']))
new_list=[]
avg_area=[]
for i ,v in enumerate(dataset['annotations']):
    if (area_pd['zscore'][i])>-0.56:
        each={}
        each['id']    = (v['id'])
        each['area']      = v['area']
        each['image_id'] = v['image_id']
        each['category_id'] = v['category_id']
        each['segmentation']= v['segmentation']
        each['bbox']=v['bbox']
        each['iscrowd']=v['iscrowd']
        new_list.append(each)
        avg_area.append(v['area'])
print(sum(avg_area)/len(avg_area))
dataset['annotations']=new_list
print(len(dataset['annotations']))
with open('/opt/ml/segmentation/baseline_code/base/eda/train_zscore.json','w') as f:
    json.dump(dataset,f)
