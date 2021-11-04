import os
from tqdm import tqdm
import numpy as np
from collections import Counter
import json

train_all_path = '/opt/ml/segmentation/input/data/train_all.json'
train_all = json.load(open(train_all_path, 'r'))

images = train_all['images']
annots = train_all['annotations']
print(train_all.keys())
print(images[0].keys())
print(annots[0].keys())
print(train_all['categories'])

cats = [x['category_id'] for x in annots]
cnt = Counter(cats)

cat_dist = sorted(cnt.items(), key=lambda x: x[0])
xs = [str(x[0]) for x in cat_dist]
ys = [x[1] for x in cat_dist]


id2fname = []
images = sorted(images, key=lambda x: x['id'])
for img in images:
    id2fname.append(img['file_name'])


file_annots = {}
for ann in tqdm(annots):
    fname = id2fname[ann['image_id']]

    if fname not in file_annots.keys():
        file_annots[fname] = np.array([0 for _ in range(10)])
    file_annots[fname][ann['category_id']-1] += 1


class Pocket:
    def __init__(self):
        self.ranks = []
        for i in tqdm(range(10)):
            sorted_fname = sorted(file_annots.items(), key=lambda x: x[1][i])
            self.ranks.append(sorted_fname)
        for i in range(10):
            print(len(self.ranks[i]))
    
    def get_file(self, cls: int):
        fannot = self.ranks[cls].pop()
        for i in range(10):
            for idx, item in enumerate(self.ranks[i]):
                if item[0] == fannot[0]:
                    self.ranks[i].pop(idx)
        return fannot

    def is_empty(self):
        assert len(set( [len(x) for x in self.ranks] )) == 1
        return len(self.ranks[0]) == 0


n_fold = 5
folds = [{'fname': [], 'dist': np.array([0 for _ in range(10)])} for _ in range(n_fold)]


base_dist = ys
def get_inferior(curr_dist):
    diff = (base_dist - curr_dist) / base_dist
    return diff.argmin()