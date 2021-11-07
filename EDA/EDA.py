import pandas as pd
import numpy as np
import json
import gc

import matplotlib.pyplot as plt
import seaborn as sns

from pandas import json_normalize
from collections import deque

#json load
with open('/opt/ml/segmentation/input/data/train_all.json', 'r') as f:
    train_dict = json.load(f)

## object check
i = 0
while i < 26240:
    
    flag = 0
    check_pt = train_dict['annotations'][i]
    if check_pt['category_id'] == 10:
        flag = 1
        print('image id is', check_pt['image_id'])
        print('segementation', check_pt['segmentation'])
    i += 1
if flag == 0:
    print("NaNN")

#def fuc for collecting coordinate
def cnt_coordinate(category_id, data):
    global board
    
    if data['category_id'] == category_id:
        seg_lst = data['segmentation']
        
        i = 0
        while i < len(seg_lst):
            storage = deque(seg_lst[i])
            
            while 1:
                x = storage.popleft()
                y = storage.popleft()
            
                board[x][y] += 1
                if len(storage) == 0:
                    break
            i += 1
    return board

#class 별로 합산
## background:0 제외
id_num = 1
N = len(train_dict['annotations'])
board_lst = [0]
while id_num < 12:
    
    anno_num = 0
    board = [[0] * 513 for _ in range(513)]
    while anno_num < N:
        cnt_coordinate(id_num, train_dict['annotations'][anno_num])
        anno_num += 1
    board_lst.append(board)
    id_num += 1

#def fuc for subsum
def ft_subsum(n,arr):
    
    y = 0
    sub_arr = []
    while y < len(arr):
        
        x = 0
        row = []
        while x < len(arr[0]):
            
            sum_int = arr[y:y + n,x:x + n].sum()
            row.append(sum_int)
            x += n
        sub_arr.append(row)
        y += n
    return sub_arr

## 가장자리 1줄씩 삭제
i = 1
np_board_lst = [0]
while i < 12:
    tmp = np.array(board_lst[i])
    
    tmp = np.delete(tmp,0,0)
    tmp = np.delete(tmp,511,0)
    tmp = np.delete(tmp,0,1)
    tmp = np.delete(tmp,511,1)
    
    np_board_lst.append(tmp)
    i += 1

## 51x51로 묶음
i = 1
np_board_51x51 = [0]
while i < 12:
    tmp = ft_subsum(51, np_board_lst[i])
    np_board_51x51.append(tmp)
    i += 1
## np_board_51x51 : 51 x 51 pixcel matrix를 한 칸으로 본 클래스별 분포도

# normalization
## 확인하려는 class를 np_board_51x51의 index로 입력
normalization_df = (pd.DataFrame(np_board_51x51[1]) - pd.DataFrame(np_board_51x51[1]).mean())/pd.DataFrame(np_board_51x51[1]).std()

#heatmap
fig = plt.figure(figsize=(12,10))
fig.set_facecolor('white')
plt.pcolor(normalization_df.values, cmap='YlGn')
plt.xticks(range(len(normalization_df.columns)), normalization_df.columns)
plt.yticks(range(len(normalization_df.index)), normalization_df.index)
plt.title('Coordinate of (1) General Trash', fontsize=20) ## class 이름 수정
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.colorbar()
plt.savefig('CoordinateHeatmap_(1)generaltrash_51x51.png', dpi=200) ## class 이름 수정
plt.show()

