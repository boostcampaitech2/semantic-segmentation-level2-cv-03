from skimage import measure
import numpy as np
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [int(0) if i < 0 else int(i) for i in segmentation]
        polygons.append(segmentation)

    return polygons

parser = argparse.ArgumentParser()
parser.add_argument('--pseudo_csv', type=str, default='/opt/ml/unilm/beit/semantic_segmentation/sample_submission.csv')
parser.add_argument('--best_submission', type=str, default='/opt/ml/unilm/beit/semantic_segmentation/work_dirs/custom_beit_slr_sr_aug.csv')
parser.add_argument('--save_path', type=str, default='pseudo_dataset.json')
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    fold_path = '/opt/ml/segmentation/input/data/sr_train.json'

    pesudo = pd.read_csv(args.pseudo_csv)      # csv with target label image ids
    best_submission = pd.read_csv(args.best_submission)      # submission csv without (256,256) resize

    # Read annotations
    with open(fold_path, 'r') as f:
        dataset = json.loads(f.read())

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    new_images = []
    new_annotations = []

    image_dict_id = 0
    annotation_dict_id = annotations[-1]['id'] + 1
    for id_ in tqdm(range(pesudo.shape[0])):
        image_dict_id += 1
        images_dict = {}
        images_dict['license'] = 0
        images_dict['url'] = None
        images_dict['file_name'] = pesudo.loc[id_, 'image_id']
        images_dict['height'] = 512
        images_dict['width'] = 512
        images_dict['date_captured'] = None
        images_dict['id'] = image_dict_id
        new_images += [images_dict]
        for i in range(1, 11): 
            pesudo_dict = {}

            A = np.zeros((512, 512))
            mask = np.array(list(map(int, re.findall("\d+", best_submission[best_submission['image_id'] == pesudo.loc[id_, 'image_id']]['PredictionString'].values[0])))).reshape(512, 512)
            x, y = np.where(mask==i)

            L = []
            for x_, y_ in zip(x, y): 
                L += [(x_, y_)]

            if len(L) != 0: 
                idx = np.r_[L].T
                A[idx[0], idx[1]] = 1
                annotation_dict_id += 1
                pesudo_dict['id'] = annotation_dict_id
                pesudo_dict['image_id'] = image_dict_id
                pesudo_dict['category_id'] = i
                pesudo_dict['segmentation'] = binary_mask_to_polygon(A)
                pesudo_dict['area'] = 0
                pesudo_dict['bbox'] = [0, 0, 0, 0]
                pesudo_dict['iscrowd'] = 0
                new_annotations += [pesudo_dict]

    train_ann = {}
    train_ann['images'] =  new_images
    train_ann['annotations'] = new_annotations
    train_ann['categories'] = categories

    # save path
    save_path = args.save_path

    with open(save_path, 'w') as f:
        json.dump(train_ann, f, indent=4)