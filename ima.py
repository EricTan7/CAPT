import os
import pickle
import json

path2id = {}
name2path = {}
nori_path = '/data/datasets/imagenet/imagenet.train.nori.list'

with open('/data/datasets/imagenet/path2id.json','r') as f:
    path2id = json.load(f)

# with open('/data/datasets/imagenet/preprocessed.pkl', "rb") as f:
#     preprocessed = pickle.load(f)
#     test = preprocessed['test']
#     for item in test:
#         name2path[ item['impath'].split('/')[-1] ] = os.path.join(*item['impath'].split('/')[-3:])

with open(nori_path) as g:
    l = g.readline()
    while l:
        ls = l.split()
        path = name2path[ls[2]]
        print(path)
        path2id[path] = ls[0]
        l = g.readline()

# with open('/data/datasets/imagenet/path2id.json','w') as f:
#     json.dump(path2id,f)

with open('/data/datasets/imagenet/wval_split_fewshot_caption/shot_16-seed_1.pkl', "rb") as f:
    preprocessed = pickle.load(f)
    for item in preprocessed['train']:
        path = item['impath']
        if path.split('/')[-3] != 'train':
            print(path)
    for item in preprocessed['val']:
        path = item['impath']
        if path.split('/')[-3] != 'train':
            print(path)
    
#     a = 1

# nori_path = '/data/datasets/imagenet/imagenet.val.nori.list'
# with open(nori_path) as g:
#     l = g.readline()
#     while l:
#         ls = l.split()
#         path = os.path.join('val/val', ls[2])
#         path2id[path] = ls[0]   