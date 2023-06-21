import os
import nori2 as nori
import json
from tqdm import tqdm

path = "/data/datasets/sun397/SUN397" #文件夹目录
path2id = {}

assert os.path.isdir(path)

nw = nori.open("/data/nori/sun397.nori", "w")    # open an empty nori dataset for writing

for root, dirs, files in os.walk(path):
     #for dir in tqdm(dirs):
          #files= os.listdir(os.path.join(root, dir))
          for file in tqdm(files):
            if '.jpg' in file or '.png' in file or '.jpeg' in file or '.JPEG' in file or '.JPG' in file or '.PNG' in file:
               f = open(os.path.join(root, file), "rb")
     
               filedata = f.read()

               # data_id is the new nori DataID of this file
               # use this data_id for your own purpose
               data_id = nw.put(filedata, filename=file)
               file_name = os.path.join(*root.split('/')[-2:], file)
               path2id[file_name] = data_id
               f.close()
            else:
                 print(f'not load {file}')


# file.close() when use fileobj in tarfile.open(), fileobj will not be closed
nw.close()
with open('/data/datasets/sun397/path2id.json', 'w') as f:
     json.dump(path2id, f)