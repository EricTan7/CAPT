import os
import pickle
from .basic import Benchmark, read_split, save_split, read_and_split_data, \
    generate_fewshot_dataset, subsample_classes, split_trainval, read_split_caption
from tools.utils import mkdir_if_missing
from clip import clip
from transformers import BertTokenizer, T5Tokenizer


class SUN397(Benchmark):

    dataset_dir = "sun397"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")
        if "bert" in cfg.TRAINER.NAME:
            self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.TEXT.ENCODER)  # 'bert-base-uncased'
        elif "t5" in cfg.TRAINER.NAME:
            self.tokenizer = T5Tokenizer.from_pretrained(cfg.MODEL.TEXT.ENCODER)

        if cfg.MODEL.CAPTION:
            if "bert" in cfg.TRAINER.NAME:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_caption_{cfg.MODEL.TEXT.ENCODER}")
            elif "t5" in cfg.TRAINER.NAME:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_caption_{cfg.MODEL.TEXT.ENCODER.split('/')[-1]}")
            else:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_caption")
            caption = dict()
            tokenized_caption = dict()
            caption_path = os.path.join(self.dataset_dir, "captions_p2_train.txt")
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split('\t')
                    caption[line[0]] = line[1]
                    if "bert" in cfg.TRAINER.NAME:
                        tokenized_caption[line[0]] = \
                        self.tokenizer(line[1], padding='max_length', max_length=77, return_tensors='pt')['input_ids'][0]
                    elif "t5" in cfg.TRAINER.NAME:
                        tokenized_caption[line[0]] = \
                        self.tokenizer(line[1], padding='max_length', max_length=77, return_tensors='pt')['input_ids'][0]
                    else:
                        tokenized_caption[line[0]] = clip.tokenize(line[1])[0]
        else:
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            if cfg.MODEL.CAPTION:
                train, val, test = read_split_caption(self.split_path, self.image_dir, caption, tokenized_caption)
            else:
                train, val, test = read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:]  # remove /
                    classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            trainval = self.read_data(cname2lab, "Training_01.txt")
            test = self.read_data(cname2lab, "Testing_01.txt")
            train, val = split_trainval(trainval)
            save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = generate_fewshot_dataset(train, num_shots=num_shots)['data']
                val = generate_fewshot_dataset(val, num_shots=min(num_shots, 4))['data']
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                items.append(item)

        return items
