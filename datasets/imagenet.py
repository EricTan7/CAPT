import os
import pickle
from collections import OrderedDict
from .basic import Benchmark, read_split, read_split_imagenet, save_split, read_and_split_data, generate_fewshot_dataset, subsample_classes
from tools.utils import listdir_nohidden, mkdir_if_missing
from clip import clip
from transformers import BertTokenizer, T5Tokenizer


class ImageNet(Benchmark):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")      # followed CoOp, dont have val set
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.TEXT.ENCODER)
        if cfg.MODEL.CAPTION:
            if "bert" in cfg.TRAINER.NAME:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, f"split_fewshot_caption_{cfg.MODEL.TEXT.ENCODER}")
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
                        tokenized_caption[line[0]] = self.tokenizer(line[1], padding='max_length', max_length=77, return_tensors='pt')['input_ids'][0]
                    else:
                        tokenized_caption[line[0]] = clip.tokenize(line[1])[0]
        else:
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = generate_fewshot_dataset(train, num_shots=num_shots)['data']
                # NOTE if seed is the same, but generate diff results, the caption wont work
                if cfg.MODEL.CAPTION:
                    for i in range(len(train)):
                        # caption, tokenized_caption
                        imname = train[i]['impath'][28:]
                        train[i]['caption'] = caption[imname]
                        train[i]['tokenized_caption'] = tokenized_caption[imname]
                    for i in range(len(test)):
                        test[i]['caption'] = None
                        test[i]['tokenized_caption'] = None
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = subsample_classes(train, test, subsample=subsample)

        super().__init__(train=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                items.append(item)

        return items


class ImageNet_wval(Benchmark):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_path = os.path.join(self.dataset_dir, "split_ImageNet.json")

        val = read_split_imagenet(self.split_path, self.image_dir)
        # # Uncomment the following lines to generate a new split
        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = self.read_classnames(text_file)
        # train = self.read_data(classnames, "train")
        # # Follow standard practice to perform evaluation on the val set
        # # Also used as the val set (so evaluate the last-step model)
        # test = self.read_data(classnames, "val")
        # # If you want to generate new split, uncomment the following lines.
        # train, val = split_trainval(train)
        # save_split(train, val, test, self.split_path, self.image_dir)

        if "bert" in cfg.TRAINER.NAME:
            self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.TEXT.ENCODER)  # 'bert-base-uncased'
        elif "t5" in cfg.TRAINER.NAME:
            self.tokenizer = T5Tokenizer.from_pretrained(cfg.MODEL.TEXT.ENCODER)
        if cfg.MODEL.CAPTION:
            if "bert" in cfg.TRAINER.NAME:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, f"wval_split_fewshot_caption_{cfg.MODEL.TEXT.ENCODER}")
            elif "t5" in cfg.TRAINER.NAME:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, f"wval_split_fewshot_caption_{cfg.MODEL.TEXT.ENCODER.split('/')[-1]}")
            else:
                self.split_fewshot_dir = os.path.join(self.dataset_dir, "wval_split_fewshot_caption")
            caption = dict()
            tokenized_caption = dict()
            caption_path = os.path.join(self.dataset_dir, "captions_p2_train.txt")
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split('\t')
                    caption[line[0]] = line[1]
                    if "bert" in cfg.TRAINER.NAME:
                        tokenized_caption[line[0]] = self.tokenizer(line[1], padding='max_length', max_length=77, return_tensors='pt')['input_ids'][0]
                    elif "t5" in cfg.TRAINER.NAME:
                        tokenized_caption[line[0]] = self.tokenizer(line[1], padding='max_length', max_length=77, return_tensors='pt')['input_ids'][0]
                    else:
                        tokenized_caption[line[0]] = clip.tokenize(line[1])[0]
        else:
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "wval_split_fewshot_baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
                    val = data["val"]
            else:
                train = generate_fewshot_dataset(train, num_shots=num_shots)['data']
                val = generate_fewshot_dataset(val, num_shots=max(num_shots, 8))['data']
                # NOTE if seed is the same, but generate diff results, the caption wont work
                if cfg.MODEL.CAPTION:
                    for i in range(len(train)):
                        # caption, tokenized_caption
                        imname = train[i]['impath'][28:]
                        # imname = os.path.join("imagenet/images", train[i]['impath'])
                        train[i]['caption'] = caption[imname]
                        train[i]['tokenized_caption'] = tokenized_caption[imname]
                    for i in range(len(val)):
                        val[i]['caption'] = None
                        val[i]['tokenized_caption'] = None
                    for i in range(len(test)):
                        test[i]['caption'] = None
                        test[i]['tokenized_caption'] = None
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = subsample_classes(train, test, subsample=subsample)

        super().__init__(train=train, val=val, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                items.append(item)

        return items

