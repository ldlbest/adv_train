import pathlib
import json
from typing import List
import torch
import fairseq
from fairseq.binarizer import Binarizer
from typing import Optional
from transformers import AutoTokenizer
from fairseq.data import data_utils
from fairseq.binarizer import LegacyBinarizer
from fairseq.file_chunker_utils import find_offsets
from fairseq.data import indexed_dataset
import numpy as np
from sklearn.model_selection import train_test_split

class DataFileManager:
    def __init__(self,) -> None:
        ...
    @staticmethod
    def read_file(self,file_path):
        with pathlib.Path(file_path).open("r") as f:
            data=f.read()
            data=json.loads(data)
        return data
    @staticmethod
    def write_file(self,write_path,datalist):
        with pathlib.Path(write_path).open("w") as f:
            json.dump(datalist,f,indent=4)


class MultiDataFileManager(DataFileManager):
    def __init__(self, filelist) -> None:
        super().__init__()
        self.filelist=filelist

    def merge_file(self,
                   write_path
                   ):
        all_data=self.merged_data()
        self.write_file(write_path=write_path,datalist=all_data)

    @property
    def merged_data(self,):
        all_data=[]
        for file in self.filelist:
            data=DataFileManager.read_file(file)
            all_data+=data
        return all_data

   
class DataProcessor():
    def __init__(self,datafile) -> None:
        self.datafile=datafile
    @property
    def number_of_dataset(self,
                        )->int:
        all_data=DataFileManager.read_file(self.datafile)
        return len(all_data)
    

class SequenceClassificationDataProcessor(DataProcessor):
    def __init__(self,
                 datafile:List[str],
                 task_type:Optional[str]) -> None:
        
        super().__init__(datafile)
        self.task_type=task_type

    def clean_text(self,text:str):
        ...
        
    def to_one_hot(self,
                  labels:Optional[List[str]]
                  ):
        if self.task_type=="mutilabel":
            one_hot=[0]*len(self.all_labels)
            for lb in labels:
                idx=self.id2label[lb]
                one_hot[idx]=1
        else:
            one_hot=self.label2id[labels[0]]
        return one_hot
    
    def generate_id2label(self):
        torch.save(self.id2label,"./ref/label2id.pt")

    def generate_label2id(self):
        torch.save(self.label2id,"./ref/label2id.pt")

    def generate_all_labels(self):
        labels=self.all_labels
        torch.save(labels,"./ref/all_labels.pt")

    @property
    def all_labels(self):
        all_label=set()
        all_data=DataFileManager.read_file(self.datafile)
        for data in all_data:
            for lb in all_label:
                all_label.add(lb)
        return sorted(all_label)
    @property
    def label2id(self):
        all_labels=self.all_labels
        label2id={lb:idx for idx,lb in enumerate(all_labels)}
        return label2id
    @property
    def id2label(self):
        all_labels=self.all_labels
        id2label={idx:lb for idx,lb in enumerate(all_labels)}
        return id2label
    
    def get_tok_Y(self,tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        source = []
        labels = []
        id2label = self.id2label
        label2id = self.label2id
        data = DataFileManager.read_file(self.datafile)
        for line in data:
            source.append(tokenizer.encode(line['text'].strip().lower(), truncation=True))
            labels.append(line['labels'])

        with open('tok.txt', 'w') as f:
            for s in source:
                f.writelines(' '.join(map(lambda x: str(x), s)) + '\n')

        with open('Y.txt', 'w') as f:
            if self.task_type=="multilabel":
                one_hot=[0]*len(id2label)
                for lb in labels:
                    for i in lb:
                        one_hot[label2id[i]]=1
            else:
                for lb in labels:
                    one_hot=[label2id[lb[0]]]
            f.writelines(' '.join(map(lambda x: str(x), one_hot)) + '\n')

        for data_path in ['tok', 'Y']:
            offsets = find_offsets(data_path + '.txt', 1)
            ds = indexed_dataset.make_builder(
                data_path + '.bin',
                impl='mmap',
                vocab_size=tokenizer.vocab_size,
            )
            LegacyBinarizer.binarize(
                data_path + '.txt', None, lambda t: ds.add_item(t), offset=0, end=offsets[1], already_numberized=True,
                append_eos=False
            )
            ds.finalize(data_path + '.idx')

    def split_dataser(self):
        id = [i for i in range(self.number_of_dataset)]
        np_data = np.array(id)
        np.random.shuffle(id)
        np_data = np_data[id]
        train, test = train_test_split(np_data, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.2, random_state=0)
        train = list(train)
        val = list(val)
        test = list(test)
        torch.save({'train': train, 'val': val, 'test': test}, './ref/split.pt')

class PseudoLabellingProcessor(DataProcessor):
    def __init__(self, filelist) -> None:
        ###filelist应该包括训练数据，预测标签，测试样本传入的顺序也是这样
        self.filelist=filelist
    def add_false_labels(self,write_path:str):
        with_pseudo=[]
        pred=DataFileManager.read_file(self.filelist[1])
        golden=DataFileManager.read_file(self.filelist[2])
        for p,g in zip(pred,golden):
            assert p["id"]==g["id"]
            with_pseudo.append({"id":p["id"],"text":g["text"],"labels":p["labels"]})
        DataFileManager.write_file(datalist=with_pseudo,write_path=write_path)
        

class NERDataProcessor(DataProcessor):
    ...






if __name__=="__main__":
    #binarizer=Binarizer()
    #binarizer.binarize("./train3.0.json")
    ...