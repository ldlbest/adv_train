import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np    
import torch.nn as nn
import torch.nn.functional as F
from adversalial_method import FGM,PGD
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForSequenceClassification,EvalPrediction,get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ad_train import MultiLabelDataset
from adversalial_method import FGM,PGD
from subtask_1_2a import G,_h_fbeta_score,_h_recall_score,_h_precision_score
from sklearn_hierarchical_classification.metrics import h_fbeta_score, h_recall_score, h_precision_score, \
    fill_ancestors, multi_labeled
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,precision_score,recall_score
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def hierarchical_compute_metric(predictions, labels, threshold=0.25):
    #1000*20
    id2label=torch.load("./id2label.pt")

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions)).cpu()
    
    predictions = np.zeros(probs.shape)##1000*20
    predictions[np.where(probs >= threshold)] = 1
    predicted_labels=[]
    goleden_labels=[]
    for line in predictions:
        label=[id2label[idx] for idx, label in enumerate(line) if label == 1.0]
        predicted_labels.append(label)
    for lb in labels:
        goleden_labels.append([id2label[idx] for idx, label in enumerate(lb) if label == 1.0])

    # return as dictionary
    with multi_labeled(goleden_labels, predicted_labels, G) as (gold_, pred_, graph_):
        _h_precision_score(gold_, pred_,graph_), _h_recall_score(gold_, pred_,graph_), _h_fbeta_score(gold_, pred_,graph_)
    metrics = {'f1': _h_fbeta_score(gold_, pred_,graph_),
               'h_recall': _h_recall_score(gold_, pred_,graph_),
               'h_precision': _h_precision_score(gold_, pred_,graph_)}
    return metrics

class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0
    
    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total= self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1

class Runner():
    def __init__(self,model,optimizer,loss_fn) -> None:
        self.model =model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')

    @torch.no_grad()
    def evaluate(self,valid_loader):
        self.model.eval()
        all_out=[]
        all_label=[]
        for batch_id, data in tqdm(enumerate(valid_loader),total=len(valid_loader)):
            out = self.model(input_ids=data["input_ids"].cuda(),attention_mask=data["attention_mask"].cuda()).logits
            all_out.append(out.cpu())
            all_label.append(data["labels"].cpu())
            concated_out=torch.concat(all_out)
            concated_label=torch.concat(all_label)
        metcir=hierarchical_compute_metric(predictions=concated_out,labels=concated_label)
        tqdm.write(str(metcir))
        return metcir
        
    @torch.no_grad()
    def predict(self,test_loader):
        self.load_model()
        self.model.eval()
        correct = 0
        total = 0
        for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(test_loader):
            out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        score = correct/total
        # print(total)
        print(f'Score on test set:{score}')
        return score
    
    def save_model(self, save_path = './modelparams/bestmodel_parms.pth'):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path='./modelparams/bestmodel_parms.pth'):
        self.model.load_state_dict(torch.load(model_path))
        
        
class Runner_FGM(Runner):
    def __init__(self,model,optimizer,loss_fn,scheduler,fgm = None) -> None:
        super(Runner_FGM,self).__init__(model,optimizer,loss_fn)
        self.fgm = fgm
        self.scheduler=scheduler
        
    def train(self,train_loader,valid_loader,num_epoch=10):
        self.model.train()
        step = 0
        best_accuracy = 0
        for step,data in enumerate(tqdm(train_loader,total=len(train_loader))):
            self.model.train()
            labels=data["labels"].cuda()
            input_ids=data["input_ids"].cuda()
            attention_mask=data["attention_mask"].cuda()
            out = self.model(input_ids=input_ids,attention_mask=attention_mask).logits
            loss = self.loss_fn(out,labels)
            loss.backward()
            self.fgm.attack()
            out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask).logits
            loss_adv = self.loss_fn(out_adv,labels)
            loss_adv.backward()
            self.fgm.restore()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            step += 1
            if step%100 == 0:
                lr = self.scheduler.get_lr()
                #print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
                tqdm.write(f'[loss]:{loss_adv},[lr]:{lr}')
                metric=self.evaluate(valid_loader)
                save_path="./temp/with_train4.0/{}_{}_{}".format(str(metric["f1"])[0:6],str(metric["h_recall"])[0:6],str(metric['h_precision'])[0:6])
                self.model.save_pretrained(save_path)
                    
class Runner_PGD(Runner):
    def __init__(self,model,optimizer,loss_fn,pgd = None) -> None:
        super(Runner_PGD,self).__init__(model,optimizer,loss_fn)
        self.pgd = pgd
        
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        K = self.pgd.k
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.pgd.backup_grad()
                for t in range(K):
                    self.pgd.attack(is_first_attack=(t==0))
                    if t == K-1:
                        self.pgd.restore_grad()
                    else:
                        self.optimizer.zero_grad()
                        
                    out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                    loss_adv = self.loss_fn(out_adv,labels)
                    loss_adv.backward() # 前面就按公式正常迭代梯度，最后一次在最初梯度上累计一次
                    
                self.pgd.restore()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')

class Runner_FreeLB(Runner):
    def __init__(self,model,optimizer,loss_fn,freelb = None):
        super(Runner_FreeLB,self).__init__(model,optimizer,loss_fn)
        self.freelb = freelb
    
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        K = self.freelb.k
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.optimizer.zero_grad()
                
                for t in range(K):
                    self.freelb.backup_grad()
                    self.optimizer.zero_grad()
                    self.freelb.attack(is_first_attack=(t==0))        
                    out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                    loss_adv = self.loss_fn(out_adv,labels)
                    loss_adv.backward()
                    self.freelb.backup_r_grad()
                    self.freelb.upgrade_grad()
                    self.freelb.upgrade_r_at()
                    
                self.freelb.restore()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%40 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
        
                
if __name__=="__main__":
    setup_seed(42)
    tokenizer=AutoTokenizer.from_pretrained("/home/lidailin/roberta_alll_data_mlm")
    model=AutoModelForSequenceClassification.from_pretrained("/home/lidailin/roberta_alll_data_mlm",
                                                             problem_type="multi_label_classification",
                                                             num_labels=20,
                                                             id2label=torch.load("id2label.pt"),
                                                             label2id=torch.load("label2id.pt")).cuda()
    train_dataset=MultiLabelDataset("./train4.0.json",tokenizer=tokenizer)
    for i in range(9):
        train_dataset+=MultiLabelDataset("./train4.0.json",tokenizer=tokenizer)
    val_dataset=MultiLabelDataset("./test3.0.json",tokenizer=tokenizer)
    print(len(train_dataset))
    Scheduler=get_cosine_schedule_with_warmup(
        optimizer=AdamW(model.parameters(),lr=3e-5), num_warmup_steps=0.3*len(train_dataset)/16, num_training_steps=len(train_dataset)/16
    )
    runner=Runner_FGM(
        model=model,
        optimizer=AdamW(model.parameters(),lr=3e-5),
        scheduler=Scheduler,
        loss_fn=F.binary_cross_entropy_with_logits,
        fgm = FGM(model),
    )

    runner.train(train_loader=DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,num_workers=0),valid_loader=DataLoader(dataset=val_dataset,batch_size=16,shuffle=True,num_workers=0))
