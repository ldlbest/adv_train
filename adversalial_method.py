import torch

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {} # 用于保存模型扰动前的参数

    def attack(
        self, 
        epsilon=1., 
        emb_name='roberta.embeddings.word_embeddings.weight' # emb_name表示模型中embedding的参数名
    ):
        '''
        生成扰动和对抗样本
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数 
            if param.requires_grad and emb_name in name: # 只取word embedding层的参数
                self.backup[name] = param.data.clone() # 保存参数值
                norm = torch.norm(param.grad) # 对参数梯度进行二范式归一化
                if norm != 0 and not torch.isnan(norm): # 计算扰动，并在输入参数值上添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
        

    def restore(
        self, 
        emb_name='roberta.embeddings.word_embeddings.weight' # emb_name表示模型中embedding的参数名
    ):
        '''
        恢复添加扰动的参数
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name] # 重新加载保存的参数值
        self.backup = {}


class PGD():
    def __init__(self,model,k = 3,epsilon=0.1,alpha=0.03):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.k = k  # 每次对抗训练迭代步数
        self.epsilon = epsilon
        self.alpha = alpha
    def attack(self,emb_name = 'word_embeddings', is_first_attack = False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] =  param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad/norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
                    # loss
    
    def restore(self,emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self,param_name,param_data,epsilon):
        r = param_data - self.emb_backup[param_name]
        n_r = torch.norm(r).data
        if n_r > epsilon:
            print('Projected')
            r = epsilon * r / n_r
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler.dense' not in name:
                # pretrained_bert.pooler.dense.weight的梯度输出是None 这层是不算进去的
                self.grad_backup[name] = param.grad.clone()
                
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler.dense' not in name:
                param.grad = self.grad_backup[name]