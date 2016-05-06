from CocoqaDataset import CocoqaDataset
from Model import Model

class ModelManager(object):


    def __init__(self, dataset, model_root='static/models', topk=10):
        self.models = {}
        self.topk = 10
        self.model_root = model_root
        self.dataset = dataset


    def __get_model(self, model_name):
        # if model not in cache, cache it
        if model_name not in self.models:
            self.models[model_name] = Model(self.dataset, model_name, 
                    model_root=self.model_root, topk=self.topk)
        return self.models[model_name]


    def get(self, model_name, split_name='train', q_type='all', 
            r_type='all', index=-1):
        model = self.__get_model(model_name)
        return model.get(split_name=split_name, q_type=q_type, 
                r_type=r_type, index=index)

    
    def acc(self, model_name, split_name='train', q_type='all'):
        model = self.__get_model(model_name)
        return model.acc(split_name=split_name, q_type=q_type)
