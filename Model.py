import numpy as np
import os
import random

class Model(dict):


    def __init__(self, dataset, model_name, topk=10, model_root='static/models'):
        self.dataset = dataset

        def load_pred(split_name):
            preds = np.load(os.path.join(model_root, model_name, split_name+'.npy'))
            sorted_index = [np.argsort(p)[::-1][:topk] for p in preds]

            # create index
            gt = self.dataset[split_name]['gt']
            correct_index = [i for i in range(len(gt)) if sorted_index[i][0]==gt[i]]
            wrong_index = [i for i in range(len(gt)) if sorted_index[i][0]!=gt[i]]
            all_index = [i for i in range(len(gt))]
            
            result = []
            ans_list = self.dataset.ans_list
            for i, idxs in enumerate(sorted_index):
                result.append([(j==gt[i], ans_list[j], preds[i][j]) for j in idxs])
            
            return {'prediction':result,
                    'correct':correct_index,
                    'wrong':wrong_index,
                    'all':all_index,}

        self['train'] = load_pred('train')
        self['test'] = load_pred('test')


    def get(self, split_name='train', q_type='all', r_type='all', index=-1):
        prediction = self[split_name]['prediction']
        sample = self.dataset[split_name]['sample']
        _index = list(set(self.dataset[split_name][q_type]).intersection(
                   set(self[split_name][r_type])))
        if index == -1:
            index = random.randint(0, len(_index))
        index = _index[index%len(_index)]
        return prediction[index], sample[index]

    def acc(self, split_name='train', q_type='all'):
        q_index = self.dataset[split_name][q_type]
        c_index = self[split_name]['correct']
        c_cnt = len(set(q_index).intersection(set(c_index)))
        return 1.0*c_cnt / len(q_index)
