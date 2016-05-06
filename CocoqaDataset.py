import random

class CocoqaDataset(dict):


    def __init__(self, dataset_root='static/COCO-QA/', 
                 image_root='coco_images/'):
        # construct a dict mapping id to image path
        with open(dataset_root + 'done/img_list.txt') as f:
            ids = f.readlines()
        with open(dataset_root + 'done/imgs_path.txt') as f:
            paths = [image_root + p.strip() for p in f.readlines()]
        id2path = dict(zip(ids, paths))

        # function for loading specific split of the dataset
        def load_dataset(split_name):
            split_path = dataset_root + split_name + '/'

            with open(split_path + 'gt.txt') as f:
                gt = [int(g)-1 for g in f.readlines()]
            
            with open(split_path + 'questions.txt') as f:
                ques = [q.strip() + '?' for q in f.readlines()]
            
            with open(split_path + 'img_ids.txt') as f:
                imgs = [id2path[i] for i in f.readlines()]

            with open(split_path + 'answers.txt') as f:
                ans = [a.strip() for a in f.readlines()]

            with open(split_path + 'types.txt') as f:
                types = [int(t) for t in f.readlines()]

            object_index = [i for i in range(len(types)) if types[i] == 0]
            number_index = [i for i in range(len(types)) if types[i] == 1]
            color_index = [i for i in range(len(types)) if types[i] == 2]
            location_index = [i for i in range(len(types)) if types[i] == 3]
            all_index = range(len(types))

            ans_list = []
            if split_name == 'train':
                ans_set = set(ans)
                for a in ans:
                    if a in ans_set:
                        ans_list.append(a)
                        ans_set.remove(a)

            return {'sample':zip(imgs, ques, ans), 
                    'object':object_index, 
                    'number':number_index, 
                    'color':color_index, 
                    'location':location_index,
                    'all':all_index,
                    'gt':gt}, ans_list
        self['train'], self.ans_list = load_dataset('train')
        self['test'], _ = load_dataset('test')


    def get(self, split_name='train', q_type='all', index=-1):
        sample = self[split_name]['sample']
        _index = self[split_name][q_type]
        if index == -1:
            index = random.randint(0, len(_index))
        return sample[_index[index]]


    def size(self, split_name='train', q_type='all'):
        return len(self[split_name][q_type])
