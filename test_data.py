import os
import copy
import json
import numpy as np
import cv2
from PIL import Image
from misc import bezier2bbox
from collections import defaultdict
# from test_transform import transform, create_operators
from test_transform import transform, create_operators

from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
def build_dataloader(config, mode, device, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet',
        'LMDBDataSetSR', 'TextSpottingDataset'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    if mode == "Train":
        # Distribute data to multiple cards
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    else:
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        collate_fn=collate_fn)

    return data_loader

class TextSpottingDataset(Dataset):
    def __init__(self, config, mode, logger, seed=None, epoch=-1):
        super(TextSpottingDataset, self).__init__()
        self.global_config = config['Global']
        dataset_config = config[mode]['dataset']
        self.ratio = dataset_config['ratio_list']
        self.get_icdar_2015_dataset(dataset_config=dataset_config)
        self.ops = create_operators(dataset_config['transforms'])
        self.mode = mode
        self.dataset_config = config[mode]

        # loader_config = config[mode]['loader']
        # if loader_config['shuffle']: random.shuffle(self.ids)
        # batch_size = loader_config['batch_size_per_card']
        # data_dir = dataset_config['data_dir']
        # ratio_list = dataset_config['ratio_list']
        # self.epoch = epoch


    def get_icdar_2015_dataset(self, dataset_config):
        image_path = os.path.join(dataset_config['data_dir'],dataset_config['label_file_list'][0])
        label_file = os.path.join(dataset_config['data_dir'],dataset_config['label_file_list'][1])
        
        dataset = {}
        with open(label_file, 'r') as f:
            labels = json.load(f)
        assert type(labels)==dict, 'annotation file format {} not supported'.format(type(labels))
        anns, imgs = {}, {}
        imgToAnns = defaultdict(list)
        if 'annotations' in labels:
            for ann in labels['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
        if 'images' in labels:
            for img in labels['images']:
                imgs[img['id']] = img

        for key in imgToAnns:
            data = {}
            image_name = imgs[key]['file_name']
            data['image'] = os.path.join(image_path, image_name)
            data['label'] = imgToAnns[key]
            dataset[key] = data
        self.data_list = dataset
        self.ids = list(sorted(self.data_list.keys()))

#######################################################################
    # #中为实验部分 
    def __getitem__(self, index):
        idx = self.ids[index]
        data = self.data_list[idx]
        # img = cv2.imread(data['image'])
        # h, w, c = img.shape
        image = Image.open(data['image']).convert('RGB')
        image_w, image_h = image.size
        anno = data['label']
        anno = [ele for ele in anno if 'iscrowd' not in anno or ele['iscrowd'] == 0]
        
        target = {}
        target['image_id'] = idx
        target['area'] = np.array([ele['area'] for ele in anno])
        target['labels'] = np.array([ann['category_id'] for ann in anno])
        target['iscrowd'] = np.array([ann['iscrowd'] for ann in anno])
        image_size = np.array([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 
        target['recog'] = np.array([ann['rec'] for ann in anno])
        target['bezier_pts'] = np.array([ann['bezier_pts'] for ann in anno])
        bboxes = []
        for bezier_pt in target['bezier_pts']:
            bbox = bezier2bbox(bezier_pt)
            bboxes.append(bbox)
        target['bboxes'] = np.array(bboxes).reshape([-1, 4])

        data = {'image':image, 'target':target}
        
        # tmp_data = copy.deepcopy(data)
        # tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
        # # if self.mode == 'Eval': tmp_data['sequence'] = tmp_data['val_sequence']
        # return_data = {}
        # return_data['image'] = tmp_data['image']
        # return_data['sequence'] = tmp_data['sequence']
        # return_data['mask'] = tmp_data['mask']
        # if self.mode == 'Eval':
        #     return_data['target'] = tmp_data['target']
        #     return_data['sequence'] = tmp_data['val_sequence']
        # return tmp_data

        return_data = {}
        padding_shape = [3, 0, 0]
        image_list = []
        # target_list = []
        seq_list = []
        # 以第一份数据的shape为准
        tmp_data = copy.deepcopy(data)
        tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
        tmp_image_data_shape = tmp_data['image'].shape
        padding_shape[1] = max(padding_shape[1], tmp_image_data_shape[1])
        padding_shape[2] = max(padding_shape[2], tmp_image_data_shape[2])
        image_list.append(tmp_data['image'])
        seq_list.append(tmp_data['sequence'] if self.mode == 'Train' else tmp_data['val_sequence'])

        for batch_idx in range(self.dataset_config['loader']['num_batch'] - 1):
            tmp_data = copy.deepcopy(data)
            tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
            while tmp_data['sequence'].shape != seq_list[0].shape:
                tmp_data = copy.deepcopy(data)
                tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
            tmp_image_data_shape = tmp_data['image'].shape
            padding_shape[1] = max(padding_shape[1], tmp_image_data_shape[1])
            padding_shape[2] = max(padding_shape[2], tmp_image_data_shape[2])
            image_list.append(tmp_data['image'])
            # target_list.append(tmp_data['target'])
            seq_list.append(tmp_data['sequence'] if self.mode == 'Train' else tmp_data['val_sequence'])

        image_list, mask_list = self.get_mask(image_list, padding_shape)
        return_data['image'] = image_list
        return_data['sequence'] = seq_list
        return_data['mask'] = mask_list
        if self.mode == 'Eval':
            return_data['image'] = image_list[0]
            return_data['sequence'] = seq_list[0]
            return_data['mask'] = mask_list[0]
            return_data['target'] = tmp_data['target']

        return return_data

    def __len__(self):
        return int(len(self.ids) * self.ratio)

    def get_mask(self, image_list, padding_shape):
        return_image_list, mask_list = [], []
        for image in image_list:
            pad_img = np.zeros(padding_shape).astype("float32")
            mask = np.ones(pad_img.shape[1:])
            pad_img[: image.shape[0], :image.shape[1], :image.shape[2]] = image
            mask[:image.shape[1], :image.shape[2]] = 0
            return_image_list.append(pad_img)
            mask_list.append(mask)
        return return_image_list, mask_list
