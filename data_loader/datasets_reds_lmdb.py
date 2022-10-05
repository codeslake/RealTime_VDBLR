'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import pickle

from data_loader.utils import *
import pyarrow as pa
from PIL import Image
import lmdb
import six
import os

class datasets(data.Dataset):
    def __init__(self, config, is_train):
        super(datasets, self).__init__()
        self.config = config
        self.is_train = is_train
        self.h = config.height
        self.w = config.width
        self.frame_num = config.frame_num
        self.frame_half = int(self.frame_num / 2)
        self.W = 1280
        self.H = 720
        self.C = 3

        if self.config.dist:
            self.rank = torch.distributed.get_rank()

        self.env_blur = None
        self.env_gt = None
        if is_train:
            self.frame_itr_num = config.frame_itr_num
            self.datapath_blur = os.path.join(config.data_path, config.input_path)
            self.datapath_gt = os.path.join(config.data_path, config.gt_path)
            with open(os.path.join(config.data_path, 'reds_info_train.pkl'), 'rb') as f:
                self.seqs_info = pickle.load(f)
        else:
            self.frame_itr_num = 1
            self.datapath_blur = os.path.join(config.VAL.data_path, config.VAL.input_path)
            self.datapath_gt = os.path.join(config.VAL.data_path, config.VAL.gt_path)
            #with open(os.path.join(config.data_path, 'reds_info_test.pkl'), 'rb') as f:
            with open(os.path.join(config.VAL.data_path, 'reds_info_valid.pkl'), 'rb') as f:
                self.seqs_info = pickle.load(f)

        self._init_idx()
        if self.is_train is False:
            idx_frame_acc = self.idx_frame.copy()
            length = 0
            for i in range(1, len(idx_frame_acc)):
                length = length + len(idx_frame_acc[i-1])
                temp = (np.array(idx_frame_acc[i]) + length).tolist()
                idx_frame_acc[i] = temp
            self.idx_frame_acc = idx_frame_acc

        self.len = int(np.ceil(len(self.idx_frame_flat)))

    def _init_idx(self):
        self.idx_video = []
        self.idx_frame_flat = []
        self.idx_frame = []
        for i in range(self.seqs_info['num']):
            total_frames = self.seqs_info[i]['length']

            if self.is_train:
                idx_frame_temp = list(range(0, total_frames - self.frame_itr_num + 1, self.frame_itr_num))
            else:
                idx_frame_temp = list(range(0, total_frames - self.frame_itr_num + 1))

            self.idx_frame_flat.append(idx_frame_temp)
            self.idx_frame.append(idx_frame_temp)

            for j in range(len(idx_frame_temp)):
                self.idx_video.append(i)

        self.idx_frame_flat = sum(self.idx_frame_flat, [])

    def _init_db(self):
        self.env_blur = lmdb.open(self.datapath_blur, map_size=1099511627776, readonly=True, lock=False, readahead=False, meminit=False)
        self.env_gt = lmdb.open(self.datapath_gt, map_size=1099511627776, readonly=True, lock=False, readahead=False, meminit=False)
        #self.env_gt = lmdb.open(self.datapath_gt,   map_size=1099511627776)
        self.txn_blur = self.env_blur.begin()
        self.txn_gt = self.env_gt.begin()

        # if self.is_train is False:
        #     keys = [ key for key, _ in self.txn_gt.cursor() ]
        #     for key in keys:
        #         print(key)

    def _read_imgbuf(self, seq_idx, frame_idx):
        key = '%03d_%08d' % (seq_idx, frame_idx)
        code = key.encode()
        blur_img = self.txn_blur.get(code)
        blur_img = np.frombuffer(blur_img, dtype='uint8')
        blur_img = blur_img.reshape(self.H, self.W, self.C)

        sharp_img = self.txn_gt.get(code)
        sharp_img = np.frombuffer(sharp_img, dtype='uint8')
        sharp_img = sharp_img.reshape(self.H, self.W, self.C)

        return blur_img, sharp_img

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.env_blur is None and self.env_gt is None:
            self._init_db()

        index = index# % self.len
        ##

        video_idx = self.idx_video[index]
        frame_offset = self.idx_frame_flat[index] - self.frame_half
        # input_file_path = self.input_file_path_list[video_idx]
        # gt_file_path = self.gt_file_path_list[video_idx]

        sampled_frame_idx = np.arange(frame_offset, frame_offset + self.frame_num + self.frame_itr_num - 1)
        if self.is_train:
            sampled_frame_idx = sampled_frame_idx.clip(min = self.idx_frame_flat[index], max = self.seqs_info[video_idx]['length'] - 1)
        else:
            sampled_frame_idx = sampled_frame_idx.clip(min = 0, max = self.seqs_info[video_idx]['length'] - 1)

        input_patches_temp = [None] * len(sampled_frame_idx)
        gt_patches_temp = [None] * len(sampled_frame_idx)

        flip_val = None
        rotate_val = None
        gauss = None
        if self.is_train:
            if random.uniform(0, 1) <= 0.5:
                ran = random.uniform(0, 1)
                if ran  <= 0.3:
                    rotate_val = cv2.ROTATE_90_COUNTERCLOCKWISE
                elif ran  <= 0.6:
                    rotate_val = cv2.ROTATE_90_CLOCKWISE
                else:
                    rotate_val = cv2.ROTATE_180

            if random.uniform(0, 1) <= 0.5:
                ran = random.uniform(0, 1)
                if ran <= 0.3:
                    flip_val = 0
                elif ran <= 0.6:
                    flip_val = 1
                else:
                    flip_val = -1

        for frame_idx in range(len(sampled_frame_idx)):
            sampled_idx = sampled_frame_idx[frame_idx]

            gauss = True if self.is_train and random.uniform(0, 1) <= 0.5 else None

            blur_img, sharp_img = self._read_imgbuf(video_idx, sampled_idx)

            input_img = augment(blur_img, rotate_val, flip_val, gauss)
            gt_img = augment(sharp_img, rotate_val, flip_val, gauss)

            input_patches_temp[frame_idx] = input_img
            gt_patches_temp[frame_idx] = gt_img

        input_patches_temp = np.concatenate(input_patches_temp[:len(sampled_frame_idx)], axis = 3)
        gt_patches_temp = np.concatenate(gt_patches_temp[:len(sampled_frame_idx)], axis = 3)

        cropped_frames = np.concatenate([input_patches_temp, gt_patches_temp], axis = 3)

        if self.is_train:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
        #else:
            #cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = False)

        input_patches = cropped_frames[:, :, :, 0:len(sampled_frame_idx) * 3]
        shape = input_patches.shape
        h = shape[1]
        w = shape[2]
        input_patches = input_patches.reshape((h, w, -1, 3))
        input_patches = torch.FloatTensor(np.transpose(input_patches, (2, 3, 0, 1)))

        gt_patches = cropped_frames[:, :, :, len(sampled_frame_idx) * 3:]
        gt_patches = gt_patches.reshape((h, w, -1, 3))
        gt_patches = torch.FloatTensor(np.transpose(gt_patches, (2, 3, 0, 1)))
        gt_patches = gt_patches[:, :, :, :]

        is_first = True
        if self.idx_video[index] == self.idx_video[index - 1]:
            is_first = False

        if self.is_train:
            return {'input': input_patches, 'gt': gt_patches, 'is_first': is_first}
        else:
            return {'input': input_patches, 'gt': gt_patches, 'is_first': is_first,
                    'video_len': self.seqs_info['num'],
                    'frame_len': self.seqs_info[video_idx]['length'],
                    'video_idx': video_idx,
                    'frame_idx': sampled_frame_idx[self.frame_half],
                    'video_name': '{0:04d}'.format(video_idx),
                    'frame_name': '{0:04d}'.format(sampled_frame_idx[self.frame_half]),
                   }

