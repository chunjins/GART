# Use InsAV to process in the wild video, then load it
import h5py
from torch.utils.data import Dataset
import logging
import json
import os
import numpy as np
from os.path import join
import os.path as osp
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
import imageio
import cv2
from plyfile import PlyData
from tqdm import tqdm
from transforms3d.euler import euler2mat
import glob


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }


class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        data_root="../data/mvhuman",
        video_name="100846",
        split="train",
        image_zoom_ratio=1.0,
        start_end_skip=None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        self.image_zoom_ratio = image_zoom_ratio

        root = osp.join(data_root, video_name)

        if split == 'train':
            path_data = f'{root}_training.h5'
        else:
            path_data = f'{root}_{split}.h5'

        dataset = h5py.File(path_data, 'r')
        self.dataset = dataset
        self.dataset_path = path_data

        self.img_shape = dataset['img_shape'][1:]
        self.img_num = dataset['img_shape'][0]

        self.smpl_params = {}
        self.smpl_params['global_orient'] = dataset['global_orient'][:]
        self.smpl_params['transl'] = dataset['smpl_transl'][:]
        self.smpl_params['body_pose'] = dataset['smpl_pose'][:]
        self.smpl_params['betas'] = dataset['smpl_betas'][:]

        self.cameras = {}
        self.cameras['intrinsics'] = dataset['cameras_K'][:]
        self.cameras['extrinsics'] = dataset['cameras_E'][:]

        return

    def __len__(self):
        return self.img_num

    def init_dataset(self):

        if self.dataset is not None:
            return
        print('init dataset')

        self.dataset = h5py.File(self.dataset_path, 'r')

    def load_image(self, frame_idx):
        self.init_dataset()
        img = self.dataset['images'][frame_idx].reshape(self.img_shape).astype('float32') # [0 to 255]
        msk = self.dataset['masks'][frame_idx].reshape(self.img_shape[0], self.img_shape[1]).astype('float32') # [0 to 1]

        if self.image_zoom_ratio != 1.:
            img = cv2.resize(img, None,
                             fx=self.image_zoom_ratio,
                             fy=self.image_zoom_ratio,
                             interpolation=cv2.INTER_LANCZOS4)
            msk = cv2.resize(msk, None,
                                    fx=self.image_zoom_ratio,
                                    fy=self.image_zoom_ratio,
                                    interpolation=cv2.INTER_LINEAR)

        img = (img[..., :3] / 255).astype(np.float32)
        bg_color = np.ones_like(img).astype(np.float32)
        img = img * msk[..., None] + (1 - msk[..., None])

        return img, msk

    def __getitem__(self, idx):
        img, msk = self.load_image(idx)

        pose = self.smpl_params["body_pose"][idx].reshape((24, 3))
        pose[0,:] = self.smpl_params["global_orient"][idx]
        K = self.cameras["intrinsics"][idx].copy()
        K[:2] *= self.image_zoom_ratio
        E = self.cameras["extrinsics"][idx]

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk,
            "K": K,
            "R": E[:3, :3],
            "T": E[:3, 3],
            "smpl_beta": self.smpl_params["betas"],
            "smpl_pose": pose,
            "smpl_trans": self.smpl_params["transl"][idx],
            "T_cw": E,
            "idx": idx,
        }

        meta_info = {
            "video": self.video_name,
        }
        viz_id = f"video{self.video_name}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info


if __name__ == "__main__":
    dataset = Dataset(
        data_root="../../data/mvhuman", video_name="100846"
    )
    ret = dataset[0]
    print(1)
