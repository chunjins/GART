import os as os
import shutil

import numpy as np

META = {
    "313": {"begin_train_frame": 0, "end_train_frame": 800, "frame_interval_train": 1, "end_eval_frame": 1000, "frame_interval_eval": 30},
    "315": {"begin_train_frame": 0, "end_train_frame": 1600, "frame_interval_train": 1, "end_eval_frame": 2000, "frame_interval_eval": 30},
    "377": {"begin_train_frame": 0, "end_train_frame": 456, "frame_interval_train": 1, "end_eval_frame": 570, "frame_interval_eval": 30},
    "386": {"begin_train_frame": 0, "end_train_frame": 432, "frame_interval_train": 1, "end_eval_frame": 540, "frame_interval_eval": 30},
    "387": {"begin_train_frame": 0, "end_train_frame": 432, "frame_interval_train": 1, "end_eval_frame": 540, "frame_interval_eval": 30},
    "390": {"begin_train_frame": 0, "end_train_frame": 937, "frame_interval_train": 1, "end_eval_frame": 1171, "frame_interval_eval": 30},
    "392": {"begin_train_frame": 0, "end_train_frame": 445, "frame_interval_train": 1, "end_eval_frame": 556, "frame_interval_eval": 30},
    "393": {"begin_train_frame": 0, "end_train_frame": 527, "frame_interval_train": 1, "end_eval_frame": 658, "frame_interval_eval": 30},
    "394": {"begin_train_frame": 0, "end_train_frame": 380, "frame_interval_train": 1, "end_eval_frame": 475, "frame_interval_eval": 30},
}

class data_extractor():
    def __init__(self, seq_name):

        self.seq_name = seq_name
        self.root = f'/media/chunjins/My Passport/project/HumanNeRF/0_dataset/zju_mocap/CoreView_{seq_name}'

        anno_fn = os.path.join(self.root, "annots.npy")
        self.annots = np.load(anno_fn, allow_pickle=True).item()
        self.cams = self.annots["cams"]
        self.num_cams = len(self.cams["K"])

        self.training_view = [0]
        self.testing_view = [i for i in range(self.num_cams) if i not in self.training_view]

        self.root_save = f'/ubc/cs/home/c/chunjins/chunjin_scratch/project/humannerf/dataset/zju_processed/data_for_gart/{seq_name}'
        self.root_save_img = os.path.join(self.root_save, 'images')
        self.root_save_msk = os.path.join(self.root_save, 'mask')
        self.root_save_smpl = os.path.join(self.root_save, 'smpl_params')
        self.root_save_verts = os.path.join(self.root_save, 'vertices')

        os.makedirs(self.root_save_img, exist_ok=True)
        os.makedirs(self.root_save_msk, exist_ok=True)
        os.makedirs(self.root_save_smpl, exist_ok=True)
        os.makedirs(self.root_save_verts, exist_ok=True)

    def copy_imgs(self, views, i_begin, i_end, i_intv):
        for view in views:
            imgs = np.array(
                [
                    np.array(ims_data["ims"])[view]
                    for ims_data in self.annots["ims"][i_begin: i_end][::i_intv]
                ]
            ).ravel()

            for img in imgs:
                if self.seq_name in ['313', '315']:
                    frame_idx = int(img.split('_')[4])
                    img_save = f'{frame_idx-1:06d}.jpg'
                    npy_save = f'{frame_idx-1}.npy'
                else:
                    img_save = img.split('/')[-1]
                    frame_idx = int(img.split('/')[-1].split('.')[0])
                    npy_save = f'{frame_idx}.npy'

                path_save_img = os.path.join(self.root_save_img, f'{view:02d}')
                os.makedirs(path_save_img, exist_ok=True)
                path_save_img = os.path.join(self.root_save_img, f'{view:02d}', img_save)
                path_source_img = os.path.join(self.root, img)
                shutil.copy(path_source_img, path_save_img)

                path_save_msk = os.path.join(self.root_save_msk, f'{view:02d}')
                os.makedirs(path_save_msk, exist_ok=True)
                path_save_msk = os.path.join(self.root_save_msk, f'{view:02d}', img_save.replace('jpg', 'png'))
                path_source_msk = os.path.join(self.root, 'mask_cihp', img.replace('jpg', 'png'))
                if os.path.exists(path_source_msk):
                    shutil.copy(path_source_msk, path_save_msk)
                else:
                    path_source_msk = os.path.join(self.root, 'mask', img.replace('jpg', 'png'))
                    shutil.copy(path_source_msk, path_save_msk)

                path_save_smpl = os.path.join(self.root_save_smpl, npy_save)
                path_source_smpl = os.path.join(self.root, "new_params", f"{frame_idx}.npy")
                shutil.copy(path_source_smpl, path_save_smpl)

                path_save_verts = os.path.join(self.root_save_verts, npy_save)
                path_source_verts = os.path.join(self.root, "new_vertices", f"{frame_idx}.npy")
                shutil.copy(path_source_verts, path_save_verts)


    def process_data(self, types = 'training'):
        # for type in types:
        #     if type == 'training':
        #         # training
        #         i_begin = META[self.seq_name]["begin_train_frame"]
        #         i_intv = META[self.seq_name]["frame_interval_train"]
        #         i_end = META[self.seq_name]["end_train_frame"]
        #         self.copy_imgs(self.training_view, i_begin, i_end, i_intv)
        #     elif type == 'novel_view':
        #         # novel_view
        #         i_begin = META[self.seq_name]["begin_train_frame"]
        #         i_intv = META[self.seq_name]["frame_interval_eval"]
        #         i_end = META[self.seq_name]["end_train_frame"]
        #         self.copy_imgs(self.testing_view, i_begin, i_end, i_intv)
        #     else:
        #         # novel_pose
        #         i_begin = META[self.seq_name]["begin_train_frame"] + META[self.seq_name]["end_train_frame"]
        #         i_intv = META[self.seq_name]["frame_interval_eval"]
        #         i_end = META[self.seq_name]["end_eval_frame"]
        #         self.copy_imgs(self.testing_view, i_begin, i_end, i_intv)

        for idx_frame in range(len(self.annots['ims'])):
            for idx_view in range(self.num_cams):
                img = self.annots['ims'][idx_frame]['ims'][idx_view]

                if self.seq_name in ['313', '315']:
                    frame_idx = int(img.split('_')[4])
                    img_save = f'{frame_idx - 1:06d}.jpg'
                    img = f'{idx_view:02d}/{img_save}'
                else:
                    img_save = img.split('/')[-1]
                    img = f'{idx_view:02d}/{img_save}'

                self.annots['ims'][idx_frame]['ims'][idx_view] = img

        np.save(os.path.join(self.root_save, "annots.npy"), self.annots)


# subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']
subjects = ['377', '386', '387', '390', '392', '393']
subjects = ['313', '315']
for sub in subjects:
    extractor = data_extractor(sub)
    extractor.process_data(types=['training', 'novel_view', 'novel_pose'])