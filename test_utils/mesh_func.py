import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.abspath(__file__)))

import torch
import numpy as np
from eval_utils_instant_avatar import Evaluator as EvalAvatar
from eval_utils_instant_nvr import Evaluator as EvalNVR
from eval_utils_instant_avatar_brightness import Evaluator as EvalAvatarBrightness

from typing import Union
from lib_render.gauspl_renderer import render_cam_pcl
import cv2, glob
import pandas as pd
from tqdm import tqdm
from lib_data.instant_avatar_people_snapshot import Dataset as InstantAvatarDataset
from lib_data.zju_mocap import Dataset as ZJUDataset, get_batch_sampler
from lib_data.instant_avatar_wild import Dataset as InstantAvatarWildDataset
from lib_data.dog_demo import Dataset as DogDemoDataset
import logging
import trimesh


def get_evaluator(mode, device):
    if mode == "avatar":
        evaluator = EvalAvatar()
    elif mode == "nvr":
        evaluator = EvalNVR()
    elif mode == "avatar_brightness":
        evaluator = EvalAvatarBrightness()
    else:
        raise NotImplementedError()
    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


class TrainingSeqWrapper:
    def __init__(self, seq) -> None:
        self.seq = seq

    def __len__(self):
        return self.seq.total_t

    def __getitem__(self, idx):
        data = {}
        data["rgb"] = self.seq.rgb_list[idx]
        data["mask"] = self.seq.mask_list[idx]
        data["K"] = self.seq.K_list[idx]
        data["smpl_pose"] = torch.cat(
            [self.seq.pose_base_list[idx], self.seq.pose_rest_list[idx]], dim=0
        )
        data["smpl_trans"] = self.seq.global_trans_list[idx]
        return data, {}


def test_mesh(
        solver,
        seq_name: str,
        tto_flag=True,
        tto_step=300,
        tto_decay=60,
        tto_decay_factor=0.5,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        dataset_mode="people_snapshot",
        training_optimized_seq=None,
):
    device = solver.device
    model = solver.load_saved_model()

    assert dataset_mode in [
        "people_snapshot",
        "zju",
        "instant_avatar_wild",
        "dog_demo",
    ], f"Unknown dataset mode {dataset_mode}"

    if dataset_mode == "people_snapshot":
        eval_mode = "avatar"
        bg = [1.0, 1.0, 1.0]
        test_dataset = InstantAvatarDataset(
            noisy_flag=False,
            data_root="./data/people_snapshot/",
            video_name=seq_name,
            split="test",
            image_zoom_ratio=0.5,
        )
    elif dataset_mode == "zju":
        eval_mode = "nvr"
        test_dataset = ZJUDataset(
            data_root="../data/zju_mocap",
            video_name=seq_name,
            test_novel_pose=False,
            split="geometry",
            image_zoom_ratio=0.5,
        )
        bg = [0.0, 0.0, 0.0]  # zju use black background
    elif dataset_mode == "instant_avatar_wild":
        eval_mode = "avatar"
        test_dataset = InstantAvatarWildDataset(
            data_root="./data/insav_wild",
            video_name=seq_name,
            split="geometry",
            image_zoom_ratio=1.0,
            # ! warning, here follow the `ubc_hard.yaml` in InstAVT setting, use slicing
            start_end_skip=[2, 1000000000, 4],
        )
        bg = [1.0, 1.0, 1.0]

        test_len = len(test_dataset)
        assert (training_optimized_seq.total_t == test_len) or (
                training_optimized_seq.total_t == 1 + test_len
        ), "Now UBC can only support the same length of training and testing or + 1"
        test_dataset.smpl_params["body_pose"] = (
            training_optimized_seq.pose_rest_list.reshape(-1, 69)[:test_len]
                .detach()
                .cpu()
                .numpy()
        )
        test_dataset.smpl_params["global_orient"] = (
            training_optimized_seq.pose_base_list.reshape(-1, 3)[:test_len]
                .detach()
                .cpu()
                .numpy()
        )
        test_dataset.smpl_params["transl"] = (
            training_optimized_seq.global_trans_list.reshape(-1, 3)[:test_len]
                .detach()
                .cpu()
                .numpy()
        )
    elif dataset_mode == "dog_demo":
        eval_mode = "avatar_brightness"
        bg = [1.0, 1.0, 1.0]
        test_dataset = DogDemoDataset(
            data_root="./data/dog_data_official/", video_name=seq_name, test=True
        )
    else:
        raise NotImplementedError()

    evaluator = get_evaluator(eval_mode, device)

    _save_eval_maps(
        solver.log_dir,
        "test",
        model,
        solver,
        test_dataset,
        dataset_mode=dataset_mode,
        device=device,
        bg=bg,
        tto_flag=tto_flag,
        tto_step=tto_step,
        tto_decay=tto_decay,
        tto_decay_factor=tto_decay_factor,
        tto_evaluator=evaluator,
        pose_base_lr=pose_base_lr,
        pose_rest_lr=pose_rest_lr,
        trans_lr=trans_lr,
    )

    return


def _save_eval_maps(
        log_dir,
        save_name,
        model,
        solver,
        test_dataset,
        dataset_mode="people_snapshot",
        bg=[1.0, 1.0, 1.0],
        # tto
        tto_flag=False,
        tto_step=300,
        tto_decay=60,
        tto_decay_factor=0.5,
        tto_evaluator=None,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        device=torch.device("cuda:0"),
):
    model.eval()

    if tto_flag:
        test_save_dir_tto = osp.join(log_dir, f"{save_name}_tto")
        os.makedirs(test_save_dir_tto, exist_ok=True)
    else:
        test_save_dir = osp.join(log_dir, save_name)
        os.makedirs(test_save_dir, exist_ok=True)

    if dataset_mode == "zju":
        # ! follow instant-nvr evaluation
        iter_test_dataset = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=get_batch_sampler(test_dataset, frame_sampler_interval=1),
            num_workers=0,
        )
    else:
        iter_test_dataset = test_dataset

    logging.info(
        f"Saving images [TTO={tto_flag}] [N={len(iter_test_dataset)}]..."
    )

    for batch_idx, batch in tqdm(enumerate(iter_test_dataset)):
        # get data
        datas, metas = batch

        rgbmaps = []
        depthmaps = []
        full_proj_transforms = []
        c2ws = []
        for data, meta in zip(datas, metas):
            if dataset_mode == "zju":
                for k in data.keys():
                    data[k] = data[k].squeeze(0)

            rgb_gt = torch.as_tensor(data["rgb"])[None].float().to(device)
            mask_gt = torch.as_tensor(data["mask"])[None].float().to(device)
            H, W = rgb_gt.shape[1:3]
            K = torch.as_tensor(data["K"]).float().to(device)
            pose = torch.as_tensor(data["smpl_pose"]).float().to(device)[None]
            trans = torch.as_tensor(data["smpl_trans"]).float().to(device)[None]

            if dataset_mode == "zju":
                fn = f"frame_{int(meta['frame_idx']):06d}.ply"
            else:
                fn = f"{batch_idx}.png"

            if tto_flag:
                # change the pose from the dataset to fit the test view
                pose_b, pose_r = pose[:, :1], pose[:, 1:]
                model.eval()
                # * for delta list
                try:
                    list_flag = model.add_bones.mode in ["delta-list"]
                except:
                    list_flag = False
                if list_flag:
                    As = model.add_bones(t=batch_idx)  # B,K,4,4, the nearest pose
                else:
                    As = None  # place holder
                new_pose_b, new_pose_r, new_trans, As = solver.testtime_pose_optimization(
                    data_pack=[
                        rgb_gt,
                        mask_gt,
                        K[None],
                        pose_b,
                        pose_r,
                        trans,
                        None,
                    ],
                    model=model,
                    evaluator=tto_evaluator,
                    pose_base_lr=pose_base_lr,
                    pose_rest_lr=pose_rest_lr,
                    trans_lr=trans_lr,
                    steps=tto_step,
                    decay_steps=tto_decay,
                    decay_factor=tto_decay_factor,
                    As=As,
                )
                pose = torch.cat([new_pose_b, new_pose_r], dim=1).detach()
                trans = new_trans.detach()

                save_fn = osp.join(test_save_dir_tto, fn)
                render_pkg, mu = _save_render_image_from_pose(
                    model,
                    pose,
                    trans,
                    H,
                    W,
                    K,
                    bg,
                    rgb_gt,
                    save_fn,
                    time_index=batch_idx,
                    As=As,
                )
            else:
                save_fn = osp.join(test_save_dir, fn)
                render_pkg, mu = _save_render_image_from_pose(
                    model, pose, trans, H, W, K, bg, rgb_gt, save_fn, time_index=batch_idx
                )

            full_proj_transforms.append(render_pkg['full_proj_transform'])
            c2ws.append(data['T_cw'].cpu().numpy())

            rgbmaps.append(render_pkg['rgb'].cpu())
            depthmaps.append(render_pkg['dep'].cpu())

            gaussians_xyz = mu[0]

        rgbmaps = torch.stack(rgbmaps, dim=0)
        depthmaps = torch.stack(depthmaps, dim=0)
        c2ws = np.array(c2ws)
        center, radius =estimate_bounding_sphere(c2ws)
        mesh = extract_mesh_unbounded(gaussians_xyz, rgbmaps, depthmaps, center, radius, full_proj_transforms, torch.from_numpy(c2ws).float().cuda(), resolution=512)

        mesh.export(save_fn)
    return


def estimate_bounding_sphere(c2ws):
    """
    Estimate the bounding sphere given camera pose
    """
    from mesh_utils.render_utils import focus_point_fn
    torch.cuda.empty_cache()
    # c2ws = np.array(c2ws)
    # c2ws = np.array(
    #     [np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
    poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
    center = torch.from_numpy(center).float().cuda()
    print(f"The estimated bounding radius is {radius:.2f}")
    print(f"Use at least {2.0 * radius:.2f} for depth_trunc")
    return center, radius


@torch.no_grad()
def extract_mesh_unbounded(gaussians_xyz, rgbmaps, depthmaps, center, radius, full_proj_transforms, c2ws, resolution=512):
    """
    Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
    return o3d.mesh
    """

    def contract(x):
        mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
        return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

    def uncontract(y):
        mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
        return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

    def compute_sdf_perframe(i, points, depthmap, rgbmap, full_proj_transform):
        """
            compute per frame sdf
        """
        new_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ full_proj_transform
        z = new_points[..., -1:]
        pix_coords = (new_points[..., :2] / new_points[..., -1:])
        mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
        sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear',
                                                        padding_mode='border', align_corners=True).reshape(-1, 1)
        sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear',
                                                      padding_mode='border', align_corners=True).reshape(3, -1).T
        sdf = (sampled_depth - z)
        return sdf, sampled_rgb, mask_proj

    def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
        """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
        """
        if inv_contraction is not None:
            mask = torch.linalg.norm(samples, dim=-1) > 1
            # adaptive sdf_truncation
            sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
            sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            samples = inv_contraction(samples)
        else:
            sdf_trunc = 5 * voxel_size

        tsdfs = torch.ones_like(samples[:, 0]) * 1
        rgbs = torch.zeros((samples.shape[0], 3)).cuda()

        weights = torch.ones_like(samples[:, 0])
        for i, full_proj_transform in tqdm(enumerate(full_proj_transforms), desc="TSDF integration progress"):
            # samples_i = torch.dot(samples, c2ws[i].T)
            samples_i = samples @ c2ws[i, :3, :3].T + c2ws[i, :3, 3]
            sdf, rgb, mask_proj = compute_sdf_perframe(i, samples_i,
                                                       depthmap=depthmaps[i],
                                                       rgbmap=rgbmaps[i],
                                                       full_proj_transform=full_proj_transform,
                                                       )

            # volume integration
            sdf = sdf.flatten()
            mask_proj = mask_proj & (sdf > -sdf_trunc)
            sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
            w = weights[mask_proj]
            wp = w + 1
            tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
            rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
            # update weight
            weights[mask_proj] = wp

        if return_rgb:
            return tsdfs, rgbs

        return tsdfs

    normalize = lambda x: (x - center) / radius
    unnormalize = lambda x: (x * radius) + center
    inv_contraction = lambda x: unnormalize(uncontract(x))

    N = resolution
    voxel_size = (radius * 2 / N)
    print(f"Computing sdf gird resolution {N} x {N} x {N}")
    print(f"Define the voxel_size as {voxel_size}")
    sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
    from mesh_utils.mcube_utils import marching_cubes_with_contraction

    gaussians_xyz_w = gaussians_xyz @ c2ws[-1, :3, :3].T + c2ws[-1, :3, 3]

    R = contract(normalize(gaussians_xyz_w)).norm(dim=-1).cpu().numpy()
    R = np.quantile(R, q=0.95)
    R = min(R + 0.01, 1.9)

    mesh = marching_cubes_with_contraction(
        sdf=sdf_function,
        bounding_box_min=(-R, -R, -R),
        bounding_box_max=(R, R, R),
        level=0,
        resolution=N,
        inv_contraction=inv_contraction,
    )

    # coloring the mesh
    torch.cuda.empty_cache()
    # mesh = mesh.as_open3d
    # print("texturing mesh ... ")
    # _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    return mesh


@torch.no_grad()
def _save_render_image_from_pose(
        model, pose, trans, H, W, K, bg, rgb_gt, save_fn, time_index=None, As=None
):
    act_sph_order = model.max_sph_order
    device = pose.device
    # TODO: handle novel time!, not does not pass in means t=None; Can always use TTO to directly find As!
    additional_dict = {"t": time_index}
    if As is not None:
        additional_dict["As"] = As
    mu, fr, sc, op, sph, _ = model(
        pose, trans, additional_dict=additional_dict, active_sph_order=act_sph_order
    )  # TODO: directly input optimized As!
    render_pkg = render_cam_pcl(
        mu[0], fr[0], sc[0], op[0], sph[0], H, W, K, False, act_sph_order, bg
    )
    # mask = (render_pkg["alpha"].squeeze(0) > 0.0).bool()
    # render_pkg["rgb"][:, ~mask] = bg[0]  # either 0.0 or 1.0
    # pred_rgb = render_pkg["rgb"]  # 3,H,W
    # pred_rgb = pred_rgb.permute(1, 2, 0)[None]  # 1,H,W,3
    #
    # errmap = (pred_rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
    # errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # errmap = torch.from_numpy(errmap).to(device)[None] / 255
    # img = torch.cat(
    #     [rgb_gt[..., [2, 1, 0]], pred_rgb[..., [2, 1, 0]], errmap], dim=2
    # )  # ! note, here already swapped the channel order
    # cv2.imwrite(save_fn, img.cpu().numpy()[0] * 255)

    return render_pkg, mu


if __name__ == "__main__":
    # debug for brightness eval
    evaluator = EvalAvatarBrightness()
    evaluator.to(torch.device("cuda:0"))
    evaluator.eval()

