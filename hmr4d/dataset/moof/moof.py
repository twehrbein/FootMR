import os
from pathlib import Path

import torch
from pytorch3d.transforms import quaternion_to_matrix
from torch.utils import data

from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.pylogger import Log


class MOOFEvalDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "MOOF"
        Log.info(f"[{self.dataset_name}] Full sequence")
        self.root = Path("inputs/MOOF/hmr4d_support/")

        self.labels = torch.load(self.root / "annotations.pt")
        self.preprocess_data = torch.load(self.root / "moof_preprocess.pt")

        self.idx2meta = list(self.labels)
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences.")

    def __len__(self):
        return len(self.idx2meta)

    def __getitem__(self, idx):
        data = {}
        vid = self.idx2meta[idx]
        label = self.labels[vid]
        processed_data = self.preprocess_data[vid]
        meta = {"dataset_id": self.dataset_name, "vid": label["video_name"]}
        data.update({"meta": meta})
        width_height = label["img_wh"]
        f_imgseq = processed_data["hmr2_features"]
        length = len(f_imgseq)
        data["length"] = length
        data["f_imgseq"] = f_imgseq
        data["width_height"] = width_height
        video_path = os.path.join(self.root, "videos", vid + ".mp4")
        data["video_path"] = video_path

        K_fullimg = estimate_K(*width_height)
        data["K_fullimg"] = K_fullimg[None].repeat(length, 1, 1)

        bbx_xyxy = label["bbx_xyxy"]
        # use tight bounding box for 2d evaluation with scale simply as max of height and width
        pck_bbox_center = (bbx_xyxy[:, 2:4] + bbx_xyxy[:, 0:2]) / 2.0
        pck_bbox_scale, _ = (bbx_xyxy[:, 2:4] - bbx_xyxy[:, 0:2]).max(dim=-1, keepdim=True)
        data["pck_bbox_center"] = pck_bbox_center
        data["pck_bbox_scale"] = pck_bbox_scale

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

        kp2d = processed_data["vitpose_kpts_17j"]  # (F, 17, 3)
        sapiens133 = processed_data["sapiens133"]
        gt_foot_kpts = label["gt_feet_kpts"].clone()
        kp2d_feet = sapiens133[:, 17:23]
        kp2d = torch.cat((kp2d, kp2d_feet), dim=1)

        if "slam_results" in label:
            traj = label["slam_results"]
            traj_quat = traj[:, [6, 3, 4, 5]]
            R_w2c = quaternion_to_matrix(traj_quat).mT
            cam_angvel = compute_cam_angvel(R_w2c)
        else:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
            cam_angvel = compute_cam_angvel(R_w2c)
        data.update({
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "cam_angvel": cam_angvel,
            "gt_foot_kpts": gt_foot_kpts,
            "bbx_xyxy": bbx_xyxy,
        })

        return data


MainStore.store(
    name="all",
    node=builds(
        MOOFEvalDataset,
    ),
    group="test_datasets/moof",
)
