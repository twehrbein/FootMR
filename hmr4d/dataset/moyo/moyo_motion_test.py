from pathlib import Path

import torch
from torch.utils import data

from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy, resize_K
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.pylogger import Log


class MoyoSmplFullSeqDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "MOYO"
        Log.info(f"[{self.dataset_name}] Full sequence")
        self.moyo_dir = Path("inputs/MOYO/hmr4d_support")
        self.labels = torch.load(self.moyo_dir / "moyo_smplx_test_labels.pt")
        # ['gender', 'vidname', 'img_wh', 'K_fullimg', 'T_w2c',
        # 'imgname', 'smplx_params', 'bbx_xyxy']
        # uses v_template instead of betas (moyo_test_gt_v_template.pt)
        self.imgfeats = torch.load(self.moyo_dir / "test_vit_imgfeats.pt")
        self.vid2kp2d = torch.load(self.moyo_dir / "test_vitpose17.pt")
        self.sapiens133 = torch.load(self.moyo_dir / "test_sapiens133.pt")

        # Setup dataset index
        self.idx2meta = list(self.labels)
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences.")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        # ['gender', 'vidname', 'img_wh', 'K_fullimg', 'T_w2c',
        # 'imgname', 'smplx_params', 'bbx_xyxy']
        data = {}
        vid = self.idx2meta[idx]
        meta = {"dataset_id": self.dataset_name, "vid": vid}
        data.update({"meta": meta})

        # Add useful data
        label = self.labels[vid]
        width_height = label["img_wh"]
        gender = label["gender"]
        vidname = label["vidname"]

        # K_fullimg = label["K_fullimg"]  # (3, 3)
        data.update({"gt_K": label["K_fullimg"]})
        # use approximated intrinsics:
        K_fullimg = estimate_K(label["img_wh"][0], label["img_wh"][1])
        T_w2c = label["T_w2c"]
        imgname = label["imgname"]
        smplx_params = label["smplx_params"]
        # use neutral hand pose for evaluation
        if "left_hand_pose" in smplx_params:
            del smplx_params["left_hand_pose"]
        if "right_hand_pose" in smplx_params:
            del smplx_params["right_hand_pose"]
        bbx_xyxy = label["bbx_xyxy"]
        length = len(bbx_xyxy)
        f_imgseq = self.imgfeats[vid]

        data.update({
            "length": length,  # F
            "smplx_params": smplx_params,  # world
            "gender": gender,  # str
            "T_w2c": label["T_w2c"],  # (4, 4)
            "img_wh": width_height,
            "num_seqs": len(self.idx2meta),
            "imgname": imgname,
        })
        data["K_fullimg"] = K_fullimg

        # Preprocessed:  bbx, kp2d, image as feature
        bboxpt = torch.from_numpy(bbx_xyxy)
        bbx_xys = get_bbx_xys_from_xyxy(bboxpt, base_enlarge=1.2).float()  # (F, 3)
        kp2d = self.vid2kp2d[vid][:, :17]  # (F, 17, 3)
        kp2d_feet = self.sapiens133[vid][:, 17:23]
        kp2d = torch.cat((kp2d, kp2d_feet), dim=1)

        R_w2c = T_w2c[:3, :3].repeat(length, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)  # (L, 6)
        data.update({"bbx_xys": bbx_xys, "kp2d": kp2d, "cam_angvel": cam_angvel})
        data["f_imgseq"] = f_imgseq  # (F, 1024)

        # to render a video
        video_path = self.moyo_dir / f"videos/{vidname}.mp4"
        frame_id = torch.arange(length)
        ds = 0.5
        K_render_gt = resize_K(data["gt_K"], ds)
        K_render = resize_K(K_fullimg, ds)
        bbx_xys_render = bbx_xys * ds
        kp2d_render = kp2d.clone()
        kp2d_render[..., :2] *= ds
        width, height = data["img_wh"] * ds
        data["meta_render"] = {
            "name": vid,
            "video_path": str(video_path),
            "ds": ds,
            "frame_id": frame_id,
            "K": K_render,
            "K_gt": K_render_gt,
            "bbx_xys": bbx_xys_render,
            "kp2d": kp2d_render,
            "width_height": (width, height),
        }

        return data

    def _process_data(self, data):
        length = data["length"]
        data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)
        data["gt_K_fullimg"] = data["gt_K"][None].repeat(length, 1, 1)
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


# MOYO
MainStore.store(
    name="all",
    node=builds(MoyoSmplFullSeqDataset, populate_full_signature=True),
    group="test_datasets/moyo",
)
