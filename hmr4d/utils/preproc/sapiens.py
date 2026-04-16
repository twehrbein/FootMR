import glob
from pathlib import Path
from subprocess import call

import numpy as np
import torch
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from tqdm import tqdm


class SapiensPoseExtractor:
    def __init__(self, tqdm_leave=True):
        self.number_joints = 133  # use coco-wholebody version
        ckpt_path = "inputs/checkpoints/sapiens/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth"
        model_config = "hmr4d/configs/sapiens_2b-210e_coco_wholebody-1024x768.py"

        self.pose_estimator = init_pose_estimator(
            model_config,
            ckpt_path,
            override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
            device="cuda",
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=False))))

        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, outfolder, bbx_xyxy):
        # extract video to images
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        call([
            'ffmpeg',
            '-i', video_path, '-qscale:v',
            '2', f"{outfolder}/%06d.jpg"
        ])
        all_frames = glob.glob(outfolder + "*.jpg")
        all_frames.sort()
        assert len(all_frames) == len(bbx_xyxy)

        poses_2d = np.zeros((len(all_frames), 23, 3), dtype=np.float32)
        for i in tqdm(range(len(all_frames)), desc="Sapiens", leave=self.tqdm_leave):
            imgpath = all_frames[i]
            pose_results = inference_topdown(self.pose_estimator, imgpath, bbx_xyxy[[i]])
            data_samples = merge_data_samples(pose_results)
            results = data_samples.get("pred_instances", None)
            kpts = results["keypoints"][0][:23]  # select first 23 joints (17 coco + 6 feet)
            conf = results["keypoint_scores"][0][:23]
            pred_pose = np.concatenate((kpts, conf[:, None]), axis=1).astype(np.float32)
            poses_2d[i] = pred_pose

            # img = cv2.imread(imgpath)[:, :, ::-1]
            # plt.imshow(img)
            # plt.scatter(pred_pose[:, 0], pred_pose[:, 1])
            # plt.show()
            # plt.close()

        poses_2d = torch.from_numpy(poses_2d)
        return poses_2d
