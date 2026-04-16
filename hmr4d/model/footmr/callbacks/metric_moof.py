import torch
import pytorch_lightning as pl
from pathlib import Path
from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import n_mpjpe
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.hmr_cam import perspective_projection


def pck_accuracy(pred, gt, mask, thr):
    """
    Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Args:
        pred (torch.Tensor[N, K, 2]): Predicted keypoint locations, already normalized.
        gt (torch.Tensor[N, K, 2]): Groundtruth keypoint locations, already normalized.
        mask (torch.Tensor[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (torch.Tensor[N, K]): Accuracy of each keypoint for each instance.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    # Calculate the Euclidean distances between predicted and groundtruth keypoints
    distances = torch.norm(pred - gt, dim=-1)
    # Only consider distances for visible keypoints
    visible_distances = distances * mask.float()
    # Determine if distances are within the threshold
    within_threshold = visible_distances < thr

    # Calculate accuracy for each keypoint for each instance
    acc = within_threshold.float()

    # Calculate number of valid keypoints
    valid_mask = mask.float()
    valid_keypoints = valid_mask.sum()

    # Calculate average accuracy
    avg_acc = (acc * valid_mask).sum() / valid_keypoints if valid_keypoints > 0 else 0

    return acc, avg_acc.item(), valid_keypoints.item()


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        self.dataset_name = "MOOF"
        # vid->result
        self.metric_aggregator = {
            "pckf005": {},  # Percentage of Correct foot Keypoints @ 0.05
            "nfke2d": {},  # Normalized 2D Foot Keypoint Error
        }
        self.num_valid_kpts_dict = {}
        self.num_valid_feet_dict = {}  # foot is valid if all kpts of the foot are valid

        # SMPLX and SMPL
        self.smplx = make_smplx("supermotion").cuda()
        self.J_regressor24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.dataset_name:
            return

        vid = batch["meta"][0]["vid"]

        # + Prediction -> Metric
        smpl_out = self.smplx(**outputs["pred_smpl_params_incam"])
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])

        # ############ compute 2d metrics:
        # select foot keypoints from SMPL mesh
        smpl_foot_indices = [3216, 3226, 3387, 6617, 6624, 6787]
        pred_foot_kpts = pred_c_verts[:, smpl_foot_indices]
        pred_feet_2d = perspective_projection(pred_foot_kpts, batch["K_fullimg"].squeeze())

        gt_feet = batch["gt_foot_kpts"].squeeze()
        gt_feet_2d = gt_feet[:, :, :2]
        feet_kpts_vis = gt_feet[:, :, 2]
        # normalize keypoints by bounding box:
        pck_bbox_center = batch["pck_bbox_center"][0].unsqueeze(1)
        pck_bbox_scale = batch["pck_bbox_scale"][0].unsqueeze(1)
        pred_feet_2d = (pred_feet_2d - pck_bbox_center) / pck_bbox_scale
        gt_feet_2d = (gt_feet_2d - pck_bbox_center) / pck_bbox_scale

        # only compute 2d metrics for visible keypoints
        kpts_mask = feet_kpts_vis == 1.0
        num_valid_kpts = kpts_mask.sum()

        # 1) compute PCK_F@0.05
        pck005, _, _ = pck_accuracy(pred_feet_2d, gt_feet_2d, kpts_mask, thr=0.05)
        avgpck_005 = 100 * (pck005 * kpts_mask).sum() / num_valid_kpts if num_valid_kpts > 0 else 0

        # 2) compute Normalized 2D Foot Keypoint Error
        # only compute if all joints of a foot are visible. otherwise cannot center the foot
        # center gt 2d foot:
        center_left_foot = gt_feet_2d[:, :3].mean(dim=1, keepdims=True)
        center_right_foot = gt_feet_2d[:, 3:].mean(dim=1, keepdims=True)
        c_target_feet = gt_feet_2d.clone()
        c_target_feet[:, :3] = gt_feet_2d[:, :3] - center_left_foot
        c_target_feet[:, 3:] = gt_feet_2d[:, 3:] - center_right_foot
        # center pred:
        center_left_foot = pred_feet_2d[:, :3].mean(dim=1, keepdims=True)
        center_right_foot = pred_feet_2d[:, 3:].mean(dim=1, keepdims=True)
        c_pred_feet = pred_feet_2d.clone()
        c_pred_feet[:, :3] = pred_feet_2d[:, :3] - center_left_foot
        c_pred_feet[:, 3:] = pred_feet_2d[:, 3:] - center_right_foot

        left_foot_mask = kpts_mask[:, :3].clone()
        left_foot_mask = left_foot_mask * torch.all(left_foot_mask, dim=1)[:, None]
        right_foot_mask = kpts_mask[:, 3:].clone()
        right_foot_mask = right_foot_mask * torch.all(right_foot_mask, dim=1)[:, None]
        foot_mask = torch.cat((left_foot_mask, right_foot_mask), dim=1)
        num_valid_feet = foot_mask.sum()
        # center and scale aligned per foot
        nmpjpe_left = n_mpjpe(c_pred_feet[:, :3], c_target_feet[:, :3], return_mean=False)
        nmpjpe_right = n_mpjpe(c_pred_feet[:, 3:], c_target_feet[:, 3:], return_mean=False)
        cs_eucl_dist = torch.cat((nmpjpe_left, nmpjpe_right), dim=1)
        avg_nfke2d = (
            100 * (cs_eucl_dist * foot_mask).sum() / num_valid_feet if num_valid_feet > 0 else 0
        )

        mean_metrics = {"pckf005": avgpck_005.item(), "nfke2d": avg_nfke2d.item()}
        self.num_valid_kpts_dict[vid] = num_valid_kpts.item()
        self.num_valid_feet_dict[vid] = num_valid_feet.item()
        for key, val in mean_metrics.items():
            self.metric_aggregator[key][vid] = val

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, _ = trainer.local_rank, trainer.world_size
        monitor_metric = "nfke2d"

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # print monitored metric per sequence
        mm_per_seq = {k: v for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:.2f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {}
        total_valid_kpts = sum(self.num_valid_kpts_dict.values())
        total_valid_feet = sum(self.num_valid_feet_dict.values())
        for metric, v in self.metric_aggregator.items():
            weighted_metrics = 0
            for vid, avg_seq_metric in v.items():
                if metric == "nfke2d":
                    num_valid_kpts = self.num_valid_feet_dict[vid]
                else:
                    num_valid_kpts = self.num_valid_kpts_dict[vid]
                weighted_metric = avg_seq_metric * num_valid_kpts
                weighted_metrics += weighted_metric
            if metric == "nfke2d":
                metrics_avg[metric] = weighted_metrics / total_valid_feet
            else:
                metrics_avg[metric] = weighted_metrics / total_valid_kpts

        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.dataset_name}:\n"
                + "\n".join(f"{k}: {v:.2f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        for k, v in metrics_avg.items():
            pl_module.log_dict({f"val_metric_{self.dataset_name}/{k}": v}, logger=True)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


MainStore.store(
    name="metric_moof",
    node=builds(MetricMocap),
    group="callbacks",
    package="callbacks.metric_moof",
)
