import torch
import pytorch_lightning as pl
from einops import einsum
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.configs import MainStore, builds
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.pylogger import Log
from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    as_np_array,
)
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.smplx_utils import make_smplx


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        self.dataset_name = "RICH"
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "n_mpjpef": {},  # scale-normalized MPJPE for foot keypoints
            "ajae": {},  # Ankle Joint Angle Error
        }

        # SMPLX
        self.smplx_model = {
            "male": make_smplx("rich-smplx", gender="male").cuda(),
            "female": make_smplx("rich-smplx", gender="female").cuda(),
            "neutral": make_smplx("rich-smplx", gender="neutral").cuda(),
        }

        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
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
        gender = batch["gender"][0]
        T_w2c = batch["T_w2c"][0]

        # Groundtruth (cam)
        target_w_params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
        target_w_output = self.smplx_model[gender](**target_w_params)
        target_w_verts = torch.stack(
            [torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices]
        )
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
        # compute gt incam global_orient:
        gt_global_orient = target_w_params["global_orient"]
        gt_global_rotmat = axis_angle_to_matrix(gt_global_orient)
        cam_rot = T_w2c[..., :3, :3]
        gt_incam_rotmat = torch.matmul(cam_rot[None], gt_global_rotmat)
        gt_incam_orient = matrix_to_axis_angle(gt_incam_rotmat)
        gt_incam_full_body_pose = torch.cat((gt_incam_orient, target_w_params["body_pose"]), dim=1)
        # + Prediction -> Metric
        # 1. cam
        pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        del smpl_out  # Prevent OOM
        pred_incam_full_body_pose = torch.cat(
            (pred_smpl_params_incam["global_orient"], pred_smpl_params_incam["body_pose"]), dim=1
        )

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
            "gt_incam_pose": gt_incam_full_body_pose,
            "pred_incam_pose": pred_incam_full_body_pose,
            "parents": self.smplx_model["neutral"].bm.parents,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, _ = trainer.local_rank, trainer.world_size
        monitor_metric = "ajae"

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
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {
            k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()
        }
        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.dataset_name}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        for k, v in metrics_avg.items():
            pl_module.log_dict({f"val_metric_{self.dataset_name}/{k}": v}, logger=True)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


rich_node = builds(MetricMocap)
MainStore.store(
    name="metric_rich", node=rich_node, group="callbacks", package="callbacks.metric_rich"
)
