import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, random_rotations
from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
)
from hmr4d.utils.geo.hmr_cam import (
    perspective_projection,
    normalize_kp2d,
    safely_render_x3d_K,
    get_bbx_xys,
)


class FootMRPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco23")

    def training_step(self, batch, batch_idx):
        B, L = batch["smpl_params_c"]["body_pose"].shape[:2]  # (bs, 120, 63)

        # Create augmented noisy-obs : gt_j3d standard: (bs, 120, 23, 3)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        # compute bounding box based on gt 2d verts
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)  # (bs, seqlen, 3)
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]

        # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]
        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:3])
        obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:3]).cuda()  # (B, L, J)
        j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = (
            False  # Set close-to-image-plane points as invisible
        )
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:3]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
        obs_kp2d = torch.cat(
            [obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        # obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        batch["obs"] = obs

        # perform foot data augmentation by sampling random global_orient
        random_global_orient = random_rotations(B, device=obs.device, dtype=obs.dtype)
        incam_global_orients = axis_angle_to_matrix(batch["smpl_params_c"]["global_orient"])
        batch["DA_global_orient_matrix"] = random_global_orient
        incam_global_orients = torch.matmul(incam_global_orients, random_global_orient[:, None])
        incam_global_orients = matrix_to_axis_angle(incam_global_orients)
        smpl_params_c_foot = dict()
        for k, v in batch["smpl_params_c"].items():
            smpl_params_c_foot[k] = v.clone()
        smpl_params_c_foot["global_orient"] = incam_global_orients
        batch["smpl_params_c_foot"] = smpl_params_c_foot
        with torch.no_grad():
            my_verts, my_j3d = self.smplx(**smpl_params_c_foot)
        # bbx_xys
        i_x2d = safely_render_x3d_K(my_verts, batch["K_fullimg"], thr=0.3)
        # compute bounding box based on gt 2d verts
        augm_bbx_xys = get_bbx_xys(i_x2d, do_augment=True)  # (bs, seqlen, 3)
        batch["bbx_xys_for_foot"] = augm_bbx_xys
        my_noisy_j3d = my_j3d + get_wham_aug_kp3d(my_j3d.shape[:3])
        my_obs_i_j2d = perspective_projection(my_noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
        my_j2d_visible_mask = get_visible_mask(my_j3d.shape[:3]).cuda()  # (B, L, J)
        my_j2d_visible_mask[my_noisy_j3d[..., 2] < 0.3] = (
            False  # Set close-to-image-plane points as invisible
        )
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(my_j3d.shape[:3]).cuda()  # (B, L, J)
            my_j2d_visible_mask[legs_invisible_mask] = False
        my_obs_kp2d = torch.cat(
            [my_obs_i_j2d, my_j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        my_obs = normalize_kp2d(my_obs_kp2d, batch["bbx_xys_for_foot"])  # (B, L, J, 3)
        batch["foot_obs"] = my_obs[:, :, 15:23]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True)

        # Log
        self.log(
            "train_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
            logger=True,
        )
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=B,
                    logger=True,
                )
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        kp2d = batch["kp2d"]
        # ROPE inference
        obs = normalize_kp2d(kp2d, batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"][0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "foot_obs": obs[:, :, 15:23].clone(),
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
        }
        outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc)
        if "pred_smpl_params_global" in outputs:
            outputs["pred_smpl_params_global"] = {
                k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
            }
            outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            }
        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(
                k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix
            )
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


footmr_pl = builds(
    FootMRPL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="footmr_pl", node=footmr_pl, group="model/footmr")
