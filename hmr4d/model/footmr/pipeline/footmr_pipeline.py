import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from hydra.utils import instantiate
from hmr4d.utils.net_utils import gaussian_smooth

from hmr4d.model.footmr.utils.endecoder import EnDecoder
from hmr4d.model.footmr.utils.postprocess import (
    pp_static_joint,
    process_ik,
    pp_static_joint_cam,
)
from hmr4d.model.footmr.utils import stats_compose

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    get_a_pred_cam,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)
from hmr4d.network.footmr.foot_transformer import FootEncoderRoPE
import hmr4d.utils.matrix as matrix


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights

        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        self.foot_motion_refiner = FootEncoderRoPE()

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer(
                "cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False
            )
            self.register_buffer(
                "cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False
            )

    # ========== Training ========== #

    def forward(self, inputs, train=False, postproc=False, static_cam=False):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample

        # *. Conditions
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        B, L, _ = cliff_cam.shape
        f_cam_angvel = inputs["cam_angvel"]
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        f_condition = {
            "obs": inputs["obs"],  # (B, L, J, 3)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_cam_angvel": f_cam_angvel,  # (B, L, C=6)
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
        }
        if train:
            f_condition = randomly_set_null_condition(f_condition, 0.1)

        foot_condition = {
            "obs": inputs["foot_obs"],  # (B, L, J, 3)
        }
        if train:
            # bounding box changes due to random rotation augmentation
            augm_cliff_cam = compute_bbox_info_bedlam(
                inputs["bbx_xys_for_foot"], inputs["K_fullimg"]
            )
            foot_condition["f_cliffcam"] = augm_cliff_cam
        else:
            foot_condition["f_cliffcam"] = cliff_cam.clone()

        # Forward & output
        model_output = self.denoiser3d(length=length, **f_condition)
        # model_output contains:
        # pred_x (bs, L, 151): 21*6=126 rotations (body_pose), 10 betas,
        #   global_orient, global_orient in GV, local transl velocity
        # pred_cam (bs, L, 3): standard [s, tx, ty] hmr cam
        # static_conf_logits (bs, L, 6): [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        decode_dict = self.endecoder.decode(model_output["pred_x"])  # (B, L, C) -> dict
        outputs.update({"model_output": model_output, "decode_dict": decode_dict})

        # Post-processing
        outputs["pred_smpl_params_incam"] = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "global_orient": decode_dict["global_orient"],  # (B, L, 3)
            "transl": compute_transl_full_cam(
                model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
            ),
        }
        # calculate global knee and ankle rotations for conditioning
        with torch.no_grad():
            if train:
                _, _, fk_mat = self.endecoder.fk_v2(
                    **inputs["smpl_params_c_foot"], get_intermediate=True
                )
                # transform pred global orient to augmented coordinate system:
                incam_global_orients = axis_angle_to_matrix(
                    outputs["pred_smpl_params_incam"]["global_orient"]
                )
                random_global_orient = inputs["DA_global_orient_matrix"]
                incam_global_orients = torch.matmul(
                    incam_global_orients, random_global_orient[:, None]
                )
                incam_global_orients = matrix_to_axis_angle(incam_global_orients)
                _, _, fk_mat_pred = self.endecoder.fk_v2(
                    body_pose=outputs["pred_smpl_params_incam"]["body_pose"],
                    betas=outputs["pred_smpl_params_incam"]["betas"],
                    transl=outputs["pred_smpl_params_incam"]["transl"],
                    global_orient=incam_global_orients,
                    get_intermediate=True,
                )
            else:
                _, _, fk_mat = self.endecoder.fk_v2(
                    **outputs["pred_smpl_params_incam"], get_intermediate=True
                )
                fk_mat_pred = fk_mat

        global_knee_rot = matrix.get_rotation(fk_mat)[:, :, 4:6]  # [B, L, 2, 3, 3]
        global_knee_r6d = matrix_to_rotation_6d(global_knee_rot).flatten(-2)  # [B, L, 12]
        global_ankle_rot = matrix.get_rotation(fk_mat_pred)[:, :, 7:9]  # [B, L, 2, 3, 3]
        global_ankle_r6d = matrix_to_rotation_6d(global_ankle_rot).flatten(-2)  # [B, L, 12]
        foot_condition["global_rot6d"] = torch.cat([global_knee_r6d, global_ankle_r6d], dim=-1)

        pred_global_ankle_r6d = self.foot_motion_refiner(
            length=length, **foot_condition
        )  # (B, L, 12)
        # fuse pred global ankle with body pose:
        global_ankle_rotmat = rotation_6d_to_matrix(pred_global_ankle_r6d.reshape(B, L, -1, 6))
        # transform from global to knee relative rotation
        rel_ankle_rotmat = torch.matmul(
            torch.transpose(global_knee_rot, -2, -1), global_ankle_rotmat
        )
        rel_ankle_aa = matrix_to_axis_angle(rel_ankle_rotmat).flatten(-2)
        decode_dict["body_pose"][:, :, 6 * 3 : 8 * 3] = rel_ankle_aa
        outputs["pred_smpl_params_incam"]["body_pose"][:, :, 6 * 3 : 8 * 3] = rel_ankle_aa

        if not train:
            pred_smpl_params_global = get_smpl_params_w_Rt_v2(  # This function has for-loop
                global_orient_gv=decode_dict["global_orient_gv"],
                local_transl_vel=decode_dict["local_transl_vel"],
                global_orient_c=decode_dict["global_orient"],
                cam_angvel=inputs["cam_angvel"],
            )
            outputs["pred_smpl_params_global"] = {
                "body_pose": decode_dict["body_pose"],
                "betas": decode_dict["betas"],
                **pred_smpl_params_global,
            }
            outputs["static_conf_logits"] = model_output["static_conf_logits"]
            if postproc:  # apply post-processing
                if static_cam:  # extra post-processing to utilize static camera prior
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(
                        outputs, self.endecoder
                    )
                else:
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                        outputs, self.endecoder
                    )
                body_pose = process_ik(outputs, self.endecoder)
                decode_dict["body_pose"] = body_pose
                outputs["pred_smpl_params_global"]["body_pose"] = body_pose
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]["valid"]  # (B, L)

        # 1. Simple loss: MSE
        pred_x = model_output["pred_x"]  # (B, L, C)
        target_x = self.endecoder.encode(inputs)  # (B, L, C)
        simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
        mask_simple = mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()  # (B, L, C)
        mask_simple[inputs["mask"]["spv_incam_only"], :, 142:] = False  # 3dpw training

        # loss on init and refined ankle prediction
        rel_ankle_r6d = matrix_to_rotation_6d(rel_ankle_rotmat).flatten(-2)  # (B, L, 12)
        norm_rel_ankle_r6d = self.endecoder.normalize_ankle_pose_r6d(rel_ankle_r6d)
        refined_x = pred_x.clone()
        refined_x[:, :, 6 * 6 : 8 * 6] = norm_rel_ankle_r6d
        simple_loss2 = F.mse_loss(refined_x, target_x, reduction="none")
        simple_loss = 0.5 * (simple_loss + simple_loss2)

        simple_loss = (simple_loss * mask_simple).mean()
        total_loss += simple_loss
        outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        extra_funcs = [
            compute_extra_incam_loss,
            compute_extra_global_loss,
        ]
        for extra_func in extra_funcs:
            extra_loss, extra_loss_dict = extra_func(inputs, outputs, self)
            total_loss += extra_loss
            outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss
        return outputs


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def compute_extra_incam_loss(inputs, outputs, ppl):
    model_output = outputs["model_output"]
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]  # effective length mask
    mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

    # Incam FK
    # prediction
    pred_c_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_incam"])
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)

    # gt
    gt_c_j3d = endecoder.fk_v2(**inputs["smpl_params_c"])  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    if weights.cr_j3d > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss = (cr_j3d_loss * mask[..., None, None]).mean()
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    if weights.transl_c > 0.0:
        # pred_transl = decode_dict["transl"]  # (B, L, 3)
        # gt_transl = inputs["smpl_params_c"]["transl"]
        # transl_c_loss = F.l1_loss(pred_transl, gt_transl, reduction="none")
        # transl_c_loss = (transl_c_loss * mask[..., None]).mean()

        # Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
        pred_cam = model_output["pred_cam"]  # (B, L, 3)
        gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
        # (compute_transl_full_cam(gt_pred_cam, inputs["bbx_xys"], inputs["K_fullimg"]) - gt_transl).abs().max()

        # Skip gts that are not good during random construction
        gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_j3d_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (inputs["bbx_xys"][..., 2] > 0)
        )[..., None]
        transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
        transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()

        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss += transl_c_loss * weights.transl_c

    if weights.j2d > 0.0:
        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(
            gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"]
        )  # (B, L, J, 2)

        valid_mask = (
            (gt_c_j3d[..., 2] > reproj_z_thr)
            * (pred_c_j3d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask).mean()

        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j = endecoder.smplx_model(**outputs["pred_smpl_params_incam"])
        root_ = pred_c_j[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        cr_vert_loss = (cr_vert_loss * mask[:, :, None, None]).mean()
        extra_loss += cr_vert_loss * weights.cr_verts
        extra_loss_dict["cr_vert_loss"] = cr_vert_loss

    if weights.verts2d > 0:
        gt_c_verts437 = inputs["gt_c_verts437"]  # (B, L, 437, 3)

        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr

        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_verts2d_01 = project_to_bi01(
            gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"]
        )  # (B, L, 437, 2)

        valid_mask = (
            (gt_c_verts437[..., 2] > reproj_z_thr)
            * (pred_c_verts437[..., 2] > reproj_z_thr)  # Be safe
            * (gt_verts2d_01[..., 0] > 0.0)
            * (gt_verts2d_01[..., 0] < 1.0)
            * (gt_verts2d_01[..., 1] > 0.0)
            * (gt_verts2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction="none")
        verts2d_loss = (verts2d_loss * mask[..., None, None] * valid_mask).mean()

        extra_loss += verts2d_loss * weights.verts2d
        extra_loss_dict["verts2d_loss"] = verts2d_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"].clone()  # (B, L)
    mask[inputs["mask"]["spv_incam_only"]] = False

    if weights.transl_w > 0:
        # compute pred_transl_w by rollout
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        gt_global_orient_w = inputs["smpl_params_w"]["global_orient"]
        local_transl_vel = decode_dict["local_transl_vel"]
        pred_transl_w = rollout_local_transl_vel(
            local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]]
        )

        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
        trans_w_loss = (trans_w_loss * mask[..., None]).mean()
        extra_loss += trans_w_loss * weights.transl_w
        extra_loss_dict["transl_w_loss"] = trans_w_loss

    # Static-Conf loss
    if weights.static_conf_bce > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]

        static_conf_loss = F.binary_cross_entropy_with_logits(
            pred_static_conf_logits, static_gt, reduction="none"
        )
        static_conf_loss = (static_conf_loss * mask[..., None]).mean()
        extra_loss += static_conf_loss * weights.static_conf_bce
        extra_loss_dict["static_conf_loss"] = static_conf_loss

    return extra_loss, extra_loss_dict


@autocast(enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[
        :, :, :, 2
    ]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt
