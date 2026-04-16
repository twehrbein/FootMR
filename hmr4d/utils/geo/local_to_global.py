# Copyright (c) Facebook, Inc. and its affiliates.
# https://github.com/facebookresearch/frankmocap

import torch
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix


def get_kinematic_map(smplx_parents, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur >= 0:
        parent = int(smplx_parents[cur])
        if cur != dst_idx:  # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map


def _batch_transfer_rot(body_pose_rotmat, part_rotmat, kinematic_map, transfer_type):
    if len(kinematic_map) == 0:
        return part_rotmat
    rotmat = body_pose_rotmat[:, 0]
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_rotmat[:, child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id

    if transfer_type == "g2l":
        part_rot_new = torch.matmul(torch.transpose(rotmat, 1, 2), part_rotmat)
    else:
        assert transfer_type == "l2g"
        part_rot_new = torch.matmul(rotmat, part_rotmat)

    return part_rot_new


def batch_transfer_rotation(
    smplx_parents, body_pose, part_rot, part_idx, transfer_type="g2l", result_format="rotmat"
):
    # body_pose (bs, 66) or (bs, 22, 3, 3); part_rot (bs, 3) or (bs, 3, 3)
    assert transfer_type in ["g2l", "l2g"]
    assert result_format in ["rotmat", "aa"]
    bs = body_pose.shape[0]

    if body_pose.dim() == 2:
        # aa
        assert body_pose.shape[1] == 66
        body_pose_rotmat = axis_angle_to_matrix(body_pose.view(bs, 22, 3))
    else:
        # rotmat
        assert body_pose.dim() == 4
        assert body_pose.shape[1] == 22
        assert body_pose.shape[2] == 3 and body_pose.shape[3] == 3
        body_pose_rotmat = body_pose

    if part_rot.dim() == 2:
        # aa
        assert part_rot.shape[1] == 3
        part_rotmat = axis_angle_to_matrix(part_rot.view(bs, 3))
    else:
        # rotmat
        assert part_rot.dim() == 3
        assert part_rot.size(1) == 3 and part_rot.size(2) == 3
        part_rotmat = part_rot

    kinematic_map = get_kinematic_map(smplx_parents, part_idx)
    part_rot_trans = _batch_transfer_rot(
        body_pose_rotmat, part_rotmat, kinematic_map, transfer_type
    )  # (bs, 3, 3)
    if result_format == "rotmat":
        return part_rot_trans
    else:
        return matrix_to_axis_angle(part_rot_trans)


def batch_transfer_multi_rotation(
    smplx_parents, body_pose, joint_indices, transfer_type="g2l", result_format="rotmat"
):
    bs = len(body_pose)
    if result_format == "rotmat":
        out_rotations = torch.zeros((bs, len(joint_indices), 3, 3), device=body_pose.device)
    else:
        assert result_format == "aa"
        out_rotations = torch.zeros((bs, len(joint_indices), 3), device=body_pose.device)
    for idx, joint_idx in enumerate(joint_indices):
        joint_global_rot = batch_transfer_rotation(
            smplx_parents,
            body_pose,  # inclusive global orient
            body_pose[:, joint_idx*3:(joint_idx+1)*3],
            part_idx=joint_idx,  # joint index of r_wrist
            transfer_type=transfer_type,  # local to global
            result_format=result_format,
        )
        out_rotations[:, idx] = joint_global_rot

    return out_rotations
