import torch
import cv2
import numpy as np
from hmr4d.utils.wis3d_utils import get_colors_by_conf


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_bbx_xys_on_image(bbx_xys, image, conf=True):
    assert isinstance(bbx_xys, np.ndarray)
    assert isinstance(image, np.ndarray)
    image = image.copy()
    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, lu_point, rd_point, color, 2)
    return image


def draw_bbx_xys_on_image_batch(bbx_xys_batch, image_batch, conf=None):
    """conf: if provided, list of bool"""
    use_conf = conf is not None
    bbx_xys_batch = to_numpy(bbx_xys_batch)
    assert len(bbx_xys_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xys_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i], conf[i]))
        else:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i]))
    return image_batch_out


def draw_bbx_xyxy_on_image(bbx_xys, image, conf=True):
    bbx_xys = to_numpy(bbx_xys)
    image = to_numpy(image)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, (int(bbx_xys[0]), int(bbx_xys[1])), (int(bbx_xys[2]), int(bbx_xys[3])), color, 2)
    return image


def draw_bbx_xyxy_on_image_batch(bbx_xyxy_batch, image_batch, mask=None, conf=None):
    """
    Args:
        conf: if provided, list of bool, mutually exclusive with mask
        mask: whether to draw, historically used
    """
    if mask is not None:
        assert conf is None
    if conf is not None:
        assert mask is None
    use_conf = conf is not None
    bbx_xyxy_batch = to_numpy(bbx_xyxy_batch)
    image_batch = to_numpy(image_batch)
    assert len(bbx_xyxy_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xyxy_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i], conf[i]))
        else:
            if mask is None or mask[i]:
                image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i]))
            else:
                image_batch_out.append(image_batch[i])
    return image_batch_out


def draw_kpts(frame, keypoints, color=(0, 255, 0), thickness=2):
    frame_ = frame.copy()
    for x, y in keypoints:
        cv2.circle(frame_, (int(x), int(y)), thickness, color, -1)
    return frame_


def draw_kpts_with_conf(frame, kp2d, conf, thickness=2):
    """
    Args:
        kp2d: (J, 2),
        conf: (J,)
    """
    frame_ = frame.copy()
    conf = conf.reshape(-1)
    colors = get_colors_by_conf(conf)  # (J, 3)
    colors = colors[:, [2, 1, 0]].int().numpy().tolist()
    for j in range(kp2d.shape[0]):
        x, y = kp2d[j, :2]
        c = colors[j]
        cv2.circle(frame_, (int(x), int(y)), thickness, c, -1)
    return frame_


def draw_kpts_with_conf_batch(frames, kp2d_batch, conf_batch, thickness=2):
    """
    Args:
        kp2d_batch: (B, J, 2),
        conf_batch: (B, J)
    """
    assert len(frames) == len(kp2d_batch)
    assert len(frames) == len(conf_batch)
    frames_ = []
    for i in range(len(frames)):
        frames_.append(draw_kpts_with_conf(frames[i], kp2d_batch[i], conf_batch[i], thickness))
    return frames_


def draw_coco17_skeleton(img, keypoints, conf_thr=0, kpt_s=6, bone_s=4):
    use_conf_thr = True if keypoints.shape[1] == 3 else False
    img = img.copy()
    # fmt:off
    coco_skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]            
    # fmt:on
    left_bones = [[15, 13], [13, 11], [5, 11], [5, 7], [7, 9], [0, 1], [1, 3], [3, 5]]
    left_joints = [1, 3, 5, 7, 9, 11, 13, 15]
    for bone in coco_skel:
        bone_color = (255, 0, 0) if bone in left_bones else (0, 255, 0)
        if use_conf_thr:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            kp1_color = (255, 0, 0) if bone[0] in left_joints else (0, 255, 0)
            kp2_color = (255, 0, 0) if bone[1] in left_joints else (0, 255, 0)

            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), kpt_s, kp1_color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), kpt_s, kp2_color, -1)

        else:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
    return img


def draw_feet_skeleton(img, keypoints, conf_thr=0, kpt_s=6, bone_s=4):
    use_conf_thr = True if keypoints.shape[1] == 3 else False
    img = img.copy()
    # fmt:off
    coco_skel = [[0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]]
    # fmt:on
    left_bones = [[0, 2], [0, 3], [0, 4]]
    left_joints = [0, 2, 3, 4]
    for bone in coco_skel:
        bone_color = (255, 0, 0) if bone in left_bones else (0, 255, 0)
        if use_conf_thr:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            kp1_color = (255, 0, 0) if bone[0] in left_joints else (0, 255, 0)
            kp2_color = (255, 0, 0) if bone[1] in left_joints else (0, 255, 0)

            if kp1_c > conf_thr and kp2_c > conf_thr:
                pass
                # img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), kpt_s, kp1_color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), kpt_s, kp2_color, -1)

        else:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
    return img


def draw_coco16_skeleton(img, keypoints, conf_thr=0, kpt_s=6, bone_s=4):
    use_conf_thr = True if keypoints.shape[1] == 3 else False
    img = img.copy()
    # fmt:off
    coco_skel = [[0, 1], [1, 2], [2, 14], [14, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 15],
                 [15, 9], [9, 10], [10, 11], [15, 12], [12, 13], [14, 15]]
    # fmt:on
    left_bones = [[14, 3], [3, 4], [4, 5],
                  [15, 9], [9, 10], [10, 11]]
    left_joints = [3, 4, 5, 9, 10, 11]
    for bone in coco_skel:
        bone_color = (255, 0, 0) if bone in left_bones else (0, 255, 0)
        if use_conf_thr:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            kp1_color = (255, 0, 0) if bone[0] in left_joints else (0, 255, 0)
            kp2_color = (255, 0, 0) if bone[1] in left_joints else (0, 255, 0)

            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), kpt_s, kp1_color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), kpt_s, kp2_color, -1)

        else:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
    return img


def draw_coco18_skeleton(img, keypoints, conf_thr=0, kpt_s=6, bone_s=4):
    # openpose as used in 3dpw
    use_conf_thr = True if keypoints.shape[1] == 3 else False
    img = img.copy()
    # fmt:off
    coco_skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 17], [17, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]            
    # fmt:on
    left_bones = [[15, 13], [13, 11], [5, 11], [5, 7], [7, 9], [0, 1], [1, 3], [3, 5], [5, 17]]
    left_joints = [1, 3, 5, 7, 9, 11, 13, 15]
    for bone in coco_skel:
        bone_color = (255, 0, 0) if bone in left_bones else (0, 255, 0)
        if use_conf_thr:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            kp1_color = (255, 0, 0) if bone[0] in left_joints else (0, 255, 0)
            kp2_color = (255, 0, 0) if bone[1] in left_joints else (0, 255, 0)

            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), kpt_s, kp1_color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), kpt_s, kp2_color, -1)

        else:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
    return img


def draw_coco23_skeleton(img, keypoints, conf_thr=0, kpt_s=6, bone_s=4):
    use_conf_thr = True if keypoints.shape[1] == 3 else False
    img = img.copy()
    # fmt:off
    coco_skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
                 [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                 [3, 5], [4, 6], [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]]
    # fmt:on
    left_bones = [[15, 13], [13, 11], [5, 11], [5, 7], [7, 9], [0, 1], [1, 3], [3, 5],
                  [15, 17], [15, 18], [15, 19]]
    left_joints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]
    for bone in coco_skel:
        bone_color = (255, 0, 0) if bone in left_bones else (0, 255, 0)
        if use_conf_thr:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            kp1_color = (255, 0, 0) if bone[0] in left_joints else (0, 255, 0)
            kp2_color = (255, 0, 0) if bone[1] in left_joints else (0, 255, 0)

            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), kpt_s, kp1_color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), kpt_s, kp2_color, -1)

        else:
            kp1 = keypoints[bone[0]][:2].astype(int)
            kp2 = keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), bone_color, bone_s)
    return img


def draw_coco_skeleton_batch(imgs, keypoints_batch, number_joints, conf_thr=0):
    assert number_joints in [17, 23]
    assert len(imgs) == len(keypoints_batch)
    keypoints_batch = to_numpy(keypoints_batch)
    imgs_out = []
    for i in range(len(imgs)):
        if number_joints == 17:
            imgs_out.append(draw_coco17_skeleton(imgs[i], keypoints_batch[i], conf_thr))
        else:
            imgs_out.append(draw_coco23_skeleton(imgs[i], keypoints_batch[i], conf_thr))
    return imgs_out
