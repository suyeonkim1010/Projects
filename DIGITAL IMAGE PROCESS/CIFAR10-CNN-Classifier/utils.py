import random
import os
import math

import cv2

import numpy as np


class Metrics:

    def __init__(self, name):
        self.name = name
        self.epoch = -1
        self.loss = -1
        self.acc = -1
        self.max_acc = 0
        self.min_loss = np.inf
        self.max_acc_epoch = 0
        self.min_loss_epoch = 0

    def update(self, epoch, save_criteria):
        save_ckpt = 0
        self.epoch = epoch
        if self.acc > self.max_acc:
            self.max_acc = self.acc
            self.max_acc_epoch = epoch
            if f'{self.name}-acc' in save_criteria:
                save_ckpt = 1

        if self.loss < self.min_loss:
            self.min_loss = self.loss
            self.min_loss_epoch = epoch
            if f'{self.name}-loss' in save_criteria:
                save_ckpt = 1
        return save_ckpt

    def from_dict(self, dict_):
        self.acc = self.max_acc = dict_[f'{self.name}_acc']
        self.loss = self.min_loss = dict_[f'{self.name}_loss']
        self.epoch = self.min_loss_epoch = self.max_acc_epoch = dict_['epoch']

    def to_str(self, prefix='\t', line_sep='\n', epoch=False):
        str_ = (f'{prefix}{self.name}_loss: {self.loss:.5f}{line_sep}'
                f'{prefix}{self.name}_acc: {self.acc:.3f}{line_sep}')
        if epoch:
            str_ = f'{prefix}epoch: {self.epoch}{line_sep}{str_}'
        return str_

    def to_dict(self, dict_):
        dict_.update({
            f'{self.name}_loss': self.loss,
            f'{self.name}_acc': self.acc,
        })

    def to_writer(self, writer):
        writer.add_scalar(f'{self.name}/loss', self.loss, self.epoch)
        writer.add_scalar(f'{self.name}/acc', self.acc, self.epoch)

    def to_df(self, status_df):
        status_df.loc[self.name, 'loss'] = self.loss
        status_df.loc[self.name, 'acc'] = self.acc
        status_df.loc[self.name, 'min_loss (epoch)'] = (f'{self.min_loss:.3f} '
                                                        f'({self.min_loss_epoch:d})')
        status_df.loc[self.name, 'max_acc (epoch)'] = (f'{self.max_acc:.3f} '
                                                       f'({self.max_acc_epoch:d})')


class CVText:
    def __init__(self, color='white', bkg_color='black', location=0, font=3,
                 size=0.5, thickness=1, line_type=2, offset=(5, 25)):
        self.color = color
        self.bkg_color = bkg_color
        self.location = location
        self.font = font
        self.size = size
        self.thickness = thickness
        self.line_type = line_type
        self.offset = offset

        self.help = {
            'font': 'Available fonts: '
                    '0: cv2.FONT_HERSHEY_SIMPLEX, '
                    '1: cv2.FONT_HERSHEY_PLAIN, '
                    '2: cv2.FONT_HERSHEY_DUPLEX, '
                    '3: cv2.FONT_HERSHEY_COMPLEX, '
                    '4: cv2.FONT_HERSHEY_TRIPLEX, '
                    '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                    '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                    '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; ',
            'location': '0: top left, 1: top right, 2: bottom right, 3: bottom left; ',
            'bkg_color': 'should be empty for no background',
        }


class CVConstants:
    similarity_types = {
        0: cv2.TM_CCOEFF_NORMED,
        1: cv2.TM_SQDIFF_NORMED,
        2: cv2.TM_CCORR_NORMED,
        3: cv2.TM_CCOEFF,
        4: cv2.TM_SQDIFF,
        5: cv2.TM_CCORR
    }
    interp_types = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_AREA,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    fonts = {
        0: cv2.FONT_HERSHEY_SIMPLEX,
        1: cv2.FONT_HERSHEY_PLAIN,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_COMPLEX,
        4: cv2.FONT_HERSHEY_TRIPLEX,
        5: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    line_types = {
        0: cv2.LINE_4,
        1: cv2.LINE_8,
        2: cv2.LINE_AA,
    }


# BGR values for different colors
col_bgr = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


def compute_marks(cls_acc, cls_speed, det_acc, det_iou, det_speed):
    min_marks = 50
    max_marks = 100

    min_det_speed = 2
    min_det_acc = 50
    max_det_acc = 80
    min_det_iou = 50
    max_det_iou = 80

    min_cls_speed = 200
    min_cls_acc = 80
    max_cls_acc = 95

    cls_wt = 0.7
    det_wt = 1.0 - cls_wt

    if cls_speed < min_cls_speed or cls_acc < min_cls_acc:
        cls_marks = 0
    elif cls_acc >= max_cls_acc:
        cls_marks = 100
    else:
        cls_marks = min_marks + (max_marks - min_marks) * (cls_acc - min_cls_acc) / (max_cls_acc - min_cls_acc)

    print(f'cls_marks: {cls_marks:.2f}%')

    if det_speed < min_det_speed or det_acc < min_det_acc or det_iou < min_det_iou:
        det_marks = 0
    else:
        if det_acc >= max_det_acc:
            det_acc_marks = 100
        else:
            det_acc_marks = min_marks + (max_marks - min_marks) * (det_acc - min_det_acc) / (max_det_acc - min_det_acc)
        print(f'det_acc_marks: {det_acc_marks:.2f}%')

        if det_iou >= max_det_iou:
            det_iou_marks = 100
        else:
            det_iou_marks = min_marks + (max_marks - min_marks) * (det_iou - min_det_iou) / (max_det_iou - min_det_iou)

        print(f'det_iou_marks: {det_iou_marks:.2f}%')

        det_marks = (det_acc_marks + det_iou_marks) / 2

    print(f'det_marks: {det_marks:.2f}%')

    overall_marks = cls_marks * cls_wt + det_marks * det_wt

    print(f'overall_marks: {overall_marks:.2f}%')


def compute_iou_multi(bboxes_1, bboxes_2):
    """

    compute overlap between each pair of objects in two sets of objects
    can be used for computing overlap between all detections and annotations in a frame
    """
    if len(bboxes_1.shape) == 1:
        bboxes_1 = bboxes_1.reshape((1, 4))

    if len(bboxes_2.shape) == 1:
        bboxes_2 = bboxes_2.reshape((1, 4))

    n1 = bboxes_1.shape[0]
    n2 = bboxes_2.shape[0]

    ul_1 = bboxes_1[:, :2]  # n1 x 2
    ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    ul_2 = bboxes_2[:, :2]  # n2 x 2
    ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_1 = bboxes_1[:, 2:]  # n1 x 2
    size_2 = bboxes_2[:, 2:]  # n2 x 2

    br_1 = ul_1 + size_1 - 1  # n1 x 2
    br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    br_2 = ul_2 + size_2 - 1  # n2 x 2
    br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_inter = np.minimum(br_1_rep, br_2_rep) - np.maximum(ul_1_rep, ul_2_rep) + 1  # n2 x 2 x n1
    size_inter[size_inter < 0] = 0
    # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1
    area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1
    area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_union = area_1_rep + area_2_rep - area_inter  # n2 x 1 x n1

    iou = np.divide(area_inter, area_union)  # n1 x n2

    return iou


def run_classifier_in_batch(classifier, patches, batch_size, device):
    import torch
    n_patches = patches.shape[0]
    n_batches = int(np.ceil(n_patches / batch_size))
    outputs = []
    for batch_id in range(n_batches):
        batch_start_id = batch_id * batch_size
        batch_end_id = (batch_id + 1) * batch_size
        if batch_end_id > n_patches:
            batch_end_id = n_patches
        patches_batch = patches[batch_start_id:batch_end_id, ...]
        outputs_batch = classifier(patches_batch.to(device))
        outputs.append(outputs_batch)
    outputs = torch.concatenate(outputs, dim=0)

    return outputs


def bbox_iou(bb_det, bb_gt):
    _, intersection_area = bbox_intersection(bb_det, bb_gt)
    if intersection_area == 0:
        return 0

    det_x1, det_y1, det_x2, det_y2 = bb_det.astype(np.float32)
    gt_x1, gt_y1, gt_x2, gt_y2 = bb_gt.astype(np.float32)

    det_w, det_h = det_x2 - det_x1, det_y2 - det_y1
    gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1

    union_area = (det_w * det_h) + (gt_w * gt_h) - intersection_area

    # compute overlap (IoU) = area of intersection / area of union
    iou = intersection_area / union_area

    return iou


def one_hot(labels, num_classes):
    probs = np.zeros((labels.size, num_classes), dtype=np.float32)
    probs[np.arange(labels.size), labels] = 1
    return probs


def draw_bbox(img, bbox, label='', col='green', thickness=1):
    import cv2
    xmin, ymin, xmax, ymax = bbox.astype(np.int32)
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                  col_bgr[col], thickness=thickness)
    if label:
        cv2.putText(img, label, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col_bgr[col])


def compute_det_acc(pred, gt, order_agnostic=True):
    assert pred.shape == gt.shape, "mismatch between the shapes of pred anf gt classes"

    if not order_agnostic:
        return (pred == gt).astype(int).sum() / gt.size

    n_images = pred.shape[0]
    correct = 0
    for i in range(n_images):
        pred1, pred2 = pred[i]
        gt1, gt2 = gt[i]

        if pred1 == gt2 or pred2 == gt1:
            """
            fix the order of the two detected classes by comparing with the ground truth
            """
            pred1, pred2 = pred2, pred1

        if pred1 == gt1:
            correct += 1

        if pred2 == gt2:
            correct += 1

    acc = correct / float(gt.size)

    return acc


def compute_det_iou(bboxes_pred, bboxes_gt, order_agnostic=True):
    """

    :param bboxes_pred: predicted bounding boxes, shape=(n_images,2,4)
    :param bboxes_gt: ground truth bounding boxes, shape=(n_images,2,4)
    :param bool order_agnostic:
    :return:
    """
    assert bboxes_pred.shape == bboxes_gt.shape, "mismatch between the shapes of pred anf gt bboxes"

    n_images = np.shape(bboxes_gt)[0]
    iou_sum = 0.0
    for i in range(n_images):
        iou1 = bbox_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 0, :])
        iou2 = bbox_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 1, :])

        iou_sum1 = iou1 + iou2

        if order_agnostic:
            """IOU evaluation is order agnostic so compare both possible combinations of predicted and GT boxes and 
            take the one that gives higher IOU"""
            iou1 = bbox_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 1, :])
            iou2 = bbox_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 0, :])

            iou_sum2 = iou1 + iou2

            if iou_sum2 > iou_sum1:
                iou_sum1 = iou_sum2

        iou_sum += iou_sum1

    mean_iou = iou_sum / (2. * n_images)

    return mean_iou


def bbox_intersection(bb1, bb2):
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1.astype(np.float32)
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2.astype(np.float32)

    bb_itsc = np.asarray([
        max(bb1_x1, bb2_x1),
        max(bb1_y1, bb2_y1),
        min(bb1_x2, bb2_x2),
        min(bb1_y2, bb2_y2)])

    iw = bb_itsc[2] - bb_itsc[0]
    ih = bb_itsc[3] - bb_itsc[1]

    area = 0 if iw <= 0 or ih <= 0 else iw * ih

    return bb_itsc, area


def stack_images(img_list, stack_order=0, grid_size=None, border=False):
    """

    :param img_list:
    :param int stack_order:
    :param list | None | tuple grid_size:
    :return:
    """
    if isinstance(img_list, (tuple, list)):
        n_images = len(img_list)
        img_shape = img_list[0].shape
        is_list = 1
    else:
        n_images = img_list.shape[0]
        img_shape = img_list.shape[1:]
        is_list = 0

    if grid_size is None:
        grid_size = [int(np.ceil(np.sqrt(n_images))), ] * 2
    else:
        if len(grid_size) == 1:
            grid_size = [grid_size[0], grid_size[0]]
        elif grid_size[0] == -1:
            grid_size = [int(math.ceil(n_images / grid_size[1])), grid_size[1]]
        elif grid_size[1] == -1:
            grid_size = [grid_size[0], int(math.ceil(n_images / grid_size[0]))]

    stacked_img = None
    list_ended = False
    inner_axis = 1 - stack_order
    for row_id in range(grid_size[0]):
        start_id = grid_size[1] * row_id
        curr_row = None
        for col_id in range(grid_size[1]):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_shape, dtype=np.uint8)
                list_ended = True
            else:
                if is_list:
                    curr_img = img_list[img_id]
                else:
                    curr_img = img_list[img_id, :, :].squeeze()
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if border:
                    border_img = np.full_like(curr_row, 255)[:, :1, ...]
                    curr_row = np.concatenate((curr_row, border_img, curr_img), axis=inner_axis)
                else:
                    curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if border:
                border_img = np.full_like(curr_row, 255)[:1, ...]
                stacked_img = np.concatenate((stacked_img, border_img, curr_row), axis=stack_order)
            else:
                stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        if list_ended:
            break
    return stacked_img


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=0, only_border=0, only_shrink=0, strict=False):
    src_height, src_width = src_img.shape[:2]
    src_aspect_ratio = float(src_width) / float(src_height)

    if len(src_img.shape) == 3:
        n_channels = src_img.shape[2]
    else:
        n_channels = 1

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        if only_shrink and width > src_width:
            width = src_width
        if only_border:
            height = src_height
        else:
            height = int(width / src_aspect_ratio)
    elif width <= 0:
        if only_shrink and height > src_height:
            height = src_height
        if only_border:
            width = src_width
        else:
            width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if strict:
        assert aspect_ratio == src_aspect_ratio, "aspect_ratio mismatch"

    if only_border:
        dst_width = width
        dst_height = height
        if placement_type == 0:
            start_row = start_col = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
            start_col = int(dst_width - src_width)
        else:
            raise AssertionError('Invalid placement_type: {}'.format(placement_type))
    else:

        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            if placement_type == 0:
                start_row = 0
            elif placement_type == 1:
                start_row = int((dst_height - src_height) / 2.0)
            elif placement_type == 2:
                start_row = int(dst_height - src_height)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            if placement_type == 0:
                start_col = 0
            elif placement_type == 1:
                start_col = int((dst_width - src_width) / 2.0)
            elif placement_type == 2:
                start_col = int(dst_width - src_width)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=src_img.dtype)
    dst_img = dst_img.squeeze()

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    if not only_border:
        dst_img = cv2.resize(dst_img, (width, height))

    if return_factors:
        resize_factor = float(height) / float(dst_height)
        start_pos = (start_row, start_col)
        return dst_img, resize_factor, start_pos
    else:
        return dst_img


def annotate(img, label, fmt=None):
    if fmt is None:
        """use default format"""
        fmt = CVText()

    size = fmt.size

    color = col_bgr[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    if len(img.shape) == 2:
        img = np.stack([img, ] * 3, axis=2)

    (text_width, text_height) = cv2.getTextSize(
        label, font,
        fontScale=fmt.size,
        thickness=fmt.thickness)[0]

    text_height += fmt.offset[1]
    text_width += fmt.offset[0]
    label_img = np.zeros((text_height, text_width), dtype=np.uint8)
    cv2.putText(label_img, label, tuple(fmt.offset),
                font, size, color, fmt.thickness, line_type)

    if len(img.shape) == 3:
        label_img = np.stack([label_img, ] * 3, axis=2)

    if text_width < img.shape[1]:
        label_img = resize_ar(label_img, width=img.shape[1], height=text_height,
                              only_border=2, placement_type=1)
    elif text_width > img.shape[1]:
        img = resize_ar(img, width=label_img.shape[1],
                        only_border=2, placement_type=1)

    border_img = np.full_like(img, 255)[:1, ...]
    img_list_label = [label_img,
                      border_img,
                      img]

    img = stack_images(img_list_label, grid_size=(-1, 1))

    return img


def get_patches(image_, patch_size, stride):
    from patchify import patchify
    if len(image_.shape) == 3:
        patch_size = tuple(list(patch_size) + [3, ])
    patches_np = patchify(image_, patch_size, step=stride)
    patches_np = patches_np.reshape((-1, 28, 28, 3))
    return patches_np


def get_patch_bboxes(image_size, patch_size, stride):
    im_H, im_W = image_size
    p_H, p_W = patch_size
    y = np.arange(0, im_H - (p_H - 1), stride)
    x = np.arange(0, im_W - (p_W - 1), stride)
    X, Y = np.meshgrid(x, y)
    patch_pos = np.dstack((X, Y)).reshape(-1, 2)
    patch_bboxes = np.concatenate((patch_pos, patch_pos + 28), axis=1)
    return patch_bboxes


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def add_suffix(src_path, suffix, dst_ext='', sep='_'):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    if not dst_ext:
        dst_ext = src_ext

    dst_path = linux_path(src_dir, src_name + sep + suffix + dst_ext)
    return dst_path


def vis_cls(batch_id, imgs, batch_size, targets, predicted, is_correct, class_names, show, save, save_dir, pause):
    imgs_np = imgs.detach().cpu().numpy()
    concat_imgs = []
    for i in range(batch_size):
        img = imgs_np[i, ...].squeeze()

        """switch image from 3 x 28 x 28 to 28 x 28 x 3 since opencv expects channels 
        to be on the last axis
        copy is needed to resolve an opencv issue with memory layout:
        https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array
        -incompatible-with-cvmat
        """
        img = np.transpose(img, (1, 2, 0)).copy()

        img = (img * 255.0).astype(np.uint8)

        target = targets[i]
        output = predicted[i]
        _is_correct = is_correct[i].item()
        if _is_correct:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)

        pred_img = np.zeros_like(img)
        pred_cls = class_names[int(output)]
        if pred_cls == 'bkg':
            pred_cls = 'K'
        elif len(pred_cls) > 1:
            pred_cls = pred_cls[0]

        cv2.putText(pred_img, f'{pred_cls:s}', (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, col, 1, cv2.LINE_AA)

        label_img = np.zeros_like(img)
        gt_cls = class_names[int(target)]
        if gt_cls == 'bkg':
            gt_cls = 'K'
        cv2.putText(label_img, f'{gt_cls:s}', (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

        concat_img = np.concatenate((img, label_img, pred_img), axis=0)
        concat_imgs.append(concat_img)

    vis_img = np.concatenate(concat_imgs, axis=1)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/batch_{batch_id}.jpg', vis_img)

    if show:
        win_title = 'Press Esc to exit, Space to toggle pause, any other key to continue'
        cv2.imshow(win_title, vis_img)
        k = cv2.waitKey(0 if pause else 500)
        if k == 27:
            exit(0)
        elif k == 32:
            pause = 1 - pause

    return vis_img, pause


def vis_patches(img_id, image, gt_bboxes, gt_cls_ids, patches, bboxes, cls_ids, probs, class_names, pause,
                show, save, save_dir, vis_size=300):
    vis_images = []

    for gt_bbox, gt_cls_id, patch, bbox, cls_id, prob in zip(gt_bboxes, gt_cls_ids, patches, bboxes, cls_ids, probs):
        vis_img, scale_factor, _ = resize_ar(image, vis_size, return_factors=True)
        cls = class_names[cls_id]
        gt_cls = class_names[gt_cls_id]
        draw_bbox(vis_img, gt_bbox * scale_factor, col='green', thickness=2)
        draw_bbox(vis_img, bbox * scale_factor, col='red', thickness=2)
        vis_patch = resize_ar(patch, vis_size)
        vis_img = stack_images((vis_img, vis_patch), grid_size=(1, 2))
        vis_img = annotate(vis_img, f'GT: {gt_cls} Pred: {cls:s}: {prob:.1f}')

        vis_images.append(vis_img)

    vis_images = stack_images(vis_images, grid_size=(-1, 1))

    vis_images = resize_ar(vis_images, width=900, only_border=2, placement_type=1)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/img_{img_id}.jpg', vis_images)

    if show:
        win_title = f'{save_dir}: Press Esc to exit, Space to toggle pause, any other key to continue'
        cv2.imshow(win_title, vis_images)

        k = cv2.waitKey(0 if pause else 200)
        if k == 27:
            exit()
        elif k == 32:
            pause = 1 - pause

    return pause


def vis_patches_and_probs(img_id, img, gt_bboxes, patches, probs, bboxes, class_names,
                          show, save, save_dir, min_prob=0.001):
    _pause = 1
    if save:
        os.makedirs(save_dir, exist_ok=True)

    for patch_id, (patch, prob, bbox) in enumerate(zip(patches, probs, bboxes)):
        # label_txt = f'img {img_id} patch {patch_id} '
        label_txt = f''
        for class_id, (class_name, class_prob) in enumerate(zip(class_names, prob)):
            if class_prob > min_prob:
                label_txt += f'{class_name}: {class_prob * 100:.1f}  '

        vis_img = (img * 255.).astype(np.uint8)
        vis_patch = (patch * 255.).astype(np.uint8)

        vis_patch = resize_ar(vis_patch, 300)
        vis_img, scale_factor, start_pos = resize_ar(vis_img, 300, return_factors=True)
        for gt_bbox in gt_bboxes:
            draw_bbox(vis_img, gt_bbox * scale_factor, col='green')
        draw_bbox(vis_img, bbox * scale_factor, col='red')
        vis_img = stack_images((vis_img, vis_patch), grid_size=(1, 2), border=True)
        vis_img = annotate(vis_img, label_txt, fmt=CVText(font=0))

        vis_img = resize_ar(vis_img, width=700, only_border=2, placement_type=1)

        if save:
            cv2.imwrite(f'{save_dir}/img_{img_id}_patch_{patch_id}.png', vis_img)

        if show:
            win_title = f'{save_dir}: Press Esc to exit, Space to toggle pause, any other key to continue'
            cv2.imshow(win_title, vis_img)

            k = cv2.waitKey(0 if _pause else 1)
            if k == 27:
                exit()
            elif k == 32:
                _pause = 1 - _pause


def get_device(use_gpu):
    import torch
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Running on CPU')
    return device


def extract_patches_and_probs(img_id, img, patches, patch_bboxes, gt_bboxes, gt_classes, bbox_area,
                              patches_per_img, class_names, show, save):
    n_patches = patches.shape[0]
    n_classes = len(class_names)

    random_patch_ids = random.sample(list(range(n_patches)), patches_per_img)

    random_bboxes = patch_bboxes[random_patch_ids, :]
    random_patches = patches[random_patch_ids, :]

    gt_itsc_bbox, gt_itsc = bbox_intersection(gt_bboxes[0], gt_bboxes[1])

    patch_itsc_1 = np.asarray([bbox_intersection(gt_bboxes[0], patch_bbox)[1] / bbox_area
                               for patch_bbox in random_bboxes])
    patch_itsc_2 = np.asarray([bbox_intersection(gt_bboxes[1], patch_bbox)[1] / bbox_area
                               for patch_bbox in random_bboxes])
    patch_itsc_frg = patch_itsc_1 + patch_itsc_2
    if gt_itsc > 0:
        patch_itsc_12 = np.asarray([bbox_intersection(gt_itsc_bbox, patch_bbox)[1] / bbox_area
                                    for patch_bbox in random_bboxes])
        patch_itsc_frg -= patch_itsc_12

    patch_itsc_bkg = 1 - patch_itsc_frg

    patch_itscs = np.stack((patch_itsc_1, patch_itsc_2, patch_itsc_bkg), axis=1)

    patch_itscs_sum = np.sum(patch_itscs, axis=1).reshape((-1, 1))
    # patch_probs = scipy.special.softmax(patch_itscs, axis=1)
    patch_probs = patch_itscs / patch_itscs_sum
    # patch_probs_sum = np.sum(patch_probs, axis=1).reshape((-1, 1))

    random_probs = np.zeros((patches_per_img, n_classes), dtype=np.float32)
    random_probs[:, gt_classes[0]] = patch_probs[:, 0]
    random_probs[:, gt_classes[1]] = patch_probs[:, 1]
    random_probs[:, -1] = patch_probs[:, 2]

    if show or show:
        vis_patches_and_probs(img_id, img, gt_bboxes, random_patches, random_probs, random_bboxes, class_names,
                              show=show, save=save, save_dir='vis/extract_patches')

    return random_patches, random_probs
