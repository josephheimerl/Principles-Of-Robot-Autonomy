import numpy as np


def box_iou(a, b):
    """
    Pairwise intersection over union of boxes in (x1, y1, x2, y2) format.
    Inputs:
        a: (N, 4), b: (M, 4)
    Output:
        (N, M) array, entry [i, j] is IoU(a[i], b[j])

    HINT: broadcast a[:, None, :] against b[None, :, :]; no for-loops needed.
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def nms(boxes, scores, iou_threshold):
    """
    Greedy non-maximum suppression.
    Inputs:
        boxes: (N, 4), scores: (N,)
        iou_threshold: a box is removed if its IoU with a higher-scoring kept box is > iou_threshold
    Output:
        (K,) integer indices of the kept boxes, sorted by decreasing score
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def batched_nms(boxes, scores, labels, iou_threshold):
    """
    NMS applied separately within each class label, so boxes of different
    classes never suppress each other.
    Output:
        (K,) integer indices of the kept boxes over all classes, sorted by decreasing score
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def match_detections(boxes, scores, gt_boxes, iou_threshold=0.5):
    """
    Labels each detection as a true positive or a false positive.
    Detections are processed in decreasing score order. Each one is matched to
    the ground-truth box it overlaps most; it is a true positive if that IoU is
    >= iou_threshold and that ground-truth box has not been matched yet,
    otherwise it is a false positive.
    Inputs:
        boxes: (N, 4), scores: (N,), gt_boxes: (G, 4)
    Output:
        (N,) boolean array aligned with the input order (True = true positive)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def precision_recall(scores, tp, num_gt):
    """
    Precision and recall when keeping the top-k detections, for k = 1, ..., N.
    Inputs:
        scores: (N,), tp: (N,) boolean from match_detections, num_gt: number of ground-truth boxes
    Output:
        precision, recall: (N,) arrays, entry k-1 uses the k highest-scoring detections
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def average_precision(precision, recall):
    """
    All-point interpolated average precision (area under the PR curve), where the
    precision at each recall is replaced by the maximum precision at any higher recall.
    Inputs:
        precision, recall: (N,) arrays from precision_recall
    Output:
        float AP in [0, 1]
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
