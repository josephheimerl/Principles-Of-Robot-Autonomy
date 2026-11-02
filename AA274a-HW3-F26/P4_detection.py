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
    # TODO: make it 3d, this is elementwise comparison currently
    # TODO: currently assumes xy pairs are ascending order
    xint = np.maximum(0, np.minimum(a[:,None,2],b[None,:,2]) - np.maximum(a[:,None,0],b[None,:,0]))
    yint = np.maximum(0, np.minimum(a[:,None,3],b[None,:,3]) - np.maximum(a[:,None,1],b[None,:,1]))

    int_area = xint * yint
    a_area = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    b_area = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union_area = a_area[:,None] + b_area[None,:] - int_area

    return int_area/union_area
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
    descending_score = np.argsort(scores,descending=True)
    boxes_sorted = boxes[descending_score,:]
    iou_matrix = box_iou(boxes_sorted,boxes_sorted)
    over_iou_thresh = iou_matrix > iou_threshold
    indices = []
    while not descending_score.size == 0:
        indices.append(descending_score[0])
        keep_mask = ~over_iou_thresh[:,0]

        #clip off iou violators
        descending_score = descending_score[keep_mask]
        over_iou_thresh = over_iou_thresh[:,keep_mask][keep_mask,:]

    return indices
    ########## Code ends here ##########


def batched_nms(boxes, scores, labels, iou_threshold):
    """
    NMS applied separately within each class label, so boxes of different
    classes never suppress each other.
    Output:
        (K,) integer indices of the kept boxes over all classes, sorted by decreasing score
    """
    ########## Code starts here ##########
    kept_indices = np.array([],dtype=int)
    all_indices = np.arange(len(scores))

    for i in range(labels.max() + 1):
        # extract current label's relevant indices
        curr_label = labels == i
        curr_label_indices = all_indices[curr_label]

        # compute which indices of the set of current label's indices to keep
        new_indices = nms(boxes[curr_label], scores[curr_label], iou_threshold)

        # append overall indices to the running list
        kept_indices = np.concatenate((kept_indices,curr_label_indices[new_indices]))

    final_scores = scores[kept_indices]
    sorted_indices = np.argsort(final_scores,descending=True)
    return kept_indices[sorted_indices]
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
    descending_score = np.argsort(scores,descending=True)
    boxes_sorted = boxes[descending_score,:]
    ious = box_iou(boxes[descending_score],gt_boxes)
    true_positive = np.zeros(len(boxes),bool)
    for i in range(len(boxes)):
        # break early if all ground truths consumed
        if ious.shape[1] == 0:
            break

        max_idx = np.argmax(ious[i,:])
        if ious[i,max_idx] > iou_threshold:
            true_positive[i] = True
            keep_mask = np.arange(ious.shape[1]) != max_idx
            ious = ious[:,keep_mask]

    return true_positive
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
    descending_score = np.argsort(scores,descending=True)
    tp_cumsum = np.cumsum(tp[descending_score])
    k = np.arange(1,len(scores)+1)
    return (tp_cumsum/k,tp_cumsum/num_gt)
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
    # recall will automatically be monotonically increasing
    modified_precision = np.maximum.accumulate(precision[::-1])[::-1]
    dr = np.diff(recall,prepend=0)
    return np.sum(modified_precision*dr)
    ########## Code ends here ##########
