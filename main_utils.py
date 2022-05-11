import numpy as np

def bbox_to_xywh_cls_conf(bbox, conf_thresh):
    '''
    bbox format: array([x1, y1, x2, y2, confidence],...)
    return: [[x1, y1, x2, y2, confidence, cls_num],...]                
    '''
    new_bbox = []
    for cls_num, box in bbox.items():
        if not box.any():
            pass
        else:
            for single_box in box:
                if not single_box.any():
                    pass
                else:
                    a = np.append(single_box, cls_num)
                    new_bbox.append(a)
    new_bbox = np.array(new_bbox)

    # filter by confidence threshold
    if any(new_bbox[:, 4] > conf_thresh):
        new_bbox = new_bbox[new_bbox[:, 4] > conf_thresh, :] # > confidence thes
        new_bbox[:, 2] = new_bbox[:, 2] - new_bbox[:, 0]
        new_bbox[:, 3] = new_bbox[:, 3] - new_bbox[:, 1]
        return new_bbox[:, :4], new_bbox[:, 4], new_bbox[:, 5] # [[x,y,w,h], ...], [confidence,...], [cls_num,...]
    else:
        return None, None, None


def is_point_in(x, y, polygon_points):
    count = 0
    x1, y1 = polygon_points[0]
    x1_part = (y1 > y) or ((x1 - x > 0) and (y1 == y))
    x2, y2 = '', ''  # points[1]
    polygon_points.append((x1, y1))
    for point in polygon_points[1:]:
        x2, y2 = point
        x2_part = (y2 > y) or ((x2 > x) and (y2 == y))
        if x2_part == x1_part:
            x1, y1 = x2, y2
            continue
        mul = (x1 - x)*(y2 - y) - (x2 - x)*(y1 - y)
        if mul > 0:
            count += 1
        elif mul < 0:
            count -= 1
        x1, y1 = x2, y2
        x1_part = x2_part
    if count == 2 or count == -2:
        return True
    else:
        return False


def intersect(A,B,C,D):
	return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()

def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum()