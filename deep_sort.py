import numpy as np
from deep.feature_extractor import Extractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker
import time
from collections import deque

points = [deque(maxlen=5) for _ in range(5400)]

class DeepSort(object):
    def __init__(self, model_path, use_cuda=True, use_trt=False):
        self.min_confidence = 0.25
        self.nms_max_overlap = 1.0
        self.extractor = Extractor(model_path, use_cuda, use_trt)

        metric = NearestNeighborDistanceMetric(metric="cosine", matching_threshold=0.2, budget=100)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, class_num, ori_img):
        self.height, self.width = ori_img.shape[:2]
        detections = []
        try:
            features = self._get_features(bbox_xywh, ori_img)
            for i, conf in enumerate(confidences):
                if conf >= self.min_confidence and features.any():
                    detections.append(Detection(bbox_xywh[i], conf, class_num[i], features[i]))
        except Exception as ex:
            print("{} Error: {}".format(time.strftime("%H:%M:%S", time.localtime()), ex))

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()  # (top left x, top left y, width, height)
            x1, y1, x2, y2 = self._xywh_to_xyxy_centernet(box)

            center = (int((x1+x2)/2),int((y1+y2)/2))
            points[track.track_id].append(center)

            track_id = track.track_id
            confidences = track.confidence * 100
            cls_num = track.class_num

            outputs.append(np.array([x1, y1, x2, y2, track_id, confidences, cls_num], dtype=np.int))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs, points

    # for centernet (x1,x2 w,h -> x1,y1,x2,y2)
    def _xywh_to_xyxy_centernet(self, bbox_xywh):
        x1, y1, w, h = bbox_xywh
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(int(x1+w), self.width-1)
        y2 = min(int(y1+h), self.height-1)
        return int(x1), int(y1), x2, y2

    # for yolo  (centerx,centerx, w,h -> x1,y1,x2,y2)
    def _xywh_to_xyxy_yolo(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        features = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy_centernet(box)
            im = ori_img[y1:y2, x1:x2]
            feature = self.extractor(im)[0]
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features

