import cv2
import random
import colorsys
import collections
import main_utils

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

# COCO classes
class_names = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def horizontal_true_vertical_false(line):
    hrz_difference = line[1][0]-line[0][0]
    vtc_difference = line[1][1]-line[0][1]
    squared_difference =  hrz_difference ** 2 - vtc_difference ** 2
    
    if squared_difference >= 0: 
        return True
    else:
        return False

class Visualizer():
    def __init__(self, line) -> None:
        self.counter_dict = collections.OrderedDict()
        self.counter_memory = dict.fromkeys(range(54000), 0)
        self.line = line
        self.horizontal_True_vertical_False = horizontal_true_vertical_false(line)
        self.font_style = cv2.FONT_HERSHEY_SIMPLEX

    def draw_data_panel(self, img, bboxes, fps):
        if bboxes is None:
            target_num = 0
        else:
            target_num = len(bboxes)
        num_recorded_class = len(self.counter_dict)

        alpha = 0.3
        image_h, image_w, _ = img.shape
        overlay = img.copy()
        cv2.rectangle(img, (0, 0), (image_w//3 - 20, num_recorded_class * 40 + 110), (32,36,46), thickness=-1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        vertical_increment = 20
        vertical_correction = 20
        horizontal_increment = image_w // 5
        up_or_left_sum = 0
        down_or_right_sum = 0
        sum_persons = 0
        text_thickness = int((image_h + image_w) / 800)
        font_scale = 1
        sum_increment = num_recorded_class * 20 + 50

        # counter_dict = { speedboat {'up': 0, 'down': 0, 'left': 0, 'right': 0}, ..., river_boat {'up': 0, 'down': 0, 'left': 0, 'right': 0} }
        for key, values in self.counter_dict.items():
            vertical_correction += vertical_increment 

            if self.horizontal_True_vertical_False:
                cv2.putText(img," up: {}".format(values['up']),(0, vertical_correction), self.font_style, 
                            font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
                cv2.putText(img," down: {}".format(values['down']),(0, vertical_correction + 30), self.font_style, 
                            font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
                up_or_left_sum += values['up']
                down_or_right_sum += values['down']
                sum_persons = values['up'] + values['down']

            else:
                cv2.putText(img," left: {}".format(values['left']),(0, vertical_correction), self.font_style, 
                            font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
                cv2.putText(img," right: {}".format(values['right']),(0, vertical_correction + 30), self.font_style, 
                            font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
                up_or_left_sum += values['left']
                down_or_right_sum += values['right']
                sum_persons = values['left'] + values['right']

        cv2.putText(img, f" total: {sum_persons}" ,(0, sum_increment + 30), self.font_style, 
                    font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, f" fps: {round(fps, 2)}" ,(0, sum_increment + 60), self.font_style, 
                    font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        
        return img

    def draw_line_and_area(self, img, line):
        cv2.line(img, line[0], line[1], (8,196,254), thickness=1, lineType=cv2.LINE_AA)
        return img

    def draw_bbox(self, img, box, cls_name, identity=None, offset=(0,0)):
        '''
            draw box of an id
        '''
        x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
        # set color and label text
        color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
        label = '{} {}'.format(cls_name, identity)
        # box text and bar
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(img,(x1, y1),(x2, y2),color,2)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
        return img


    def draw_bboxes(self, img, bbox, identities=None, offset=(0,0)):
        for i, box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = COLORS_10[id%len(COLORS_10)]
            label = '{} {}'.format("object", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        
        return img


    def add_cls_confi_draw_bboxes(self, img, bbox, identities=None, confidences=None, class_nums=None,points=None, offset=(0,0)):
        image_h, image_w, _ = img.shape
        num_classes = len(class_names)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        p0 = (0, 0)
        p1 = (0, 0)

        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]

            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            track_id = int(identities[i]) if identities is not None else 0

            class_name = class_names[class_nums[i]]
            confidence = confidences[i]/100

            bbox_color = colors[class_nums[i]]
            bbox_thick = int(0.7 * (image_h + image_w) / 500)
            fontScale = 0.5
            
            if class_name not in self.counter_dict:
                self.counter_dict[class_name] = {}
                self.counter_dict[class_name]['up'] = 0
                self.counter_dict[class_name]['down'] = 0
                self.counter_dict[class_name]['left'] = 0
                self.counter_dict[class_name]['right'] = 0

            if len(points[track_id]) >= 3:
                p0 = points[track_id][-1]
                p1 = points[track_id][-3]

                if main_utils.intersect(p0, p1, self.line[0], self.line[1]) and self.counter_memory[track_id] != 1:
                    if self.horizontal_True_vertical_False:
                        if p0[1] < p1[1]:
                            self.counter_dict[class_name]['up'] += 1
                            self.counter_memory[track_id] = 1
                        elif p0[1] > p1[1]:
                            self.counter_dict[class_name]['down'] += 1
                            self.counter_memory[track_id] = 1
                    else:
                        if p0[0] < p1[0]:
                            self.counter_dict[class_name]['left'] += 1
                            self.counter_memory[track_id] = 1
                        elif p0[0] > p1[0]:
                            self.counter_dict[class_name]['right'] += 1
                            self.counter_memory[track_id] = 1

            cv2.rectangle(img,(x1, y1),(x2,y2), bbox_color, bbox_thick)
            label = "{}: {}".format(class_name, confidence)
            t_size = cv2.getTextSize(label, 0, fontScale, thickness=bbox_thick)[0]
            cv2.rectangle(img, (x1, y1 -3), (x1 + t_size[0], y1 - t_size[1] - 6), bbox_color, thickness=-1)
            cv2.putText(img, label, (x1,y1 - 5), self.font_style, fontScale, (0,0,0), bbox_thick//3,lineType=cv2.LINE_AA)

            # draw track id
            for j in range(1, len(points[track_id])):
                if points[track_id][j - 1] is None or points[track_id][j] is None:
                    continue
                cv2.line(img,(points[track_id][j-1]), (points[track_id][j]),(8, 196, 255),thickness = 2,lineType=cv2.LINE_AA)

            cv2.circle(img,  (p0), radius=3, color=(0, 0, 250), thickness=-1,lineType=cv2.LINE_AA)
            cv2.putText(img,"{}".format(track_id),(p0[0]+5, p0[1]+5), self.font_style, 0.8*fontScale, (0, 0, 250), 1,lineType=cv2.LINE_AA)

        return img