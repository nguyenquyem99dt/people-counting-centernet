import cv2
from tqdm import tqdm
import argparse
import consts
from deep_sort import DeepSort
from visualization import Visualizer
import main_utils
import time
import sys
sys.path.append('./CenterNet/src/lib')
from detectors.detector_factory import detector_factory
from opts import opts

class Detector(object):
    def __init__(self, opt):
        self.video = cv2.VideoCapture(args.input_video)

        # CenterNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort(model_path=args.track_model, use_cuda=True, use_trt=args.use_trt)
        self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = (0, 0, self.im_width, self.im_height)
        self.filter_class = args.filter_class
        self.write_video = True if args.output_video is not None else False
        self.show_video = args.show_video
        self.output_video = args.output_video

    def detect(self):
        visualizer = Visualizer(line=consts.LINE)
        if self.write_video:
            fps = self.video.get(cv2.CAP_PROP_FPS)
            encode = cv2.VideoWriter_fourcc(*'mp4v')
            self.output = cv2.VideoWriter(self.output_video, encode, fps, (self.im_width, self.im_height))
        
        xmin, ymin, xmax, ymax = self.area
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        _, ori_im = self.video.read()

        t0 = time.time()
        fpss = []
        for _ in tqdm(range(total_frames)):
            im = ori_im[ymin:ymax, xmin:xmax]
            start = time.time()

            results = self.detector.run(im)['results']
            if self.filter_class is not None:
                try:
                    results = dict((key, value) for key, value in results.items() if key == self.filter_class)
                except NameError:
                    pass

            bbox_xywh, cls_conf, cls_num = main_utils.bbox_to_xywh_cls_conf(results, conf_thresh=opt.vis_thresh)
            im = visualizer.draw_line_and_area(im, line=consts.LINE)

            if bbox_xywh is not None:
                outputs, points = self.deepsort.update(bbox_xywh, cls_conf, cls_num, im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4] # [[x1,y1,x2,y2],....]
                    identities = outputs[:, 4] # track_id [1, 2]
                    confidences = outputs[:, 5] # confidences [54, 62]
                    class_nums = outputs[:, -1] # class_num [1, 1]
                    ori_im = visualizer.add_cls_confi_draw_bboxes(ori_im, bbox_xyxy, identities, confidences, class_nums, points, offset=(xmin, ymin))

            end = time.time()
            fps = 1 / (end-start)
            fpss.append(fps)
            ori_im = visualizer.draw_data_panel(ori_im, bbox_xywh, fps)

            if self.show_video:
                win_name = 'People Counting - CLV'
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 960, 540)
                cv2.imshow(win_name, ori_im)
                cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)

            _, ori_im = self.video.read()

        total_infer_time = time.time() - t0
        print('Total inference time:', total_infer_time)
        print('Avg FPS:', sum(fpss)/len(fpss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', type=str, required=True)
    parser.add_argument('--output-video', type=str, default=None)
    parser.add_argument('--task', type=str, default='ctdet')
    parser.add_argument('--filter-class', type=int, default=1) # person
    parser.add_argument('--det-model', type=str, default='./weights/ctdet_coco_resdcn18.pth')
    parser.add_argument('--track-model', type=str, default='./weights/ckpt.t7')
    parser.add_argument('--arch', choices=['res_18', 'res_101', 'resdcn_18', 'resdcn_101', 'dlav0_34', 'dla_34', 'hourglass'], default='resdcn_18')
    parser.add_argument('--show-video', action='store_true')
    parser.add_argument('--use-trt', action='store_true')
    args = parser.parse_args()

    opt = opts().init('--task {} --load_model {} --arch {}'.format(args.task, args.det_model, args.arch).split(' ')) # opts of centernet
    if args.use_trt:
        opt.use_trt = True
    print(args)
    det = Detector(opt)
    det.detect()
