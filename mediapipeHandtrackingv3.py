import depthai as dai
from pathlib import Path
from collections import namedtuple
import cv2
import numpy as np
import argparse
import utils.anchors
import utils.processing as mpu

FINGER_COLOR = [(128, 128, 128), (80, 190, 168),
         (234, 187, 105), (175, 119, 212),
         (81, 110, 221)]

JOINT_COLOR = [(0, 0, 0), (125, 255, 79),
            (255, 102, 0), (181, 70, 255),
            (13, 63, 255)]

# depth var
config = None
newConfig = False
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)
stepSize = 0.05
points_storage = []

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    return resized

pipeline = dai.Pipeline()

SSDAnchorOptions = namedtuple('SSDAnchorOptions', [
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

class HandTracker:
    def __init__(self,
                 pd_path="models/palm_detection_6_shaves.blob",
                 pd_score_thresh=0.5, pd_nms_thresh=0.3,
                 lm_path="models/hand_landmark_6_shaves.blob",
                 lm_score_threshold=0.5,
                 show_landmarks=True,
                 show_hand_box=True):

        self.pd_path = pd_path
        self.pd_input_length = 128
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.show_landmarks = show_landmarks
        self.show_hand_box = show_hand_box

        anchor_options = SSDAnchorOptions(num_layers=4,
                                          min_scale=0.1484375,
                                          max_scale=0.75,
                                          input_size_height=128,
                                          input_size_width=128,
                                          anchor_offset_x=0.5,
                                          anchor_offset_y=0.5,
                                          strides=[8, 16, 16, 16],
                                          aspect_ratios=[1.0],
                                          reduce_boxes_in_lowest_layer=False,
                                          interpolated_scale_aspect_ratio=1.0,
                                          fixed_anchor_size=True)

        self.anchors = utils.anchors.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        self.preview_width = 576
        self.preview_height = 324
        self.hand_middle = None
        self.hand_text_placement = None

        self.frame_size = None

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

        # color camera
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        # depth
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(self.preview_width, self.preview_height)
        spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutSpatialData = pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

        xoutDepth.setStreamName("depth")
        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # depth config
        global config
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(topLeft, bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(config)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)

        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

        # palm detection
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        # hand landmarks
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        self.lm_input_length = 224
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

        print("pipeline created")
        return pipeline

    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16)  # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape(
            (self.nb_anchors, 18))  # 896x18
        # Decode bboxes
        self.regions = utils.processing.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        mpu.detections_to_rect(self.regions)
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Squeeze"))

        lm = []
        for i in range(int(len(lm_raw) / 3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3 * i:3 * (i + 1)] / self.lm_input_length)
        region.landmarks = lm

    def lm_render(self, frame, original_frame, region):
        cropped_frame = None
        hand_bbox = []
        if region.lm_score > self.lm_score_threshold:
            palmar = True
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([(x, y) for x, y in region.rect_points[1:]],
                           dtype=np.float32)  # region.rect_points[0] is left bottom point !
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
            if self.show_landmarks:
                list_connections = [[0, 1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16],
                                    [17, 18, 19, 20]]
                palm_line = [np.array([lm_xy[point] for point in [0, 5, 9, 13, 17, 0]])]

                # Draw lines connecting the palm
                if region.handedness > 0.5:
                    # Simple condition to determine if palm is palmar or dorasl based on the relative
                    # position of thumb and pinky finger
                    if lm_xy[4][0] > lm_xy[20][0]:
                        cv2.polylines(frame, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.polylines(frame, palm_line, False, (128, 128, 128), 2, cv2.LINE_AA)
                else:
                    # Simple condition to determine if palm is palmar or dorasl based on the relative
                    # position of thumb and pinky finger
                    if lm_xy[4][0] < lm_xy[20][0]:
                        cv2.polylines(frame, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.polylines(frame, palm_line, False, (128, 128, 128), 2, cv2.LINE_AA)

                # Draw line for each finger
                for i in range(len(list_connections)):
                    finger = list_connections[i]
                    line = [np.array([lm_xy[point] for point in finger])]
                    if region.handedness > 0.5:
                        if lm_xy[4][0] > lm_xy[20][0]:
                            palmar = True
                            cv2.polylines(frame, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
                            for point in finger:
                                pt = lm_xy[point]
                                cv2.circle(frame, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
                        else:
                            palmar = False
                    else:
                        if lm_xy[4][0] < lm_xy[20][0]:
                            palmar = True
                            cv2.polylines(frame, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
                            for point in finger:
                                pt = lm_xy[point]
                                cv2.circle(frame, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
                        else:
                            palmar = False

                    # Use different colour for the hand to represent dorsal side
                    if not palmar:
                        cv2.polylines(frame, line, False, (128, 128, 128), 2, cv2.LINE_AA)
                        for point in finger:
                            pt = lm_xy[point]
                            cv2.circle(frame, (pt[0], pt[1]), 3, (0, 0, 0), -1)

            # Calculate the bounding box for the entire hand
            max_x = 0
            max_y = 0
            min_x = frame.shape[1]
            min_y = frame.shape[0]
            for x, y in lm_xy:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            box_width = max_x - min_x
            box_height = max_y - min_y
            x_center = min_x + box_width / 2
            y_center = min_y + box_height / 2

            # Enlarge the hand bounding box for drawing use
            draw_width = box_width / 2 * 1.2
            draw_height = box_height / 2 * 1.2
            draw_size = max(draw_width, draw_height)

            draw_min_x = int(x_center - draw_size)
            draw_min_y = int(y_center - draw_size)
            draw_max_x = int(x_center + draw_size)
            draw_max_y = int(y_center + draw_size)

            hand_bbox = [draw_min_x, draw_min_y, draw_max_x, draw_max_y]


            if self.show_hand_box:

                cv2.rectangle(frame, (draw_min_x, draw_min_y), (draw_max_x, draw_max_y), (36, 152, 0), 2)
                cv2.putText(frame, f"({int(x_center)}, {int(y_center)})", (int(x_center), int(y_center)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                global points_storage
                # def mapping_val(old_val, old_min, old_max, new_min, new_max):
                #     old_range = (old_max - old_min)
                #     new_range = (new_max - new_min)
                #     return (((old_val - old_min) * new_range) / old_range) + new_min
                # points_storage = dict(topLeft=(mapping_val(draw_min_x, 0, 576, 0.05, 0.95),
                #                                mapping_val(draw_max_y, 0, 324, 0.05, 0.95)),
                #                       bottomRight=(mapping_val(draw_max_x, 0, 576, 0.05, 0.95),
                #                                mapping_val(draw_min_y, 0, 324, 0.05, 0.95)))
                # map values to
                draw_min_x = draw_min_x/576
                draw_max_y = draw_max_y/324
                draw_max_x = draw_max_x/576
                draw_min_y = draw_min_y/324
                points_storage = dict(topLeft=(draw_min_x, draw_max_y),
                                      bottomRight=(draw_max_x, draw_min_y))

        return cropped_frame, region.handedness, hand_bbox

    def run(self):
        device = dai.Device(self.create_pipeline())
        device.startPipeline()

        q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        q_pd_in = device.getInputQueue(name="pd_in")
        q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
        q_lm_in = device.getInputQueue(name="lm_in")
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_spatial_calc = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        q_spatial_calc_conf = device.getInputQueue("spatialCalcConfig")

        while True:
            in_video = q_video.get()
            video_frame = in_video.getCvFrame()

            h, w = video_frame.shape[:2]
            self.frame_size = max(h, w)
            self.pad_h = int((self.frame_size - h) / 2)
            self.pad_w = int((self.frame_size - w) / 2)

            video_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w,
                                             cv2.BORDER_CONSTANT)

            frame_nn = dai.ImgFrame()
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
            q_pd_in.send(frame_nn)

            annotated_frame = video_frame.copy()

            # Get palm detection
            inference = q_pd_out.get()
            self.pd_postprocess(inference)

            # Send data for hand landmarks
            for i, r in enumerate(self.regions):
                img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                q_lm_in.send(nn_data)

            # Retrieve hand landmarks
            for i, r in enumerate(self.regions):
                inference = q_lm_out.get()
                self.lm_postprocess(r, inference)
                hand_frame, handedness, hand_bbox = self.lm_render(video_frame, annotated_frame, r)


            # depthmap
            inDepth = q_depth.get()  # Blocking call, will wait until a new data has arrived

            depthFrame = inDepth.getFrame()  # depthFrame values are in millimeters

            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = q_spatial_calc.get().getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                depthMin = depthData.depthMin
                depthMax = depthData.depthMax

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (255,255,255),
                              cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                            fontType, 0.5, (255, 255, 255))
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                            fontType, 0.5, (255, 255, 255))
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                            fontType, 0.5, (255, 255, 255))
            # Show the frame
            cv2.imshow("depth", depthFrameColor)

            video_frame = video_frame[self.pad_h:self.pad_h + h, self.pad_w:self.pad_w + w]
            cv2.imshow("hand tracker", video_frame)

            global newConfig
            global points_storage

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('1'):
                self.show_hand_box = not self.show_hand_box
            elif key == ord('2'):
                self.show_landmarks = not self.show_landmarks
            # elif key == ord('w'):
            #     if topLeft.y - stepSize >= 0:
            #         topLeft.y -= stepSize
            #         bottomRight.y -= stepSize
            #         newConfig = True
            # elif key == ord('a'):
            #     if topLeft.x - stepSize >= 0:
            #         topLeft.x -= stepSize
            #         bottomRight.x -= stepSize
            #         newConfig = True
            # elif key == ord('s'):
            #     if bottomRight.y + stepSize <= 1:
            #         topLeft.y += stepSize
            #         bottomRight.y += stepSize
            #         newConfig = True
            # elif key == ord('d'):
            #     if bottomRight.x + stepSize <= 1:
            #         topLeft.x += stepSize
            #         bottomRight.x += stepSize
            #         newConfig = True

            if points_storage:
                # xdiff = abs(bottomRight.x - topLeft.x)
                # ydiff = abs(topLeft.y - bottomRight.y)
                # topLeft.x = int(points_storage[0] - xdiff)
                # topLeft.y = int(points_storage[1] + ydiff)
                # bottomRight.x = int(points_storage[0] + xdiff)
                # bottomRight.y = int(points_storage[1] - ydiff)
                topLeft = dai.Point2f(points_storage["topLeft"][0], points_storage["topLeft"][1])
                bottomRight = dai.Point2f(points_storage["bottomRight"][0],points_storage["bottomRight"][1])
                newConfig = True
                print(points_storage)
            # print(topLeft.x, " ", topLeft.y)
            # print(bottomRight.x, " ", bottomRight.y)

            if newConfig:
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                q_spatial_calc_conf.send(cfg)
                newConfig = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd_m", default="models/palm_detection_6_shaves.blob", type=str,
                        help="Path to a blob file for palm detection model (default=%(default)s)")
    parser.add_argument("--lm_m", default="models/hand_landmark_6_shaves.blob", type=str,
                        help="Path to a blob file for landmark model (default=%(default)s)")
    args = parser.parse_args()

    ht = HandTracker(pd_path=args.pd_m, lm_path=args.lm_m)
    ht.run()

