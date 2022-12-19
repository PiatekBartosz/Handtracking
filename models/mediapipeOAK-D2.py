import depthai as dai
import cv2
import time
import sys
from pathlib import Path
import mediapipe as mp
import mediapie_utils as mpu

# model paths
SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")

class HandTracker:

    def __init__(self, input_src=None,
                 pd_model=PALM_DETECTION_MODEL,
                 pd_score_thresh=0.5, pd_nms_thresh=0.3,
                 use_lm=True,
                 lm_model="lite",
                 lm_score_thresh=0.5,
                 use_world_landmarks=False,
                 solo=False,
                 xyz=False,
                 crop=False,
                 internal_fps=23,
                 resolution="full",
                 internal_frame_height=640,
                 use_gesture=False,
                 use_handedness_average=True,
                 single_hand_tolerance_thresh=10,
                 lm_nb_threads=2,
                 stats=False,
                 trace=0,
                 ):

        self.pd_model = pd_model
        print(f"Palm detection blob : {self.pd_model}")
        if use_lm:
            if lm_model == "full":
                self.lm_model = LANDMARK_MODEL_FULL
            elif lm_model == "lite":
                self.lm_model = LANDMARK_MODEL_LITE
            elif lm_model == "sparse":
                self.lm_model = LANDMARK_MODEL_SPARSE
            else:
                self.lm_model = lm_model
            print(f"Landmark blob       : {self.lm_model}")
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        if not use_lm and solo:
            print("Warning: solo mode desactivated when not using landmarks")
            self.solo = False
        else:
            self.solo = solo
        if self.solo:
            print("In Solo mode, # of landmark model threads is forced to 1")
            self.lm_nb_threads = 1
        else:
            assert lm_nb_threads in [1, 2]
            self.lm_nb_threads = lm_nb_threads
        if self.use_lm:
            self.max_hands = 1 if self.solo else 2
        else:
            self.max_hands = 20
        self.xyz = False
        self.crop = crop
        self.use_world_landmarks = use_world_landmarks
        self.internal_fps = internal_fps
        self.stats = stats
        self.trace = trace
        self.use_gesture = use_gesture
        self.use_handedness_average = use_handedness_average
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh

        self.device = dai.Device()

        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frames are systematically transferred to the host
            self.input_type = "rgb"  # OAK* internal color camera
            self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            self.video_fps = self.internal_fps  # Used when saving the output in a video file. Should be close to the real fps

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png'):
            self.input_type = "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)

        if self.input_type != "rgb":
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h)
            else:
                self.frame_size = max(self.img_w, self.img_h)
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)

            if self.crop: self.img_h = self.img_w = self.frame_size
            print(f"Frame working size: {self.img_w}x{self.img_h}")

        # Create SSD anchors
        self.pd_input_length = 128  # Palm detection
        # self.pd_input_length = 192 # Palm detection
        self.anchors = mpu.generate_handtracker_anchors(self.pd_input_length, self.pd_input_length)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")

        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")

        self.fps = FPS()

        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0
        self.nb_spatial_requests = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.glob_spatial_rtrip_time = 0

        self.use_previous_landmarks = False
        self.nb_hands_in_previous_frame = 0
        if not self.solo: self.single_hand_count = 0

        if use_handedness_average:
            # handedness_avg: for more robustness, instead of using the last inferred handedness, we prefer to use the average
            # of the inferred handedness since use_previous_landmarks is True.
            self.handedness_avg = [mpu.HandednessAverage() for i in range(self.max_hands)]