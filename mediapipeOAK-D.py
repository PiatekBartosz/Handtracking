import depthai as dai
import cv2
import time
import sys
from pathlib import Path
from utils import postprocess

#todo consider making functions
SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
PD_INPUT_SIZE = 128

print("Creating pipeline...")
pipeline = dai.Pipeline()
# todo figure out the openvino version
# pipeline.setOpenVINO.Version()

# Define source
color = pipeline.createColorCamera()
monoL = pipeline.createMonoCamera()
monoR = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)  # check for occlusion of left, right mono
resize = pipeline.create(dai.node.ImageManip)
pd_nn = pipeline.createNeuralNetwork()

# Define output

# Video stream
xoutVideo = pipeline.createXLinkOut()
# Hand detection (open/closed + location)
xoutPD = pipeline.createXLinkOut()
# Used for depth perception
xoutDisparity = pipeline.createXLinkOut()


# Name streams
xoutVideo.setStreamName("video")
xoutPD.setStreamName("palm detection")
xoutDisparity.setStreamName("disparity")

# Properties
color.setPreviewSize(416, 416)
color.setBoardSocket(dai.CameraBoardSocket.RGB)
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setInterleaved(True)
color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
color.setPreviewSize(PD_INPUT_SIZE, PD_INPUT_SIZE)
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# resize.setResize(300, 300)

# Neural Network properties
pd_nn.setBlobPath(PALM_DETECTION_MODEL)

# nn_setup
anchors, anchors_count = postprocess.create_SSD_anchors(PD_INPUT_SIZE)

# Linking depth
monoL.out.link(stereo.left)
monoR.out.link(stereo.right)
stereo.disparity.link(xoutDisparity.input)

# Link nn
color.preview.link(pd_nn.input)
pd_nn.out.link(xoutPD.input)


# Linking RGB
color.video.link(xoutVideo.input)
# color.video.link(resize)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue('video')
    disparity = device.getOutputQueue('disparity')
    pd = device.getOutputQueue('palm detection')

    while True:
        videoFrame = video.get()
        disparityFrame = disparity.get()
        pd_output = pd.get()
        hands = postprocess.pd_postprocess(pd_output, anchors, anchors_count, PD_INPUT_SIZE)
        print(hands)
        # print(pd_output)

        # Get BGR frame from NV12 encoded video frame to show with opencv
        cv2.imshow("video", videoFrame.getCvFrame())
        cv2.imshow("disparity", disparityFrame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
