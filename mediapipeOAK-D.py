import depthai as dai
import cv2
import time
import sys
from pathlib import Path
import mediapipe as mp

#todo consider making functions
SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")

print("Creating pipeline...")
pipeline = dai.Pipeline()
# todo figure out the openvino version
# pipeline.setOpenVINO.Version()

# Define source
color = pipeline.createColorCamera()
# monoL = pipeline.createMonoCamera()
# monoP = pipeline.createMonoCamera()
# stereo = pipeline.createStereoDepth()
# stereo.setLeftRightCheck(True)  # check for occlusion of left, right mono
# resize = pipeline.create(dai.node.ImageManip)
# pd_nn = pipeline.createNeuralNetwork()

# Define output
# Video stream
xoutVideo = pipeline.createXLinkOut()
# Hand detection (open/closed + location)
# xoutPD = pipeline.createXLinkOut()
# Used for depth perception
# xoutRectL = pipeline.createXLinkOut()
# xoutDis = pipeline.createXLinkOut()
# xoutRectR = pipeline.createXLinkOut()


# Name streams
xoutVideo.setStreamName("video")
# xoutPD.setStreamName("palm detection")
# xoutRectL.setStreamName("rectified left")
# xoutDis.setStreamName("disparity")
# xoutRectR.setStreamName("rectified right")

# Properties
color.setPreviewSize(416, 416)
color.setBoardSocket(dai.CameraBoardSocket.RGB)
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setInterleaved(True)
color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
# monoP.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
# resize.setResize(300, 300)

# Neural Network properties
# pd_nn.setBlobPath(PALM_DETECTION_MODEL)

# Linking depth
# monoL.out.link(stereo.left)
# monoP.out.link(stereo.right)
# xoutRectL = stereo.rectifiedLeft
# xoutRectRstereo.rectifiedRight
# xoutDis.link(stereo.disparity)
# Linking RGB
color.video.link(xoutVideo.input)
# color.video.link(resize)

# resize.out.link(pd_nn)
# pd_nn.out(xoutPD)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue('video')

    while True:
        videoFrame = video.get()

        # Get BGR frame from NV12 encoded video frame to show with opencv
        cv2.imshow("video", videoFrame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
