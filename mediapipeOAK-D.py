import depthai as dai
import cv2
import time
import sys

#todo consider making functions
print("Creating pipeline...")
pipeline = dai.Pipeline()

# Define source
color = pipeline.createColorCamera()
mono = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)  # check for occlusion of left, right mono

# Define output
# Video stream
xoutVideo = pipeline.createXLinkOut()
# Hand detection (open/closed + location)
xoutDetection = pipeline.createXLinkOut()
# Used for depth perception
xoutRectL = pipeline.createXLinkOut()
xoutDis = pipeline.createXLinkOut()
xoutRectR = pipeline.createXLinkOut()

# Name streams
xoutVideo.setStreamName("video")
xoutDetection.setStreamName("detection")
xoutRectL.setStreamName("rectified left")
xoutDis.setStreamName("disparity")
xoutRectR.setStreamName("rectified right")

# Properties
color.setPreviewSize(300, 300)
color.setBoardSocket(dai.CameraBoardSocket.RGB)
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setInterleaved(True)
color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)

# Linking
color.video.link(xoutVideo.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue('video')

    while True:
        videoFrame = video.get()

        # Get BGR frame from NV12 encoded video frame to show with opencv
        cv2.imshow("video", videoFrame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
