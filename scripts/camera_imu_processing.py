#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image, Imu # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from scipy.linalg import expm
import pickle

# Determine whether to use the IMU in tracking
IMU_TRACKING = True

# Whether to show rotated image (default false)
SHOW_ROTATED_IMAGE = False

# Show feature tracks?
SHOW_TRACKS = True

SHOW_IMAGE = True

# Parameters for the goodFeaturesToTrack function
feature_params = dict(maxCorners=0,
                      qualityLevel=0.03,
                      minDistance=10,
                      blockSize=3)

# Parameters for the LK Optic Flow function
lk_params=dict(winSize=(15,15),
               maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Line colors
colors = np.array([[0, 0, 63], [0, 255, 0]])

mask = None

if IMU_TRACKING:
    #grab the calibration matrix from the file
    calib_result_pickle = pickle.load(open("camera_calib_pickle_scale_larger_res.p", "rb"))
    K = calib_result_pickle["mtx"]
    Kinv = np.linalg.inv(K)

# Refresh rate of gyros (Hz)
gyro_refresh_rate = 200

# Define initial rotation matrix as identity to start
rotation_matrix = np.identity(3)

# List of features to track
p0 = None
old_gray = None

first_features = None
first_image = None

num_frames = 0
told_of_frames = False
inliers_calculated = False
delay = 17.5 #seconds between initial and final frames
highest = 0.0

first_stamp = None

# Convert gyro data to Omega cross
def gyro_data(data):
    global highest
    if data.z > highest:
        highest = data.z
    return np.asarray([[0., -data.z, data.y],
                       [data.z, 0., -data.x],
                       [-data.y, data.x, 0.]])


def callback_imu_processing(data):
    global rotation_matrix, told_of_frames

    # Output debugging info to the terminal
    if not told_of_frames:
        rospy.loginfo("receiving imu data")
        told_of_frames = True

    # Get omegra cross
    gyro = gyro_data(data.angular_velocity)

    # Get rotation matrix from one gyro data point to the next
    R_p_2_p = expm(gyro*1./gyro_refresh_rate).T

    # Get the new rotation matrix by multiplying it by the point-2-point matrix
    rotation_matrix = R_p_2_p @ rotation_matrix

 
def callback_image_processing(data):
    global rotation_matrix, p0, old_gray, mask, num_frames, first_features, first_image, inliers_calculated, highest, first_stamp

    if first_stamp is None:
        first_stamp = data.header.stamp

    if data.header.stamp - first_stamp >= rospy.Duration(delay) and not inliers_calculated:
        if inliers_calculated:
            return
        # calculator = cv2.xfeatures2d.SIFT_create()
        #
        # # test if mask can be applied without affecting scale
        # keypoint_mask = np.zeros((540, 960, 1), dtype=np.uint8)
        # for i, (point) in enumerate(p0):
        #     a, b = np.ravel(point)
        #     keypoint_mask = cv2.circle(keypoint_mask, (int(a), int(b)), 2, 1, -1)
        #
        # keypoints = calculator.detect(old_gray, keypoint_mask)

        rospy.loginfo("Points tracked: " + str(p0.shape[0]))
        distances = np.linalg.norm(first_features - p0, axis=(1,2)).reshape((1,-1))
        highest = round(highest * 2) / 2
        imu_str = "" if not IMU_TRACKING else "_imu"
        np.savetxt("tracking_data_" + str(highest) + imu_str + ".csv", distances, delimiter=",")

        # df = pd.read_csv("tracking_data.csv")
        # df[str(highest)] = distances
        # H, point_mask = cv2.findHomography(p0, first_features, method=cv2.LMEDS, ransacReprojThreshold=10)
        # inliers = p0[point_mask==1]
        # rospy.loginfo("Inliers: " +str(inliers.shape[0]))
        # if SHOW_IMAGE:
        #     new_first = cv2.warpPerspective(first_image, H, (first_image.shape[1], first_image.shape[0]))
        #     cv2.imshow("first frame", new_first)
        #     cv2.waitKey(0)

        # descriptors idea
        # kp_initial, descriptors_initial = calculator.compute(first_image, first_features)
        # kp_final, descriptors_final = calculator.compute(old_gray, p0)
        inliers_calculated = True
        rospy.signal_shutdown("Finished Tracking")
        return
    num_frames += 1

    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    # rospy.loginfo("receiving video frame")

    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data, desired_encoding="bgr8")

    # Convert the frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Check if we have any frames (or later if we have enough frames)
    if p0 is None:
        first_image = np.copy(current_gray)
        p0 = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)
        first_features = np.copy(p0)
        original_num_features = p0.shape[0]
        rospy.loginfo(original_num_features)

        # Reset rotation_matrix to identity
        rotation_matrix = np.identity(3)

        # Keep track of old gray frame
        old_gray = current_gray.copy()

        # Set up the mask
        mask = np.zeros_like(current_frame)

        # Skip the rest of the function, no need to do anything with it.
        return

    if IMU_TRACKING:
        H = K @ rotation_matrix @ Kinv
        H /= H[2, 2]
        p0 = cv2.perspectiveTransform(p0, H)
        old_gray = cv2.warpPerspective(old_gray, H, (old_gray.shape[1], old_gray.shape[0]))
        if SHOW_ROTATED_IMAGE:
            cv2.imshow("old_frame", old_gray)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        temp = np.copy(first_features)
        first_features = temp[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[0].tolist(), 1)
        current_frame = cv2.circle(current_frame, (int(a), int(b)), 2, colors[1].tolist(), -1)
    if SHOW_TRACKS:
        current_frame = cv2.add(current_frame, mask)


    # Display image
    if SHOW_IMAGE:
        cv2.imshow("camera", current_frame)

    cv2.waitKey(1)

    # Keep track of old gray frame
    old_gray = current_gray.copy()

    # Reset rotation_matrix to identity
    rotation_matrix = np.identity(3)

    # Only keep the good features for the next iteration
    p0 = good_new.reshape(-1, 1, 2)
    first_features = first_features.reshape(-1, 1, 2)
      
def receive_message():
 
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('camera_imu_processing_py', anonymous=True)

    # Node is subscribing to the image topic
    rospy.Subscriber('/camera/color/image_raw', Image, callback_image_processing)

   # Node is subscribing to the imu topic
    rospy.Subscriber("/camera/gyro/sample", Imu, callback_imu_processing)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    # Close down the video stream when done
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
    receive_message()
