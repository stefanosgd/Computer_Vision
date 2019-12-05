################################################################################

# This code: significantly based on code provided by Toby Breckon, toby.breckon@durham.ac.uk,
# taken from the files stereo_disparity.py and yolo.py, available at:
# https://github.com/tobybreckon/stereo-disparity
# https://github.com/tobybreckon/python-examples-cv

# To use first download the following files:

# https://pjreddie.com/media/files/yolov3.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true

################################################################################

import cv2
import argparse
import sys
import math
import numpy as np
import os

################################################################################

keep_processing = True

# parse command line arguments for camera ID or video file, and YOLO files
parser = argparse.ArgumentParser(
    description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')

args = parser.parse_args()


################################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, distance):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label. If no depth is provided, do not display a distance
    if not distance:
        label = '%s' % class_name
    else:
        label = '%s: %.2f m' % (class_name, distance)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.0 * labelSize[1])),
                  (left + round(0.75 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])
    # return post processed lists of classIds, confidences and bounding boxes
    return classIds_nms, confidences_nms, boxes_nms


################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


################################################################################

# init YOLO CNN object detection model

confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

################################################################################

# define display window name + create it

windowName = 'YOLOv3 object detection: ' + args.weights_file
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

# where is the data ? - set this to where you have it

master_path_to_dataset = "./TTBB-durham-02-10-17-sub10"  # edit this if needed
directory_to_cycle_left = "left-images"  # edit this if needed
directory_to_cycle_right = "right-images"  # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""  # set to timestamp to skip forward to
# 1506943412.479849
# 1506943260.487874
pause_playback = False  # pause until key press after each image

# camera information

baseline_distance = 0.2090607502
focal_length = 399.9745178222656

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left))

################################################################################

# parameters for the disparity map

window_size = 5

max_disparity = 512

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=max_disparity,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (1024, 780))

for filename_left in left_file_list:
    # from the left image filename get the corresponding right image

    if (len(skip_forward_file_pattern) > 0) and not (skip_forward_file_pattern in filename_left):
        continue
    elif (len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left):
        skip_forward_file_pattern = ""

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # start a timer (to see how long processing and display takes)
    start_t = cv2.getTickCount()

    # if video file successfully opened then read frame and crop the bonnet of the car out
    frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
    frame = frame[0:390, 0:frame.shape[1]]
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        # read left and right images
        # get left image from current frame and convert it to grayscale
        # read right image as a grayscale image and crop it to remove car bonnet

        imgL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_GRAYSCALE)
        imgR = imgR[0:390, 0:imgR.shape[1]]

        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # imgL = clahe.apply(imgL)
        # imgR = clahe.apply(imgR)

        displ = left_matcher.compute(imgL, imgR)
        dispr = right_matcher.compute(imgR, imgL)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        disparity = wls_filter.filter(displ, imgL, None, dispr)

        imgL2 = cv2.flip(imgL, 1)
        imgR2 = cv2.flip(imgR, 1)
        dispr = left_matcher.compute(imgR2, imgL2)
        displ = right_matcher.compute(imgL2, imgR2)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        disparity2 = cv2.flip(wls_filter.filter(dispr, imgL2, None, displ), 1)
        disparity = disparity[0:disparity.shape[0], disparity.shape[1] // 2:disparity.shape[1]]
        disparity2 = disparity2[0:disparity2.shape[0], 0:disparity2.shape[1] // 2]
        disparity = np.concatenate((disparity2, disparity), axis=1)
        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)

        disparity_scaled = (disparity / 16.).astype(np.uint8)
        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        # crop out the left side of the image where there is no disparity
        # width = np.size(disparity_scaled, 1)
        # disparity_scaled = disparity_scaled[0:390, 135:width]
        # frame = frame[0:390, 135:width]

        # rescale if specified
        if args.rescale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        small_frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])), interpolation=cv2.INTER_AREA)
        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(small_frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(small_frame, results, confThreshold, nmsThreshold)

        min_depth = math.inf
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            colour = [0, 0, 0]
            if classes[classIDs[detected_object]] == "car":
                colour = [0, 0, 255]
            elif classes[classIDs[detected_object]] == "bus":
                colour = [255, 0, 0]
            elif classes[classIDs[detected_object]] == "person":
                colour = [0, 255, 0]
            disparity_difference = np.amax(disparity_scaled[max(top, 0):min(top + height, disparity_scaled.shape[0]),
                                           max(left, 0):min(left + width, disparity_scaled.shape[1])])
            # disparity_difference = np.mean(disparity_scaled[max(top, 0):min(top + height, disparity_scaled.shape[0]),
            #                                max(left, 0):min(left + width, disparity_scaled.shape[1])])
            if disparity_difference != 0:
                depth = focal_length * baseline_distance / disparity_difference
                if depth < min_depth:
                    min_depth = depth
            else:
                depth = False

            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width,
                     top + height, colour, depth)

        print(filename_left)
        if min_depth == math.inf:
            min_depth = 0.0
        print(filename_right, "Nearest detected scene object (", min_depth, "m)")

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
        # timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        # disparity_scaled = cv2.equalizeHist(disparity_scaled)

        disparity_scaled = cv2.cvtColor(disparity_scaled, cv2.COLOR_GRAY2BGR)
        vis = np.concatenate((frame, disparity_scaled), axis=0)
        cv2.imshow(windowName, vis)
        out.write(vis)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & args.fullscreen)

    # stop the timer and convert to ms. (to see how long processing and display takes)
    stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000

    # start the event loop + detect specific key strokes
    # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
    key = cv2.waitKey(max(2 * (not pause_playback), (40 - int(math.ceil(stop_t))) * (not pause_playback))) & 0xFF
    # if user presses "x" then exit  / press "f" for fullscreen display
    if (key == ord('x')):
        break
    elif (key == ord('f')):
        args.fullscreen = not (args.fullscreen)
    elif (key == ord(' ')):  # pause (on next frame)
        pause_playback = not (pause_playback)

out.release()
# close all windows
cv2.destroyAllWindows()

# else:
# print("No video file specified or camera connected.")

################################################################################
