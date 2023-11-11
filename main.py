from __future__ import print_function
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='output.h264')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

counter = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    img = cv2.imread('target.png')


    counter += 1
    frame = cv2.blur(frame, (7, 7))
    fgMask = backSub.apply(frame)

    threshold = 200
    ret, T = cv2.threshold(fgMask, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    T = cv2.erode(T, kernel, iterations=1)

    ## opening
    kernel = np.ones((7, 7), np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_OPEN, kernel)

    ## closing
    kernel = np.ones((23, 23), np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel)

    n, C, stats, centroids = cv2.connectedComponentsWithStats(T)

    p1 = (22, 148)
    p2 = (1225, 86)
    p3 = (1138, 117)
    p4 = (873, 780)

    points1 = np.array([p1, p2, p3, p4], dtype=np.float32)

    p11 = (5, 5)
    p22 = (1049, 5)
    p33 = (890, 152)
    p44 = (525, 655)

    points2 = np.array([p11, p22, p33, p44], dtype=np.float32)

    u = 1055
    m = 705
    output_size = (u, m)

    H = cv2.getPerspectiveTransform(points1, points2)
    J = cv2.warpPerspective(frame, H, output_size)

    if counter >= 60:
        for k in range(n):
            if k == 0:
                continue
            # show the k-th connected component
            Ck = np.zeros(T.shape, dtype=T.dtype)
            Ck[C == k] = 255
            x1=int(centroids[k][0])-14
            y1=int(centroids[k][1])-14
            x2=int(centroids[k][0])+14
            y2=int(centroids[k][1])+14
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 255), 3)
            sample = frame[y1:y2, x1:x2]
            pts = np.float32([[centroids[k][0], centroids[k][1]]]).reshape(-1, 1, 2)
            J = cv2.perspectiveTransform(pts, H)

            # batch = np.expand_dims(sample, axis=0)
            # prediction = model.predict(batch)

            prediction=1
            if np.argmax(prediction)==1:
                cv2.circle(img, (int(J[0][0][0]),int(J[0][0][1])), 5, [0,0,0],2)
            else:
                cv2.circle(img, (int(J[0][0][0]),int(J[0][0][1])), 5, [255,0,0],2)


        cv2.imshow('Img', img);

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    # cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
