from logging import getLogger
import os
import numpy as np
import cv2
import copy


logger = getLogger(__name__)


def tracking(filepath, corner_count):
    video = cv2.VideoCapture(filepath)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    logger.info(f"width={width}")
    logger.info(f"height={height}")
    logger.info(f"frame_num={frame_num}")

    log = np.empty([frame_num, corner_count, 2])
    frame_list = []
    once_flag = True
    for count in range(frame_num):
        _, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(once_flag):
            once_flag = False

            corners1 = cv2.goodFeaturesToTrack(gray_frame, corner_count, 0.000001,
                15, mask=None, blockSize=3, useHarrisDetector=1, k=0.01)
            corners1 = cv2.cornerSubPix(gray_frame, corners1, (3, 3), (-1, -1),
                (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.03))
            corners2, status, error = cv2.calcOpticalFlowPyrLK(gray_frame,
                gray_frame, corners1, None, winSize=(10, 10), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 64, 0.01))
        else:
            for i in range(corner_count):
                if(status[i] == 0):
                    corners1[i, 0, [0, 1]] = -100
                else:
                    corners1[i] = corners2[i]
            corners2, status, error = cv2.calcOpticalFlowPyrLK(pre_gray_frame,
                gray_frame, corners1, corners2, status, error, (10, 10), 4,
                (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 64, 0.01),
                cv2.OPTFLOW_USE_INITIAL_FLOW)
        pre_gray_frame = gray_frame

        for i in range(corner_count):
            log[count][i] = corners2[i, 0]
        for i in corners1:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, [0x00]*0, -1)
            cv2.circle(frame, (x, y), 1, [0xff]*3, -1)
        cv2.imshow('corner', frame)
        frame_list.append(frame)
        cv2.waitKey(round(1000 / fps))
    video.release()
    cv2.destroyAllWindows()
    frame_list = np.array(frame_list)

    coord_list = []
    for i in range(frame_num):
        coords = np.empty([0, 2], dtype=np.float32)
        for j in range(corner_count):
            if(status[j] == 1):
                coords = np.vstack([coords, log[i, j]])
        coord_list += [coords]
    coord_list = np.array(coord_list)

    return coord_list, frame_list
