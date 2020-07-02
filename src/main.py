from logging import getLogger
import hydra
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tracking
import factorization


logger = getLogger(__name__)


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger.info('\n' + str(cfg.pretty()))

    coord_list, frame_list = tracking.tracking(hydra.utils.to_absolute_path('videos/' + cfg.video), cfg.corner_count)

    logger.info(f"coord_list.shape={coord_list.shape}")
    logger.info(f"frame_list.shape={frame_list.shape}")

    logger.info("os.mkdir('tracked_images')")
    os.mkdir('tracked_images')
    logger.info("Save tracked images.")
    for i, frame in enumerate(frame_list):
        cv2.imwrite(f"tracked_images/{i}.png", frame)

    points = factorization.factorization(coord_list)

    logger.info("os.mkdir('3d_points_images')")
    os.mkdir('3d_points_images')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(*points, s=1, depthshade=False)
    plt.show()

    return


if __name__ == "__main__":
    main()
