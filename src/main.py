from logging import getLogger
import hydra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tracking
import factorization


logger = getLogger(__name__)


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger.info('\n' + str(cfg.pretty()))

    coord_list = tracking.tracking(hydra.utils.to_absolute_path('videos/' + cfg.video), cfg.corner_count)
    logger.info(f"coord_list.shape={coord_list.shape}")
    points = factorization.factorization(coord_list)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(*points, s=1, depthshade=False)
    plt.show()

    return


if __name__ == "__main__":
    main()
