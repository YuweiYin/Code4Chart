import os
import sys
import json
import time
import fire
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code4chart.init_functions import logger_setup


def main(
    verbose: bool = False,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the whole data analysis pipeline.

    :param verbose: Verbose mode: show logs.
    :param seed: Random seed of all modules.
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param debug: Debugging / developing mode.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Code4Chart")

    # ##### Vis Code Start #####
    data = pd.read_csv("../data/code4chart/raw/New_York_City_Airbnb_Open_Data_2019.csv")
    column = data["latitude"].tolist()

    plt.hist(column, bins=50, color='skyblue', edgecolor='black')
    # 'C0' default blue color
    # 'C1' second default orange color
    # 'b' as blue
    # 'g' as green
    # 'r' as red
    # 'c' as cyan
    # 'm' as magenta
    # 'y' as yellow
    # 'k' as black
    # 'orange' as orange
    # 'w' as white
    plt.title('Distribution of Latitude Values in NYC Airbnb Open Data 2019')
    plt.xlabel('Latitude')
    plt.ylabel('Frequency')
    # plt.savefig('NYCAirbnb_LatitudeDistribution.png')
    plt.show()

    # ##### Vis Code End #####

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
