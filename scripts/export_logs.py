import argparse
import csv
import math
import glob
import os


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_run(experiment:str) -> str:
    """
    Get the latest run directory for a given experiment.

    Args:
        experiment (str): The name of the experiment.
        """
    runs = 