import argparse
import os

import torch
import yaml
from easydict import EasyDict as edict


def prep_env_tsf():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="config.yaml")
    args = parser.parse_args()

    local_path = os.path.split(os.path.realpath(__file__))[0]
    settings = edict(yaml.load(open(os.path.join(local_path, args.conf)), Loader=yaml.FullLoader))

    # Prepare the device
    if torch.cuda.is_available():
        settings["use_gpu"] = True
        torch.cuda.set_device('cuda:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        device = torch.device('cpu')

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings


if __name__ == "__main__":
    config = prep_env_tsf()
