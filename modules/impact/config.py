import os

from comfy.cli_args import args
from ..impact_config.add_configuration import ImpactPackConfiguration

cached_config: ImpactPackConfiguration = args

version_code = [7, 9]
version = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')

dependency_version = 23

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")


def write_config():
    pass


def read_config():
    global cached_config
    return cached_config


def get_config():
    global cached_config
    return cached_config
