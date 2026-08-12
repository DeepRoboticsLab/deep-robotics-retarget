"""Configuration parameters for supported robots and human motion sources.

This module defines the paths to robot URDF/XML models, IK configuration files,
and viewer camera settings for each supported robot-human pair.
"""

import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "DR02_pro": ASSET_ROOT / "DR02" / "DR02_pro.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "DR02_pro": IK_CONFIG_ROOT / "smplx_to_DR02_pro.json",
    },
    "bvh_lafan1":{
        "DR02_pro": IK_CONFIG_ROOT / "bvh_lafan1_to_DR02_pro.json",
    },
    "bvh_nokov":{
        "DR02_pro": IK_CONFIG_ROOT / "bvh_nokov_to_DR02_pro.json",
    },
}


ROBOT_BASE_DICT = {
    "DR02_pro": "base_link",
}

VIEWER_CAM_DISTANCE_DICT = {
    "DR02_pro": 2.5,
}
