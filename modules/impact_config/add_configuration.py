import os

import configargparse

from comfy.cli_args_types import Configuration


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--dependency-version",
                        type=int,
                        default=0,
                        help="Dependency version for Impact Pack",
                        env_var="IMPACT_PACK_DEPENDENCY_VERSION")
    parser.add_argument("--mmdet-skip",
                        action="store_true",
                        default=True,
                        help="Skip MMDet initialization",
                        env_var="IMPACT_PACK_MMDET_SKIP")
    parser.add_argument("--sam-editor-cpu",
                        action="store_true",
                        default=False,
                        help="Use CPU for SAM editor",
                        env_var="IMPACT_PACK_SAM_EDITOR_CPU")
    parser.add_argument("--sam-editor-model",
                        type=str,
                        default="sam_vit_b_01ec64.pth",
                        help="SAM editor model file",
                        env_var="IMPACT_PACK_SAM_EDITOR_MODEL")
    parser.add_argument("--custom-wildcards",
                        type=str,
                        default=os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
                        help="Path to custom wildcards directory",
                        env_var="IMPACT_PACK_CUSTOM_WILDCARDS")
    parser.add_argument("--disable-gpu-opencv",
                        action="store_true",
                        default=True,
                        help="Disable GPU acceleration for OpenCV",
                        env_var="IMPACT_PACK_DISABLE_GPU_OPENCV")
    return parser


class ImpactPackConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.dependency_version: int = 0
        self.mmdet_skip: bool = True
        self.sam_editor_cpu: bool = False
        self.sam_editor_model: str = "sam_vit_b_01ec64.pth"
        self.custom_wildcards: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards"))
        self.disable_gpu_opencv: bool = True
