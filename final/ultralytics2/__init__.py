# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.19"

from ultralytics2.data.explorer.explorer import Explorer
from ultralytics2.models import RTDETR, SAM, YOLO, YOLOWorld
from ultralytics2.models.fastsam import FastSAM
from ultralytics2.models.nas import NAS
from ultralytics2.utils import ASSETS, SETTINGS
from ultralytics2.utils.checks import check_yolo as checks
from ultralytics2.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)