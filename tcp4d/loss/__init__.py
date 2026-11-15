"""Training losses for TCP4D."""

from .geometry import GeometryLoss
from .tcp_tracking import TCPTrackingLoss

__all__ = ["GeometryLoss", "TCPTrackingLoss"]
