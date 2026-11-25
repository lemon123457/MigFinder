"""Frame-to-frame tracking."""
from .bot_sort_lite import BoTSORTLite, STrack, TrackingConfig
Tracker = BoTSORTLite
__all__ = ['Tracker', 'BoTSORTLite', 'STrack', 'TrackingConfig']
