"""Create the detection model used by the demo and figure analyses."""


def DetectionModel(**kwargs):
    """Load the supported detection architecture with the supplied configuration."""
    from rfdetr import RFDETRLarge

    return RFDETRLarge(**kwargs)


__all__ = ['DetectionModel']
