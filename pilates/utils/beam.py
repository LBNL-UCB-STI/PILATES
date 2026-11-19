from pathlib import Path
from pilates.utils.settings_helper import get as get_setting


def get_beam_omx_skims_name(settings, default: str = "skims.omx") -> str:
    """
    Resolve the downstream OMX skim filename for BEAM outputs.

    ``shared.skims.fname`` may point at a canonical Zarr skim store. Downstream
    OMX consumers still need a sibling ``.omx`` export, so normalize any
    non-OMX filename to the same basename with an ``.omx`` suffix.
    """
    configured = get_setting(settings, "shared.skims.fname", default)
    if not isinstance(configured, str) or not configured:
        return default
    if configured.endswith(".omx"):
        return configured
    return str(Path(configured).with_suffix(".omx"))
