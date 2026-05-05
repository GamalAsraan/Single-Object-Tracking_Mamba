"""
lib.train.dataset

Dataset registry. AerialTracking is the only dataset actually used by the
Kaggle MTC-AIC4 port; the upstream LaSOT / Got10k / TrackingNet / COCO /
ImagenetVID modules (and their lmdb variants) are imported defensively so
that a stale import in any one of them does not break the training pipeline.
"""

import warnings as _warnings

# --- AerialTracking: REQUIRED. Let any failure here propagate loudly. -------
from .aerial_tracking import AerialTracking

# --- Legacy upstream datasets: best-effort. Track failures for debugging. ---
_legacy_import_errors = {}


def _try_import(modname, attrname):
    try:
        mod = __import__(f"{__name__}.{modname}", fromlist=[attrname])
        globals()[attrname] = getattr(mod, attrname)
    except Exception as e:  # ImportError, FutureWarning-as-error, etc.
        _legacy_import_errors[attrname] = f"{type(e).__name__}: {e}"


_try_import("lasot",              "Lasot")
_try_import("got10k",             "Got10k")
_try_import("tracking_net",       "TrackingNet")
_try_import("imagenetvid",        "ImagenetVID")
_try_import("coco",               "MSCOCO")
_try_import("coco_seq",           "MSCOCOSeq")
_try_import("got10k_lmdb",        "Got10k_lmdb")
_try_import("lasot_lmdb",         "Lasot_lmdb")
_try_import("imagenetvid_lmdb",   "ImagenetVID_lmdb")
_try_import("coco_seq_lmdb",      "MSCOCOSeq_lmdb")
_try_import("tracking_net_lmdb",  "TrackingNet_lmdb")

if _legacy_import_errors:
    _warnings.warn(
        "lib.train.dataset: {} legacy dataset(s) failed to import "
        "(non-fatal; AerialTracking is the only one used by MTC-AIC4): {}".format(
            len(_legacy_import_errors),
            ", ".join(_legacy_import_errors.keys()),
        ),
        stacklevel=2,
    )