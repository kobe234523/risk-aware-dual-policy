"""Select an explicit risk backend; the learned route has no geometry fallback."""
import os
import numpy as np
from .model import predict_risk

RISK_METHODS = {"risk_switch", "risk_fusion", "ours", "ours_rs"}


def validate_options(args):
    if not 0 <= args.delta <= 1:
        raise ValueError("--delta must lie in [0, 1]")
    if args.method in RISK_METHODS and args.risk_estimator == "learned":
        if not args.risk_head_ckpt or not os.path.isfile(args.risk_head_ckpt):
            raise ValueError("Learned risk requires --risk_head_ckpt pointing to a trained head; use --risk_estimator legacy only for the old privileged baseline")
    if args.collect_risk_data and not str(args.risk_scene_group).strip():
        raise ValueError("Collection requires --risk_scene_group identifying route + layout, shared across seeds/weather")


def compute_online_risk(mode, head=None, features=None, privileged_reference=None, legacy=None):
    info = dict(rho=0.0, ttc=-1.0, risk_longitudinal=0.0, risk_lateral=0.0,
                risk_distance=-1.0, risk_backend=mode, uses_privileged_risk=False)
    if mode == "learned":
        if head is None or features is None:
            raise ValueError("Learned risk needs an initialized checkpoint and observable features")
        info["rho"] = predict_risk(head, features)
    elif mode == "reference":
        if privileged_reference is None or not np.isfinite(privileged_reference):
            raise ValueError("Reference gating needs a valid privileged reference")
        info.update(rho=float(privileged_reference), uses_privileged_risk=True)
    elif mode == "legacy":
        if legacy is None:
            raise ValueError("Legacy baseline callback missing")
        info.update(legacy())
        if info.get("risk_error"):
            raise RuntimeError("Legacy risk failed: {}".format(info["risk_error"]))
        info.update(risk_backend="legacy", uses_privileged_risk=True)
    elif mode != "none":
        raise ValueError("Unknown risk backend: {}".format(mode))
    if not np.isfinite(info["rho"]) or not 0 <= info["rho"] <= 1:
        raise ValueError("Invalid online risk")
    return info
