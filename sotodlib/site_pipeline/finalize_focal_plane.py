import os
import sys
import argparse as ap
import h5py
import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util
from sotodlib.coords import optics as op

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _get_db(ctx, name):
    db = None
    for meta in ctx["metadata"]:
        if "name" not in meta:
            continue
        if meta["name"] == "position_match":
            db = meta["db"]
            break
    if db is None:
        raise ValueError(f"Context does not contain {name}")
    if db.startswith("./"):
        db = os.path.join(os.path.dirname(ctx.filename), db[2:])
    return metadata.ManifestDb(db)


def _encs_notclose(az, el, bs):
    return not (
        np.isclose(az, az[0], equal_nan=True).all()
        and np.isclose(el, el[0], equal_nan=True).all()
        and np.isclose(bs, bs[0], equal_nan=True).all()
    )


def _avg_focalplane(fp_dict):
    focal_plane = []
    det_ids = np.array(list(fp_dict.keys()))
    for did in det_ids:
        avg_pointing = np.nanmedian(np.vstack(fp_dict[did]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.column_stack(focal_plane)

    if np.isnan(focal_plane[:2]).all():
        raise ValueError("All detectors are outliers. Check your inputs")

    return det_ids, focal_plane


def _log_vals(shift, scale, shear, rot):
    axis = ["xi", "eta", "gamma"]
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    for ax, s in zip(axis, shift):
        logger.info("Shift along %s axis is %f", ax, s)
    for ax, s in zip(axis, scale):
        logger.info("Scale along %s axis is %f", ax, s)
        if np.isclose(s, deg2rad):
            logger.warning(
                "Scale factor for %s looks like a degrees to radians conversion", ax
            )
        elif np.isclose(s, rad2deg):
            logger.warning(
                "Scale factor for %s looks like a radians to degrees conversion", ax
            )
    logger.info("Shear param is %f", shear)
    logger.info("Rotation of the xi-eta plane is %f radians", rot)


def _mk_fpout(det_id, focal_plane):
    outdt = [
        ("dets:det_id", det_id.dtype),
        ("xi", np.float32),
        ("eta", np.float32),
        ("gamma", np.float32),
    ]
    fpout = np.fromiter(zip(det_id, *focal_plane[:3]), dtype=outdt, count=len(det_id))

    return metadata.ResultSet.from_friend(fpout)


def _mk_tpout(shift, scale, shear, rot):
    outdt = [
        ("d_xi", np.float32),
        ("d_eta", np.float32),
        ("d_gamma", np.float32),
        ("s_xi", np.float32),
        ("s_eta", np.float32),
        ("s_gamma", np.float32),
        ("shear", np.float32),
        ("rot", np.float32),
    ]
    row = shift + scale + (shear, rot)
    tpout = np.array(row, outdt)

    return tpout


def _mk_refout(lever_arm, encoders):
    outdt = [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
    ]
    refout = np.array([lever_arm, tuple(encoders)], outdt)

    return refout


def _add_attrs(dset, attrs):
    for k, v in attrs.items():
        dset.attrs[k] = v


def _mk_plot(nominal, measured, affine, shift, show_or_save):
    plt.style.use("tableau-colorblind10")
    _, ax = plt.subplots()
    ax.set_xlabel("Xi Nominal (rad)")
    ax.set_ylabel("Eta Nominal (rad)")
    p1 = ax.scatter(nominal[0], nominal[1], label="nominal", color="grey")
    ax1 = ax.twinx()
    ax1.set_ylabel("Eta Measured (rad)")
    ax2 = ax1.twiny()
    ax2.set_xlabel("Xi Measured (rad)")
    p2 = ax2.scatter(measured[0], measured[1], label="measured")
    transformed = affine @ nominal + shift[:, None]
    p3 = ax2.scatter(transformed[0], transformed[1], label="transformed")
    ax2.legend(handles=[p1, p2, p3])
    if isinstance(show_or_save, str):
        plt.savefig(show_or_save)
        plt.cla()
    else:
        plt.show()


def get_nominal(focal_plane, config, encoders):
    """
    Get nominal pointing from detector xy positions.

    Arguments:

        focal_plane: Focal plane array as generated by _avg_focalplane.

        config: Transformation configuration.
                Nominally config["coord_transform"].

        encoders: Encoder values to compute LOS rotation from.

    Returns:

        xi_nominal: The nominal xi values.

        eta_nominal: The nominal eta values.

        gamma_nominal: The nominal gamma values.
    """
    transform_pars = op.get_ufm_to_fp_pars(
        config["telescope"], config["slot"], config["config_path"]
    )
    x, y, pol = op.ufm_to_fp(
        None, x=focal_plane[3], y=focal_plane[4], pol=focal_plane[5], **transform_pars
    )
    if config["telescope"] == "LAT":
        rot = np.nan_to_num(np.rad2deg(encoders[1]) - 60 - np.rad2deg(encoders[2]))
        xi_nominal, eta_nominal, gamma_nominal = op.LAT_focal_plane(
            None, config["zemax_path"], x, y, pol, rot, config["tube"]
        )
    elif config["coord_transform"]["telescope"] == "SAT":
        rot = np.nan_to_num(-1.0 * np.rad2deg(encoders[2]))
        xi_nominal, eta_nominal, gamma_nominal = op.SAT_focal_plane(
            None, x, y, pol, rot
        )
    else:
        raise ValueError("Invalid telescope provided")

    return xi_nominal, eta_nominal, gamma_nominal


def gamma_fit(src, dst):
    """
    Fit the transformation for gamma.
    Note that the periodicity here assumes things are in radians.

    Arguments:

        src: Source gamma in radians

        dst: Destination gamma in radians

    Returns:

       scale: Scale applied to src

       shift: Shift applied to scale*src
    """

    def _gamma_min(scale, shift, gamma):
        src, dst = gamma
        transformed = np.sin(src * scale + shift)
        diff = np.sin(dst) - transformed

        return np.sqrt(np.mean(diff**2))

    res = minimize(_gamma_min, (1.0, 0.0), (src, dst))
    return res.x


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load context
    ctx = Context(config["context"]["path"])
    name = config["context"]["position_match"]
    db = _get_db(ctx, name)
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    # Add in manually loaded paths
    obs_ids = np.append(obs_ids, config.get("multi_obs", []))
    if len(obs_ids) == 0:
        raise ValueError("No position match results provided in configuration")
    detmaps = config["detmaps"]
    if len(obs_ids) != len(detmaps):
        raise ValueError(
            "Number of DetMaps doesn't match number of position match results"
        )

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    os.makedirs(config["outdir"], exist_ok=True)
    outpath = os.path.join(config["outdir"], f"{ufm}{append}.h5")
    outpath = os.path.abspath(outpath)

    fp_dict = {}
    encoders = []
    use_matched = config.get("use_matched", False)
    for obs_id, detmap in zip(obs_ids, detmaps):
        # Load data
        if os.path.isfile(obs_id):
            logger.info("Loading information from file at %s", obs_id)
            rset = read_dataset(obs_id, "focal_plane")
            _aman = rset.to_axismanager(axis_key="dets:readout_id")
            aman = AxisManager(_aman.dets)
            aman.wrap(name, _aman)
            encs = read_dataset(obs_id, "encoders")
        else:
            logger.info("Loading information from observation %s", obs_id)
            aman = ctx.get_meta(obs_id, dets=config["context"].get("dets", {}))
            db_match = db.match({"obs:obs_id": obs_id})
            if db_match:
                encs = read_dataset(db_match["filename"], "encoders")
            else:
                logger.warning("\tMissing encoder information, nans will be used")
                encs = None
        if name not in aman:
            logger.warning(
                "\tNo position_match associated with this observation. Skipping."
            )
            continue

        # Figure out encoders
        if encs is None:
            encoders.append((np.nan,) * 3)
        else:
            az = encs["az"]
            el = encs["el"]
            bs = encs["bs"]
            if _encs_notclose(az, el, bs):
                logger.warning("\tNot all encoder values are close. Skipping.")
                continue
            encoders.append((np.nanmedian(az), np.nanmedian(el), np.nanmedian(bs)))
        # Put SMuRF band channel in the correct place
        smurf = AxisManager(aman.dets)
        smurf.wrap("band", aman[name].band, [(0, smurf.dets)])
        smurf.wrap("channel", aman[name].channel, [(0, smurf.dets)])
        aman.det_info.wrap("smurf", smurf)

        if detmap is not None:
            g3u.add_detmap_info(aman, detmap)
        have_wafer = "wafer" in aman.det_info
        if not have_wafer:
            logger.error("\tThis observation has no detmap results, skipping")
            continue

        det_ids = aman.det_info.det_id
        x = aman.det_info.wafer.det_x
        y = aman.det_info.wafer.det_y
        pol = aman.det_info.wafer.angle
        if use_matched:
            det_ids = aman[name].matched_det_id
            dm_sort = np.argsort(aman.det_info.det_id)
            mapping = np.argsort(np.argsort(det_ids))
            x = x[dm_sort][mapping]
            y = y[dm_sort][mapping]
            pol = pol[dm_sort][mapping]

        focal_plane = np.column_stack(
            (aman[name].xi, aman[name].eta, aman[name].polang, x, y, pol)
        ).astype(float)
        out_msk = aman[name].outliers
        focal_plane[out_msk, :3] = np.nan

        for di, fp in zip(det_ids, focal_plane):
            try:
                fp_dict[di].append(fp)
            except KeyError:
                fp_dict[di] = [fp]

    if not fp_dict:
        logger.error("No valid observations provided")
        sys.exit()

    # Compute nominal encoder vals
    encoders = np.column_stack(encoders)
    if _encs_notclose(*encoders):
        encoders = (np.nan,) * 3
        logger.error(
            "Not all of the inputs were taken at similar measurements. Outputs will have nans for encoder related fields."
        )
        logger.warning("FOV rotation will be 0")
    else:
        encoders = np.nanmedian(encoders, axis=1)
        if np.isnan(encoders).any():
            logger.error("Some or all of the encoders are nan")
            logger.warning("FOV rotation may be 0")

    # Compute the average focal plane while ignoring outliers
    det_id, focal_plane = _avg_focalplane(fp_dict)
    measured = focal_plane[:3]

    # Get nominal xi, eta, gamma
    nominal = get_nominal(focal_plane, config["coord_transform"], encoders)

    # Compute transformation between the two nominal and measured pointing
    measured_gamma = np.isfinite(measured[2]).all()
    if measured_gamma:
        gamma_scale, gamma_shift = gamma_fit(nominal[2], measured[2])
    else:
        logger.warning(
            "No polarization data availible, gammas will be filled with the nominal values."
        )
        focal_plane[2] = nominal[2]
        gamma_scale = 1.0
        gamma_shift = 0.0

    nominal = np.vstack(nominal[:2])
    measured = np.vstack(measured[:2])
    affine, shift = op.get_affine(nominal, measured)
    scale, shear, rot = op.decompose_affine(affine)
    shear = shear.item()
    rot = op.decompose_rotation(rot)[-1]

    plot = config.get("plot", False)
    if plot:
        _mk_plot(nominal, measured, affine, shift, plot)

    shift = (*shift, gamma_shift)
    scale = (*scale, gamma_scale)

    _log_vals(shift, scale, shear, rot)

    # Compute the lever arm
    lever_arm = get_nominal(np.zeros(6), config["coord_transform"], encoders)

    # Make final outputs and save
    logger.info("Saving data to %s", outpath)
    fpout = _mk_fpout(det_id, focal_plane)
    tpout = _mk_tpout(shift, scale, shear, rot)
    refout = _mk_refout(lever_arm, encoders)
    with h5py.File(outpath, "w") as f:
        write_dataset(fpout, f, "focal_plane", overwrite=True)
        _add_attrs(f["focal_plane"], {"measured_gamma": measured_gamma})
        write_dataset(tpout, f, "offsets", overwrite=True)
        _add_attrs(f["offsets"], {"affine_matrix": affine})
        write_dataset(refout, f, "reference", overwrite=True)


if __name__ == "__main__":
    main()
