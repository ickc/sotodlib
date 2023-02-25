# Copyright (c) 2022-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import re
import datetime

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import Column, QTable

import toast
from toast.timing import function_timer
from toast.traits import trait_docs, Int, Unicode, Instance, List, Unit
from toast.ops.operator import Operator
from toast.utils import Environment, Logger, Timer
from toast.dist import distribute_discrete
from toast.observation import default_values as defaults

import so3g

from ...core import Context, AxisManager
from ...core.axisman import AxisInterface

from ..instrument import SOFocalplane, SOSite


@trait_docs
class LoadContext(Operator):
    """Load one or more observations from a Context.

        Given a context, load one or more observations.  The observation

        - can take context as an instance trait

        - use get_obs() and get_meta()  get_obs returns data plus meta

        - could provide an instance of preprocess operator


    # Notes:


    # det_info key has focalplane properties

    # det_info has other axis managers- recursively promote and flatten

    # array_data has other focalplane properties

    # dets: label axis

    # samples: offset axis


    # Use metadata to create observation (fp, etc)
    # Use local dets to get_obs

    # Use other amans to populate shared and detdata

    # - sample index only == shared
    # - det and sample index == detdata


        For ax_flags tuples, a negative bit value indicates that the flag data
        should first be inverted before combined.



    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    context = Instance(
        klass=Context,
        help="The Context, which should exist on all processes",
    )

    observations = List(list(), help="List of observation IDs to load")

    readout_ids = List(list(), help="Only load this list of readout_id values")

    detsets = List(list(), help="Only load this list of detset values")

    bands = List(list(), help="Only load this list of band values")

    ax_times = Unicode(
        "timestamps",
        help="Name of field to associate with times",
    )

    ax_flags = List(
        [],
        help="Tuples of (field, bit value) merged to shared_flags",
    )

    ax_det_signal = Unicode(
        "signal",
        help="Name of field to associate with det_data",
    )

    ax_det_flags = List(
        [],
        help="Tuples of (field, bit_value) merged to det_flags",
    )

    ax_boresight_az = Unicode("boresight_az", help="Field with boresight Az")

    ax_boresight_el = Unicode("boresight_el", help="Field with boresight El")

    ax_boresight_roll = Unicode("boresight_roll", help="Field with boresight Roll")

    axis_detector = Unicode(
        "dets", help="Name of the LabelAxis for the detector direction"
    )

    axis_sample = Unicode(
        "samps", help="Name of the OffsetAxis for the sample direction"
    )

    telescope_name = Unicode("UNKNOWN", help="Name of the telescope")

    detset_key = Unicode(
        None,
        allow_none=True,
        help="Column of the focalplane detector_data to use for data distribution",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for detector signal",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for detector flags",
    )

    azimuth = Unicode(
        defaults.azimuth, help="Observation shared key for boresight Azimuth"
    )

    elevation = Unicode(
        defaults.elevation, help="Observation shared key for boresight Elevation"
    )

    roll = Unicode(defaults.roll, help="Observation shared key for boresight Roll")

    boresight_azel = Unicode(
        defaults.boresight_azel,
        help="Observation shared key for boresight Az/El quaternions",
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        help="Observation shared key for boresight RA/DEC quaternions",
    )

    corotator_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for corotator_angle (if it is used)",
    )

    boresight_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for boresight rotation angle (if it is used)",
    )

    hwp_angle = Unicode(
        None,
        allow_none=True,
        help="Observation shared key for HWP angle (if it is used)",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        if len(self.observations) == 0:
            raise RuntimeError("No observation IDs specified")

        comm = data.comm

        # Build our detector selection dictionary
        det_select = None
        if len(self.readout_ids) > 0 or len(self.bands) > 0 or len(self.detsets) > 0:
            # We have some selection
            det_select = dict()
            if self.readout_ids is not None:
                det_select["readout_id"] = list(self.readout_ids)
            if self.bands is not None:
                det_select["band"] = list(self.bands)
            if self.detsets is not None:
                det_select["detset"] = list(self.detsets)

        # One global process queries the observation metadata and computes
        # the observation distribution among process groups.

        obs_props = None
        if comm.world_rank == 0:
            obs_props = list()
            for iobs, obs_id in enumerate(self.observations):
                meta = self.context.get_meta(obs_id, dets=det_select)
                oprops = dict()
                oprops["name"] = obs_id
                oprops["n_samp"] = meta["samps"].count
                oprops["n_det"] = len(meta["dets"].vals)
                obs_props.append(oprops)

        if comm.comm_world is not None:
            obs_props = comm.comm_world.bcast(obs_props, root=0)

        # Distribute observations among groups

        obs_sizes = [x["n_det"] * x["n_samp"] for x in obs_props]
        groupdist = distribute_discrete(obs_sizes, comm.ngroups)
        group_firstobs = groupdist[comm.group].offset
        group_numobs = groupdist[comm.group].n_elem

        # Every group loads its observations
        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obs_name = obs_props[obindx]["name"]
            n_samp = obs_props[obindx]["n_samp"]

            # One process in the group loads the metadata, builds the focalplane
            # model, and broadcasts to the rest of the group.

            det_props = None
            obs_meta = None
            if comm.group_rank == 0:
                meta = self.context.get_meta(obs_name, dets=det_select)
                # For each element of meta we do the following:
                # - If the object has one axis and it is the detector axis,
                #   treat it as a column in the detector property table
                # - If the object has one axis and it is the sample axis,
                #   it will be loaded later as shared data.
                # - If the object has multiple axes, load it later
                # - If the object has no axes, then treat it as observation
                #   metadata
                # - If the object is a nested AxisManager, descend and apply
                #   the same steps as above.
                obs_meta = dict()
                fp_cols = dict()
                self._parse_meta(meta, None, obs_meta, None, fp_cols)

                # Construct table
                det_props = QTable(fp_cols)

            if comm.comm_group is not None:
                obs_meta = comm.comm_group.bcast(obs_meta, root=0)
                det_props = comm.comm_group.bcast(det_props, root=0)

            # Create the observation.  We intentionally use the generic focalplane
            # and class here, in case we are loading data from legacy experiments.

            # We will get the timestamps later and update everything to the true sample
            # rate.  Use a placeholder for now.
            fake_rate = 1.0 * u.Hz

            # Convert any focalplane quaternion offsets to toast format

            print(obs_meta)
            print(det_props.colnames)
            print(det_props)

            name_col = Column(name="name", data=det_props["det_info_readout_id"])
            quat_col = Column(
                name="quat",
                data=toast.instrument_coords.xieta_to_quat(
                    det_props["focal_plane_xi"],
                    det_props["focal_plane_eta"],
                    det_props["focal_plane_gamma"],
                ),
            )
            det_props.add_column(quat_col, index=0)
            det_props.add_column(name_col, index=0)

            focalplane = toast.instrument.Focalplane(
                detector_data=det_props,
                sample_rate=fake_rate,
            )

            if self.detset_key is None:
                detsets = None
            else:
                detsets = focalplane.detector_groups(self.detset_key)

            # For now, this should be good enough position for instruments near the
            # S.O. location.
            site = SOSite()

            telescope = toast.instrument.Telescope(
                self.telescope_name, focalplane=focalplane, site=site
            )

            # Note:  the session times will be updated later when reading timestamps
            session = toast.instrument.Session(obs_name)

            ob = toast.Observation(
                comm,
                telescope,
                n_samp,
                name=obs_name,
                session=session,
                detector_sets=detsets,
                sample_sets=None,
                process_rows=comm.group_size,
            )

            # Create observation fields
            ob.shared.create_column(
                self.times,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.azimuth,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.elevation,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.roll,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.boresight_azel,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.boresight_radec,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )
            if self.hwp_angle is not None:
                ob.shared.create_column(
                    self.hwp_angle,
                    shape=(ob.n_local_samples,),
                    dtype=np.float64,
                )
            if self.boresight_angle is not None:
                ob.shared.create_column(
                    self.boresight_angle,
                    shape=(ob.n_local_samples,),
                    dtype=np.float64,
                )
            if self.corotator_angle is not None:
                ob.shared.create_column(
                    self.corotator_angle,
                    shape=(ob.n_local_samples,),
                    dtype=np.float64,
                )
            ob.shared.create_column(
                self.shared_flags,
                shape=(ob.n_local_samples,),
                dtype=np.uint8,
            )
            ob.shared.create_column(
                defaults.position,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
            )
            ob.shared.create_column(
                defaults.velocity,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
            )
            ob.detdata.create(
                self.det_data, dtype=np.float64, units=self.det_data_units
            )
            ob.detdata.create(self.det_flags, dtype=np.uint8)

            # Now every process loads its data
            axtod = self.context.get_obs(obs_name, dets=ob.local_detectors)
            print(axtod)
            self._parse_data(ob, axtod, None)

            # Position and velocity of the observatory are simply computed.  Only the
            # first row of the process grid needs to do this.
            position = None
            velocity = None
            if ob.comm_col_rank == 0:
                position, velocity = site.position_velocity(ob.shared[self.times])
            ob.shared[defaults.position].set(position, offset=(0, 0), fromrank=0)
            ob.shared[defaults.velocity].set(velocity, offset=(0, 0), fromrank=0)

            # First row of the process grid computes boresight quaternions from
            # boresight angles.
            bore_azel = None
            bore_radec = None
            if ob.comm_col_rank == 0:
                bore_azel = toast.qarray.from_lonlat_angles(
                    -ob.shared[self.azimuth].data,
                    ob.shared[self.elevation].data,
                    ob.shared[self.roll].data,
                )
                bore_radec = toast.coordinates.azel_to_radec(
                    site,
                    ob.shared[self.times].data,
                    ob.shared[self.boresight_azel].data,
                    use_ephem=True,
                )
            ob.shared[self.boresight_azel].set(bore_azel, offset=(0, 0), fromrank=0)
            ob.shared[self.boresight_radec].set(bore_radec, offset=(0, 0), fromrank=0)
            data.obs.append(ob)

    def _parse_data(self, obs, axman, base):
        # Some metadata has already been parsed, but some new values
        # may only show up when reading data, so we need to handle those
        # as well.
        shared_ax_to_obs = {
            self.ax_times: self.times,
            self.ax_boresight_az: self.azimuth,
            self.ax_boresight_el: self.elevation,
            self.ax_boresight_roll: self.roll,
        }
        shared_flag_invert = {x[0]: (x[1] < 0) for x in self.ax_flags}
        shared_flag_fields = {x[0]: abs(x[1]) for x in self.ax_flags}
        det_flag_invert = {x[0]: (x[1] < 0) for x in self.ax_det_flags}
        det_flag_fields = {x[0]: abs(x[1]) for x in self.ax_det_flags}

        if base is None:
            om = obs
        else:
            if base not in obs:
                obs[base] = dict()
            om = obs[base]
        for key in axman.keys():
            if base is not None:
                data_key = f"{base}_{key}"
            else:
                data_key = key
            if isinstance(axman[key], AxisInterface):
                # This is one of the axes
                continue
            if isinstance(axman[key], AxisManager):
                # Descend
                self._parse_data(obs, axman[key], key)
            else:
                # FIXME:  It would be nicer if this information was available
                # through a public member...
                field_axes = axman._assignments[key]
                if len(field_axes) == 0:
                    # This data is not associated with an axis.  If it does not
                    # yet exist in the observation metadata, then add it.
                    if key not in om:
                        om[key] = axman[key]
                elif field_axes[0] == self.axis_detector:
                    if len(field_axes) == 1:
                        # This is a detector property
                        if data_key in obs.telescope.focalplane.detector_data.colnames:
                            # We already included this in the detector properties
                            continue
                        # This must be some per-detector derived data- add to the
                        # observation dictionary
                        if data_key not in om:
                            om[key] = axman[key]
                    elif field_axes[1] == self.axis_sample:
                        # This is detector data.  See if it is one of the standard
                        # fields we are parsing.
                        if data_key == self.ax_det_signal:
                            print(f"Add detector signal {data_key}")
                            obs.detdata[self.det_data][:, :] = axman[key]
                        elif data_key in det_flag_fields:
                            print(f"Add detector flags {data_key}")
                            if isinstance(axman[key], so3g.proj.RangesMatrix):
                                temp = np.empty(obs.n_local_samples, dtype=np.uint8)
                                if det_flag_invert[data_key]:
                                    for idet, det in enumerate(obs.local_detectors):
                                        temp[:] = det_flag_fields[data_key]
                                        for rg in axman[key][idet].ranges():
                                            temp[rg[0] : rg[1]] = 0
                                        obs.detdata[self.det_flags][det] |= temp
                                else:
                                    for idet, det in enumerate(obs.local_detectors):
                                        temp[:] = 0
                                        for rg in axman[key][idet].ranges():
                                            temp[rg[0] : rg[1]] = det_flag_fields[
                                                data_key
                                            ]
                                        obs.detdata[self.det_flags][det] |= temp
                            else:
                                # Explicit flags per sample
                                temp = det_flag_fields[data_key] * np.ones_like(
                                    obs.detdata[self.det_flags][:]
                                )
                                if det_flag_invert[data_key]:
                                    temp[axman[key] != 0] = 0
                                else:
                                    temp[axman[key] == 0] = 0
                                obs.detdata[self.det_flags][:] |= temp
                        else:
                            # Some other kind of detector data
                            print(f"Add detector data {data_key} <-- {key}")
                            if len(axman[key].shape) > 2:
                                shp = axman[key].shape[2:]
                            else:
                                shp = None
                            obs.detdata.create(
                                data_key,
                                sample_shape=shp,
                                dtype=axman[key].dtype,
                                units=u.dimensionless_unscaled,
                            )
                    else:
                        # Must be some other type of object...
                        if data_key not in om:
                            om[key] = axman[key]
                elif field_axes[0] == self.axis_sample:
                    # This is shared data
                    print(f"{key} --> {data_key}")
                    if isinstance(axman[key], so3g.proj.Ranges):
                        # This is a set of 1D shared ranges.  Translate this to a
                        # toast interval list.
                        samplespans = list()
                        for rg in axman[key].ranges():
                            samplespans.append((rg[0], rg[1] - 1))
                        obs.intervals[data_key] = toast.intervals.IntervalList(
                            obs.shared[self.times], samplespans=samplespans
                        )
                    elif data_key in shared_ax_to_obs:
                        print(
                            f"Add shared telescope field {shared_ax_to_obs[data_key]} <-- {key}"
                        )
                        axbuf = None
                        if obs.comm_col_rank == 0:
                            axbuf = axman[key]
                        obs.shared[shared_ax_to_obs[data_key]].set(
                            axbuf,
                            offset=(0,),
                            fromrank=0,
                        )
                    elif data_key in shared_flag_fields:
                        axbuf = None
                        print(f"Add shared flag {self.shared_flags} <-- {key}")
                        if obs.comm_col_rank == 0:
                            axbuf = np.array(obs.shared[self.shared_flags])
                            temp = shared_flag_fields[data_key] * np.ones_like(axbuf)
                            if shared_flag_invert[data_key]:
                                temp[axman[key] != 0] = 0
                            else:
                                temp[axman[key] == 0] = 0
                            axbuf |= temp
                        obs.shared[self.shared_flags].set(
                            axbuf,
                            offset=(0,),
                            fromrank=0,
                        )
                    else:
                        # This is some other shared data.
                        print(f"Add shared data {data_key} <-- {key}")
                        obs.shared.create_column(
                            data_key,
                            shape=axman[key].shape,
                            dtype=axman[key].dtype,
                        )
                        sdata = None
                        if obs.comm_col_rank == 0:
                            sdata = axman[key]
                        obs.shared[data_key].set(sdata)
                else:
                    # Some other object...
                    if data_key not in om:
                        om[key] = axman[key]
        if base is not None and len(om) == 0:
            # We created a dictionary that was not used, clean it up
            del obs[base]

    def _parse_meta(self, axman, obs_base, obs_meta, fp_base, fp_cols):
        if obs_base is None:
            om = obs_meta
        else:
            obs_meta[obs_base] = dict()
            om = obs_meta[obs_base]
        for key in axman.keys():
            if fp_base is not None:
                fp_key = f"{fp_base}_{key}"
            else:
                fp_key = key
            if isinstance(axman[key], AxisInterface):
                # This is one of the axes
                continue
            if isinstance(axman[key], AxisManager):
                # Descend
                self._parse_meta(axman[key], key, obs_meta, fp_key, fp_cols)
            else:
                # FIXME:  It would be nicer if this information was available
                # through a public member...
                field_axes = axman._assignments[key]
                if len(field_axes) == 0:
                    # This data is not associated with an axis.
                    om[key] = axman[key]
                elif len(field_axes) == 1 and field_axes[0] == self.axis_detector:
                    # This is a detector property
                    if fp_key in fp_cols:
                        msg = f"Context meta key '{fp_key}' is duplicated in nested"
                        msg += " AxisManagers"
                        raise RuntimeError(msg)
                    fp_cols[fp_key] = Column(name=fp_key, data=np.array(axman[key]))
        if obs_base is not None and len(om) == 0:
            # There were no meta data keys- delete this dict
            del obs_meta[obs_base]

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {
            "shared": [
                self.times,
                self.boresight_azel,
            ],
            "detdata": [self.det_data],
        }
        if self.boresight_radec is not None:
            prov["shared"].append(self.boresight_radec)
        if self.boresight_angle is not None:
            prov["shared"].append(self.boresight_angle)
        if self.corotator_angle is not None:
            prov["shared"].append(self.corotator_angle)
        if self.hwp_angle is not None:
            prov["shared"].append(self.hwp_angle)
        return prov
