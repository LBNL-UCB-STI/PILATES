import pandas as pd
import logging
import os
import h5py
import openmatrix as omx
import numpy as np
import shutil

from pilates.utils.geog import geoid_to_zone_map

logger = logging.getLogger(__name__)

skim_dtypes = {
    'timePeriod': str,
    'pathType': str,
    'origin': int,
    'destination': int,
    'TIME_minutes': float,
    'TOTIVT_IVT_minutes': float,
    'VTOLL_FAR': float,
    'DIST_meters': float,
    'WACC_minutes': float,
    'WAUX_minutes': float,
    'WEGR_minutes': float,
    'DTIM_minutes': float,
    'DDIST_meters': float,
    'KEYIVT_minutes': float,
    'FERRYIVT_minutes': float,
    'BOARDS': float,
    'DEBUG_TEXT': str
}


def _load_raw_skims(settings, asim_data_dir, skim_format):
    skims_fname = settings.get('skims_fname', False)

    try:
        if skim_format == 'beam':
            if skims_fname.endswith('csv'):
                path_to_skims = os.path.join(
                    settings['beam_local_output_folder'], skims_fname)
                # load skims from disk or url
                skims = pd.read_csv(path_to_skims, dtype=skim_dtypes)
                skims = skims.loc[(
                                          skims['pathType'] == 'SOV') & (
                                          skims['timePeriod'] == 'AM')]
                skims = skims[[
                    'origin', 'destination', 'TOTIVT_IVT_minutes',
                    'DIST_meters']]
                skims = skims.rename(columns={
                    'origin': 'from_zone_id',
                    'destination': 'to_zone_id',
                    'TOTIVT_IVT_minutes': 'SOV_AM_IVT_mins'})
            elif skims_fname.endswith('omx'):
                skims_fname = "skims.omx"
                mutable_skims_location = os.path.join(asim_data_dir, skims_fname)
                skims = omx.open_file(mutable_skims_location, 'r')
                zone_ids = skims.mapping('zone_id').keys()
                index = pd.Index(zone_ids, name="from_zone_id", dtype=str)
                columns = pd.Index(zone_ids, name="to_zone_id", dtype=str)
                travel_time_mins = np.array(skims['SOV_TIME__AM'])
                out = pd.DataFrame(travel_time_mins, index=index, columns=columns).stack().rename('SOV_AM_IVT_mins')
                skims.close()
                return out.to_frame()
            else:
                raise NotImplementedError("Invalid skim format {0}".format(skims_fname.split('.')[-1]))
        elif skim_format == 'polaris':
            path_to_skims = os.path.join(
                settings['polaris_local_data_folder'], skims_fname)
            f = h5py.File(path_to_skims, 'r')
            ivtt_8_9 = pd.DataFrame(list(f['auto_skims']['t4']['ivtt']))
            cost_8_9 = pd.DataFrame(list(f['auto_skims']['t4']['cost']))
            f.close()
            ivtt_8_9 = pd.DataFrame(
                ivtt_8_9.stack(), columns=['auto_ivtt_8_9_am'])
            cost_8_9 = pd.DataFrame(
                cost_8_9.stack(), columns=['auto_cost_8_9_am'])
            skims = ivtt_8_9.join(cost_8_9)
            skims.index.names = ['from_zone_id', 'to_zone_id']
            skims = skims.reset_index()

    except KeyError:
        raise KeyError(
            "Couldn't find input skims named {0}".format(skims_fname))

    logger.info("Converting skims to UrbanSim data format.")
    skims['from_zone_id'] = skims['from_zone_id'].astype('str')
    skims['to_zone_id'] = skims['to_zone_id'].astype('str')

    # for GEOID/FIPS-based skims, we have to convert the zone IDs
    if settings['skims_zone_type'] in ['block', 'block_group']:
        mapping = geoid_to_zone_map(settings)
        for col in ['from_zone_id', 'to_zone_id']:
            skims[col] = skims[col].map(mapping)

    skims = skims.set_index(['from_zone_id', 'to_zone_id'])

    return skims


def copy_data_to_mutable_location(settings, output_dir):
    region = settings['region']
    region_id = settings['region_to_region_id'][region]
    model_data_fname = settings['usim_formattable_input_file_name'].format(region_id=region_id)
    data_dir = settings['usim_local_data_input_folder']
    src = os.path.join(data_dir, model_data_fname)
    dest = os.path.join(output_dir, model_data_fname)
    logger.info("Copying input urbansim data from {0} to {1}".format(src, dest))
    other_data_fnames = [
        "hsize_ct_{0}.csv".format(region_id),
        "income_rates_{0}.csv".format(region_id),
        "relmap_{0}.csv".format(region_id),
        "blocks_school_districts_2010.csv"
    ]
    for fname in other_data_fnames:
        src = os.path.join(data_dir, fname)
        dest = os.path.join(output_dir, fname)
        logger.info("Copying input urbansim file from {0} to {1}".format(src, dest))
        shutil.copyfile(src, dest)


def add_skims_to_model_data(settings, data_dir=None, skims_dir=None):
    mapping = geoid_to_zone_map(settings)
    if not data_dir:
        data_dir = settings['usim_local_data_input_folder']
    output_geoid_loc = os.path.join(data_dir, "geoid_to_zone.csv")
    logger.info("Writing zone mapping to {0}".format(output_geoid_loc))
    pd.Series(mapping).to_frame("zone_id").rename_axis("GEOID").to_csv(output_geoid_loc)

    # load skims
    logger.info("Loading skims from disk")
    region = settings['region']
    skim_format = settings['travel_model']
    df = _load_raw_skims(settings, skims_dir, skim_format=skim_format)

    # load datastore
    region_id = settings['region_to_region_id'][region]
    model_data_fname = settings['usim_formattable_input_file_name'].format(region_id=region_id)

    model_data_fpath = os.path.join(data_dir, model_data_fname)
    if not os.path.exists(model_data_fpath):
        raise ValueError('No input data found at {0}'.format(
            model_data_fpath))
    store = pd.HDFStore(model_data_fpath)

    # add skims
    store['travel_data'] = df
    del df

    # update blocks table with zone ID's that match the skims.
    # note: should only have to be run the first time the
    # the base year urbansim data is touched by pilates
    zone_id_col = 'zone_id'
    if zone_id_col not in store['blocks'].columns:

        blocks = store['blocks'].copy()
        mapping = geoid_to_zone_map(settings)
        zone_type = settings['skims_zone_type']

        if zone_type == 'block':
            logger.info("Mapping block IDs")
            blocks[zone_id_col] = blocks.index.astype(str).replace(mapping)

        elif zone_type == 'block_group':
            logger.info("Mapping blocks to block group IDS")
            blocks[zone_id_col] = blocks.block_group_id.astype(str).replace(
                mapping)

        elif zone_type == 'taz':
            logger.info("Mapping block IDs to TAZ")
            geoid_to_zone_fpath = \
                "pilates/utils/data/{0}/{1}/geoid_to_zone.csv".format(
                    region, skim_format)

            block_taz = pd.read_csv(
                geoid_to_zone_fpath, dtype={'GEOID': str, zone_id_col: str})
            block_taz = block_taz.set_index('GEOID')[zone_id_col]
            block_taz.index.name = 'block_id'
            blocks = blocks.join(block_taz)

        blocks[zone_id_col] = blocks[zone_id_col].fillna('foo')
        blocks = blocks[blocks[zone_id_col] != 'foo'].copy()
        blocks[zone_id_col] = blocks[zone_id_col].astype(str)

        logger.info("Write out to the data store.")
        households = store['households'].copy()
        persons = store['persons'].copy()
        jobs = store['jobs'].copy()
        units = store['residential_units'].copy()
        assert households['block_id'].isin(blocks.index).all()
        assert persons['household_id'].isin(households.index).all()
        assert jobs['block_id'].isin(blocks.index).all()
        assert units['block_id'].isin(blocks.index).all()
        store['blocks'] = blocks
        store['households'] = households
        store['persons'] = persons
        store['jobs'] = jobs
        store['residential_units'] = units

    store.close()
