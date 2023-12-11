import logging
import os

import numpy as np
import pandas as pd

import pilates.beam.postprocessor as beam_post

logger = logging.getLogger(__name__)


def prepare_input(settings, is_initial_year: bool) -> bool:
    logger.info("Preparing input for frism")

    if is_initial_year:
        path_to_beam_skims = os.path.join(settings['beam_local_output_folder'], settings['skims_fname'])
    else:
        path_to_beam_skims = beam_post.find_produced_od_skims(settings['beam_local_output_folder'], suffix="omx")

    if not os.path.exists(path_to_beam_skims):
        logger.warning("Beam skims not found: %s", path_to_beam_skims)
        return False

    tt_minutes = beam_post.read_beam_omx_skims_to_pd_series(path_to_beam_skims, measure='SOV_TIME__AM')
    if tt_minutes is None:
        return False
    tt_minutes = tt_minutes[tt_minutes > 0].rename('TIME_minutes').to_frame()

    frism_schema = {
        'origin': str,
        'destination': str,
        'TIME_minutes': np.float32,
    }
    frism_columns = frism_schema.keys()
    frism_tt_path = os.path.join(settings['frism_data_folder'], settings['region'], 'Geo_data', 'tt_df_cbg.csv.gz')
    frism_tt_df = pd.read_csv(frism_tt_path, usecols=frism_columns, index_col=['origin', 'destination'], dtype=frism_schema)
    new_tt_df = pd.concat([tt_minutes, frism_tt_df.loc[frism_tt_df.index.difference(tt_minutes.index, sort=False)]])
    new_tt_df.reset_index(inplace=True)
    new_tt_df.to_csv(frism_tt_path, index=False)
    return True

