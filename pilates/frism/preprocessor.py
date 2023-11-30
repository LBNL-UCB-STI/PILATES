import logging
import os
import pandas as pd
import numpy as np
import pilates.activitysim.preprocessor as asim_pre

logger = logging.getLogger(__name__)


def prepare_input(settings):
    logger.info("Preparing input for frism")
    frism_schema = {
        'origin': str,
        'destination': str,
        'TIME_minutes': np.float32,
    }
    frism_columns = frism_schema.keys()
    beam_skims = asim_pre._load_raw_beam_skims(settings)
    if isinstance(beam_skims, pd.DataFrame):
        beam_skims = beam_skims[(beam_skims['timePeriod'] == 'AM') & (beam_skims['pathType'] == 'SOV')]
        beam_skims = beam_skims[frism_columns].copy()
        beam_skims = beam_skims.set_index(['origin', 'destination'])
        frism_tt_path = os.path.join(settings['frism_data_folder'], 'Geo_data', 'tt_df_cbg.csv.gz')

        frism_tt_df = pd.read_csv(frism_tt_path, usecols=frism_columns, index_col=['origin', 'destination'])
        new_tt = pd.concat([beam_skims, frism_tt_df.loc[frism_tt_df.index.difference(beam_skims.index, sort=False)]])
        new_tt.reset_index()\
            .to_csv(os.path.join(settings['frism_data_folder'], 'Geo_data', 'tt_df_cbg-2.csv'), index=False)
    else:
        raise NotImplementedError("Only pandas.DataFrame beam skims are supported")
