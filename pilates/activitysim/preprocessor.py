import glob
import logging
import os
import shutil
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import openmatrix as omx
import pandas as pd
import requests
from pandas.api.types import is_string_dtype
from shapely import wkt
from tqdm import tqdm

from pilates.utils.geog import get_block_geoms, \
    map_block_to_taz, get_zone_from_points, \
    get_taz_geoms, get_county_block_geoms, geoid_to_zone_map

from pilates.utils.io import read_datastore

logger = logging.getLogger(__name__)

beam_skims_types = {'timePeriod': str,
                    'pathType': str,
                    'origin': str,
                    'destination': str,
                    'TIME_minutes': np.float32,
                    'TOTIVT_IVT_minutes': np.float32,
                    'VTOLL_FAR': np.float32,
                    'DIST_meters': np.float32,
                    'WACC_minutes': np.float32,
                    'WAUX_minutes': np.float32,
                    'WEGR_minutes': np.float32,
                    'DTIM_minutes': np.float32,
                    'DDIST_meters': np.float32,
                    'KEYIVT_minutes': np.float32,
                    'FERRYIVT_minutes': np.float32,
                    'BOARDS': np.float32,
                    'DEBUG_TEXT': str
                    }

beam_origin_skims_types = {"origin": str,
                           "timePeriod": str,
                           "reservationType": str,
                           "waitTimeInMinutes": np.float32,
                           "costPerMile": np.float32,
                           "unmatchedRequestPortion": np.float32,
                           "observations": int
                           }

default_speed_mph = {
    "COM": 25.0,
    "HVY": 20.0,
    "LRF": 12.5,
    "LOC": 12.5,
    "EXP": 17.0,
    "TRN": 15.0
}
default_fare_dollars = {
    "COM": 10.0,
    "HVY": 4.0,
    "LRF": 2.5,
    "LOC": 2.5,
    "EXP": 4.0,
    "TRN": 15.0
}


#########################
#### Common functions ###
#########################
def zone_order(settings, year):
    """ 
    Returns the order of the zones to create consistent skims. 
    
    Return:
    -------
    Numpy Array. One dimension array, the index of the array represents the order. 
    
    """
    zone_type = settings['skims_zone_type']
    mapping = geoid_to_zone_map(settings, year)

    if zone_type == 'taz':
        num_taz = len(set(mapping.values()))
        order = np.array(range(1, num_taz + 1)).astype(str)
    else:
        order = pd.DataFrame.from_dict(mapping, orient='index', columns=['zone_id']).astype(int)
        order = np.array(order.sort_values('zone_id').index)
    return order


def read_skims(settings, mode='a', data_dir=None, file_name='skims.omx'):
    """
    Opens skims OMX file. 
    Parameters:
    ------------
    mode : string
        'r' for read-only; 
        'w' to write (erases existing file); 
        'a' to read/write an existing file (will create it if doesn't exist).
        Ignored in read-only mode.  
    """
    if data_dir is None:
        data_dir = settings['asim_local_input_folder']
    path = os.path.join(data_dir, file_name)
    skims = omx.open_file(path, mode=mode)
    return skims


def zone_id_to_taz(zones, asim_zone_id_col='TAZ',
                   default_zone_id_col='zone_id'):
    if zones.index.name != asim_zone_id_col:
        if asim_zone_id_col in zones.columns:
            zones.set_index(asim_zone_id_col, inplace=True)
        elif zones.index.name == default_zone_id_col:
            zones.index.name = asim_zone_id_col
        elif asim_zone_id_col not in zones.columns:
            zones.rename(columns={default_zone_id_col: asim_zone_id_col})
        else:
            logger.error(
                "Not sure what column in the zones table is the zone ID!")
    return zones


def read_zone_geoms(settings, year,
                    asim_zone_id_col='TAZ',
                    default_zone_id_col='zone_id'):
    """
    Returns a GeoPandas dataframe with the zones geometries. 
    """
    store, table_prefix_year = read_datastore(settings, year, True)
    zone_type = settings['skims_zone_type']
    zone_key = '/{0}_zone_geoms'.format(zone_type)

    if zone_key in store.keys():
        logger.info(
            "Loading {0} zone geometries from .h5 datastore!".format(
                zone_type))
        zones = store[zone_key]

        if 'geometry' in zones.columns:
            zones.loc[:, 'geometry'] = zones.loc[:, 'geometry'].apply(wkt.loads)
            zones = gpd.GeoDataFrame(
                zones, geometry='geometry', crs='EPSG:4326')
        else:
            raise KeyError(
                "Table 'zone_geoms' exists in the .h5 datastore but "
                "no geometry column was found!")
    else:
        logger.info("Downloading zone geometries on the fly!")
        region = settings['region']
        if zone_type == 'taz':
            zones = get_taz_geoms(settings, zone_id_col_out=default_zone_id_col)
            zones.set_index(default_zone_id_col, inplace=True)
        else:
            mapping = geoid_to_zone_map(settings, year)
            zones = get_block_geoms(settings)
            # zones['GEOID'] = zones['GEOID'].astype(str)
            assert is_string_dtype(zones['GEOID']), "GEOID dtype should be str"
            zones.loc[:, default_zone_id_col] = zones.loc[:, 'GEOID'].replace(mapping)
            zones.set_index(default_zone_id_col, inplace=True)
            assert zones.index.inferred_type == 'string', "zone_id dtype should be str"

        # save zone geoms in .h5 datastore so we don't
        # have to do this again
        out_zones = pd.DataFrame(zones.copy())
        out_zones.loc[:, 'geometry'] = out_zones['geometry'].apply(lambda x: x.wkt)

        logger.info("Storing zone geometries to .h5 datastore!")
        store[zone_key] = out_zones

    store.close()

    # Sort zones by zone_id. 
    # momentary int transformation to 
    # make sure it sort 1, 2, 10 instead of '1', 10', '2'
    zones.index = zones.index.astype(int)
    zones = zones.sort_index()
    zones.index = zones.index.astype(str)
    return zone_id_to_taz(zones, asim_zone_id_col, default_zone_id_col)


####################################
#### RAW BEAM SKIMS TO SKIMS.OMX ###
####################################
def read_skim(filename):
    logger.info("Loading raw beam skims from disk: {}".format(filename))
    df = pd.read_csv(filename, index_col=None, header=0, dtype=beam_skims_types)
    return df


def _load_raw_beam_skims(settings, convertFromCsv=True, blank=False):
    """ Read BEAM skims (csv format) from local storage. 
    Parameters: 
    ------------
    - settings:
    
    Return:
    --------
    - pandas DataFrame. 
    """
    zone_type = settings['skims_zone_type']
    skims_fname = settings.get('skims_fname', False)
    path_to_beam_skims = os.path.join(
        settings['beam_local_output_folder'], skims_fname)

    if (not os.path.exists(path_to_beam_skims)) & (not convertFromCsv):
        return None
    if blank:
        return None

    try:
        if '.csv' in path_to_beam_skims:
            skims = read_skim(path_to_beam_skims)
        elif path_to_beam_skims.endswith('.omx'):
            # We've gotten a head start
            skims = omx.open_file(path_to_beam_skims, mode='a')
        else:  # path is a folder with multiple files
            all_files = glob.glob(path_to_beam_skims + "/*")
            agents = len(all_files)
            pool = Pool(processes=agents)
            result = pool.map(read_skim, all_files)
            skims = pd.concat(result, axis=0, ignore_index=True)
    except KeyError:
        raise KeyError(
            "Couldn't find input skims at {0}".format(path_to_beam_skims))
    return skims


def _load_raw_beam_origin_skims(settings):
    """ Read BEAM skims (csv format) from local storage.
    Parameters:
    ------------
    - settings:

    Return:
    --------
    - pandas DataFrame.
    """

    origin_skims_fname = settings.get('origin_skims_fname', False)
    path_to_beam_skims = os.path.join(
        settings['beam_local_output_folder'], origin_skims_fname)
    skims = pd.read_csv(path_to_beam_skims, dtype=beam_origin_skims_types)
    return skims


def _create_skim_object(settings, overwrite=True, output_dir=None):
    """ Creates mutable OMX file to store skim matrices in the beam_output directory.
    This later needs to be copied over to the asim input directory
    Parameters: 
    -----------
    - settings: 
    - overwrite: Default True. To overwrite if existing file. 
    
    Returns: 
    --------
    - True if skim.omx file exist or overwrite is True, False otherwise. 
    
    """
    if output_dir is None:
        output_dir = settings['asim_local_input_folder']
    skims_path = os.path.join(output_dir, 'skims.omx')
    final_skims_exist = os.path.exists(skims_path)

    skims_fname = settings.get('skims_fname', False)
    omx_skim_output = skims_fname.endswith('.omx')
    beam_output_dir = settings['beam_local_output_folder']
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
    mutable_skims_exist = os.path.exists(mutable_skims_location)
    should_use_csv_input_skims = mutable_skims_exist & (not omx_skim_output)

    if final_skims_exist:
        if overwrite:
            logger.info("Found existing skims, removing.")
            os.remove(skims_path)
            return True, should_use_csv_input_skims, False
        else:
            if not mutable_skims_exist:
                shutil.copyfile(skims_path, mutable_skims_location)
            logger.info("Found existing skims, no need to re-create.")
            return False, False, False

    if (~final_skims_exist | overwrite) & omx_skim_output:
        if mutable_skims_exist:
            return True, should_use_csv_input_skims, False
        else:
            logger.info("Creating skims.omx")
            skims = omx.open_file(mutable_skims_location, 'w')
            skims.close()
            return True, should_use_csv_input_skims, True

    logger.exception("We should not be here")
    return False, False, False


def _raw_beam_skims_preprocess(settings, year, skims_df):
    """
    Validates and preprocess raw beam skims.
    
    parameter
    ----------
    - settings:
    - year: 
    - skims_df: pandas Dataframe. Raw beam skims dataframe 
    
    return
    --------
    Pandas dataFrame. Skims
    """
    # Validations:
    origin_taz = skims_df.origin.unique()
    destination_taz = skims_df.destination.unique()
    assert len(origin_taz) == len(destination_taz)

    order = zone_order(settings, year)

    test_1 = set(origin_taz).issubset(set(order))
    test_2 = set(destination_taz).issubset(set(order))
    test_3 = len(set(order) - set(origin_taz))
    test_4 = len(set(order) - set(destination_taz))
    assert test_1, 'There are {} missing origin zone ids in BEAM skims'.format(test_3)
    assert test_2, 'There are {} missing destination zone ids in BEAM skims'.format(test_4)
    # Preprocess skims:
    skims_df.set_index(['timePeriod', 'pathType', 'origin', 'destination'], inplace=True)
    skims_df.loc[:, 'DIST_miles'] = skims_df['DIST_meters'] * (0.621371 / 1000)
    skims_df.loc[:, 'DDIST_miles'] = skims_df['DDIST_meters'] * (0.621371 / 1000)
    skims_df.replace({np.inf: np.nan, 0: np.nan}, inplace=True)  ## TEMPORARY FIX

    inf = np.isinf(skims_df['DDIST_miles']).values.sum() > 0
    zeros = (skims_df['DDIST_miles'] == 0).sum() > 0
    if inf or zeros:
        raise ValueError('Origin-Destination distances contains inf or zero values.')

    return skims_df


def _raw_beam_origin_skims_preprocess(settings, year, origin_skims_df):
    """
    Validates and preprocess raw beam skims.

    parameter
    ----------
    - settings:
    - year:
    - skims_df: pandas Dataframe. Raw beam skims dataframe

    return
    --------
    Pandas dataFrame. Skims
    """
    # Validations:
    origin_taz = origin_skims_df.origin.unique()

    order = zone_order(settings, year)

    #     test_1 = set(origin_taz).issubset(set(order))
    #     test_2 = set(destination_taz).issubset(set(order))
    test_3 = len(set(order) - set(origin_taz))
    assert test_3 == 0, 'There are {} missing origin zone ids in BEAM skims'.format(test_3)
    return origin_skims_df.loc[origin_skims_df['origin'].isin(order)].set_index(['timePeriod',
                                                                                 'reservationType', 'origin'])


def _create_skims_by_mode(settings, skims_df):
    """
    Generates 2 OD pandas dataframe for auto and transit
    
    Parameters:
    ------------
    settings: 
    skims_df: Pandas Dataframe. Clean beam skims. 
    
    Returns:
    ---------
    2 pandas dataframe for auto and transit respectively. 
    """
    logger.info("Splitting BEAM skims by mode.")

    # Settings
    hwy_paths = settings['beam_simulated_hwy_paths']
    transit_paths = settings['transit_paths'].keys()

    logger.info('Splitting out auto skims.')
    # auto_df = skims_df[skims_df['pathType'].isin(hwy_paths)]
    auto_df = skims_df.loc[pd.IndexSlice[:, hwy_paths, :, :], :]
    assert len(auto_df) > 0, 'No auto skims'

    logger.info('Splitting out transit skims.')
    # transit_df = skims_df[skims_df['pathType'].isin(transit_paths)]
    transit_df = skims_df.loc[pd.IndexSlice[:, transit_paths, :, :], :]
    assert len(transit_df) > 0, 'No transit skims'

    del skims_df
    return auto_df, transit_df


def _build_square_matrix(series, num_taz, source="origin", fill_na=0):
    out = np.tile(series.fillna(fill_na).values, (num_taz, 1))
    if source == "origin":
        return out.transpose()
    elif source == "destination":
        return out
    else:
        logger.error("1-d skims must be associated with either 'origin' or 'destination'")


def _build_od_matrix(df, metric, order, fill_na=0.0):
    """ Tranform skims from pandas dataframe to numpy square matrix (O-D matrix format) 
    Parameters: 
    -----------
    - df: Pandas dataframe. 
        Clean skims 
    - origin: str. 
        Name of the column for origin.
    - destiantion:
        Name of the column for destination. 
    - metric:
        Name of the metric that is used to generate the skims. E.g.: "SOV_TIME"
    - order: array
        raw and colum order for the OD metrix. The values in order should be of the same type of 
        the values in the origin and destiantion column. 
    - fill_na: Default = 0. If OD Pair is not represented in df, fill missing value with fill_na. 
    
    Returns:
    ---------
    - numpy square 0-D matrix 
    """
    out = pd.DataFrame(np.nan, index=order, columns=order, dtype=np.float32).rename_axis(index="origin",
                                                                                         columns="destination")
    if metric in df.columns:
        pivot = df[metric].unstack()
        out.loc[pivot.index, pivot.columns] = pivot.fillna(np.nan)
        useDefaults = False
        infs = np.isinf(out)
        if np.any(infs):
            logger.warning("Replacing {0} infs in skim {1}".format(infs.sum().sum(), metric))
            out[infs] = fill_na
    else:
        useDefaults = True

    # vals = df.pivot(index=origin,
    #                 columns=destination,
    #                 values=metric,
    #                 aggfunc='first')

    num_zones = len(order)

    # if (num_zones, num_zones) != vals.shape:
    #
    #     missing_rows = list(set(order) - set(vals.index))  # Missing origins
    #     missing_cols = list(set(order) - set(vals.columns))  # Missing destinations
    #     axis = 0
    #
    #     if len(missing_rows) == 0:
    #         missing_rows = vals.index
    #
    #     if len(missing_cols) == 0:
    #         missing_cols = vals.columns
    #
    #     else:
    #         axis = 1
    #
    #     array = np.empty((len(missing_rows), len(missing_cols)))
    #     array[:] = np.nan
    #     empty_df = pd.DataFrame(array, index=missing_rows, columns=missing_cols)
    #     vals = pd.concat((vals, empty_df), axis=axis)

    assert out.index.isin(order).all(), 'There are missing origins'
    assert out.columns.isin(order).all(), 'There are missing destinations'
    assert (num_zones, num_zones) == out.shape, 'Origin-Destination matrix is not square'

    return out.fillna(fill_na).values, useDefaults


def impute_distances(zones, origin=None, destination=None):
    """
    impute distances in miles for missing OD pairs by calculating 
    the cartesian distance between origin and destination zone. 
    
    Parameters:
    -------------
    - zones: geoPandas or Pandas DataFrame,
        Dataframe with the zones information. If Pandas DataFrame, it expects a geometry column 
        for which wkt.loads can be read. 
    - origin: list, array-like (Optional).
        list of origins. Origins should correspond to the zone identifier in zones. 
        Origin should be the same lenght as destination. 
    - destination: list, array-like (Optional),
        list of destination. Destinations should correspond to the zone identifier in zones
    
    Returns
    --------
    numpy array of shape (len(origin),). Imputed distance for all OD pairs
    """
    if isinstance(zones, pd.core.frame.DataFrame):
        zones.geometry = zones.geometry.astype(str).apply(wkt.loads)
        zones = gpd.GeoDataFrame(zones, geometry='geometry', crs='EPSG:4326')

    assert isinstance(zones, gpd.geodataframe.GeoDataFrame), "Zones needs to be a GeoPandas dataframe"

    gdf = zones.copy()

    # Tranform gdf to CRS in meters
    gdf = gdf.to_crs('EPSG:3857')

    if (origin is None) and (destination is None):
        gdf = gdf.reset_index(drop=True).geometry.centroid
        x, y = gdf.geometry.x.values, gdf.geometry.y.values
        distInMeters = np.linalg.norm([x[:, None] - x[None, :], y[:, None] - y[None, :]], axis=0)
        np.fill_diagonal(distInMeters, 400)
        return distInMeters * (0.621371 / 1000)

    else:
        assert len(origin) == len(destination), 'parameters "origin" and "destination" should have the same lenght'
        origin = (origin + 1).astype(str)
        destination = (destination + 1).astype(str)

        # Select origin and destination pairs
        orig = gdf.loc[origin].reset_index(drop=True).geometry.centroid
        dest = gdf.loc[destination].reset_index(drop=True).geometry.centroid

        # Distance in Miles

        return orig.distance(dest).replace({0: 100}).values * (0.621371 / 1000)


def _distance_skims(settings, year, input_skims, order, data_dir=None):
    """
    Generates distance matrices for drive, walk and bike modes.
    Parameters:
    - settings:
    - year:
    - auto_df: pandas.DataFrame
        Dataframe with driving modes only. 
    - order: numpy.array
        zone_id order to create the num_zones x num_zones skim matrix.
    """
    logger.info("Creating distance skims.")

    skims_fname = settings.get('skims_fname', False)
    beam_output_dir = settings['beam_local_output_folder']
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
    needToClose = True
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode='a')

    # TO DO: Include walk and bike distances,
    # for now walk and bike are the same as drive.
    dist_column = settings['beam_asim_hwy_measure_map']['DIST']
    if isinstance(input_skims, pd.DataFrame):
        dist_df = input_skims[[dist_column]].groupby(level=[2, 3]).agg('first')
        mx_dist, useDefaults = _build_od_matrix(dist_df, dist_column, order, fill_na=np.nan)
        if useDefaults:
            logger.warning(
                "Filling in default skim values for measure {0} because they're not in BEAM outputs".format(
                    dist_column))
    elif isinstance(input_skims, omx.File):
        if 'DIST' in input_skims.list_matrices():
            mx_dist = np.array(input_skims['DIST'], dtype=np.float32)
        else:
            mx_dist = np.full((len(order), len(order)), np.nan, dtype=np.float32)
    else:
        mx_dist = np.full((len(order), len(order)), np.nan, dtype=np.float32)
    # Impute missing distances 
    missing = np.isnan(mx_dist)
    if missing.all():
        logger.info("Imputing all missing distance skims.")
        zones = read_zone_geoms(settings, year)
        mx_dist = impute_distances(zones)
        output_skims['DIST'] = mx_dist
    elif missing.any():
        orig, dest = np.where(missing == True)
        logger.info("Imputing {} missing distance skims.".format(len(orig)))
        zones = read_zone_geoms(settings, year)
        imputed_dist = impute_distances(zones, orig, dest)
        mx_dist[orig, dest] = imputed_dist
        output_skims['DIST'] = mx_dist
    else:
        logger.info("No need to impute missing distance skims.")
        mx_dist = np.array(input_skims['DIST'])

    # Distance matrices
    if 'DISTBIKE' not in output_skims.list_matrices():
        output_skims['DISTBIKE'] = mx_dist
    if 'DISTWALK' not in output_skims.list_matrices():
        output_skims['DISTWALK'] = mx_dist
    if 'BIKE_TRIPS' not in output_skims.list_matrices():
        output_skims['BIKE_TRIPS'] = np.zeros_like(mx_dist)
    if 'WALK_TRIPS' not in output_skims.list_matrices():
        output_skims['WALK_TRIPS'] = np.zeros_like(mx_dist)

    if needToClose:
        output_skims.close()


def _build_od_matrix_parallel(tup):
    df, measure_map, num_taz, order, fill_na = tup
    out = dict()
    for measure in measure_map.keys():
        if len(df.index) == 0:
            mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
            useDefaults = True
        elif (measure == 'FAR') or (measure == 'BOARDS'):
            mtx, useDefaults = _build_od_matrix(df, measure_map[measure], order, fill_na)
        elif measure_map[measure] in df.columns:
            # activitysim estimated its models using transit skims from Cube
            # which store time values as scaled integers (e.g. x100), so their
            # models also divide transit skim values by 100. Since our skims
            # aren't coming out of Cube, we multiply by 100 to negate the division.
            # This only applies for travel times.
            mtx, useDefaults = _build_od_matrix(df, measure_map[measure], order, fill_na)

        else:
            mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
        out[measure] = mtx
    return out


def _transit_skims(settings, transit_df, order, data_dir=None):
    """ Generate transit OMX skims"""

    logger.info("Creating transit skims.")
    transit_paths = settings['transit_paths']
    periods = settings['periods']
    measure_map = settings['beam_asim_transit_measure_map']
    skims = read_skims(settings, mode='a', data_dir=data_dir)
    num_taz = len(order)
    fill_na = 0.0

    groupBy = transit_df.groupby(level=[0, 1])

    with Pool(cpu_count() - 1) as p:
        ret_list = p.map(_build_od_matrix_parallel,
                         [(group.loc[name], measure_map, num_taz, order, fill_na) for name, group in groupBy])

    resultsDict = dict()

    for (period, path), processedDict in zip(groupBy.groups.keys(), ret_list):
        for measure, mtx in processedDict.items():
            name = '{0}_{1}__{2}'.format(path, measure, period)
            resultsDict[name] = mtx

    for path, measures in transit_paths.items():
        for period in periods:
            for measure in measures:
                name = '{0}_{1}__{2}'.format(path, measure, period)
                if name in resultsDict:
                    mtx = resultsDict[name]
                else:
                    logger.warning(
                        "Filling in default skim values for measure {0} because they're not in BEAM outputs".format(
                            name))
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
                if np.any(np.isinf(mtx)):
                    logger.warning("Replacing {0} infs in skim {1}".format(np.isinf(mtx).sum().sum(), name))
                    mtx[np.isinf(mtx)] = np.nan
                if (measure == 'FAR') or (measure == 'BOARDS'):
                    skims[name] = mtx
                else:
                    skims[name] = mtx * 100
    skims.close()


def _ridehail_skims(settings, ridehail_df, order, data_dir=None):
    """ Generate transit OMX skims"""

    logger.info("Creating ridehail skims.")
    ridehail_path_map = settings['ridehail_path_map']
    periods = settings['periods']
    measure_map = settings['beam_asim_ridehail_measure_map']
    skims = read_skims(settings, mode='a', data_dir=data_dir)
    num_taz = len(order)
    df = ridehail_df.copy()

    for path, skimPath in ridehail_path_map.items():
        for period in periods:
            df_ = df.loc[(period, skimPath), :].loc[order, :]
            for measure, skimMeasure in measure_map.items():
                name = '{0}_{1}__{2}'.format(path, measure, period)
                if measure == 'REJECTIONPROB':
                    mtx = _build_square_matrix(df_[skimMeasure], num_taz, 'origin', 0.0)
                elif measure_map[measure] in df_.columns:
                    # activitysim estimated its models using transit skims from Cube
                    # which store time values as scaled integers (e.g. x100), so their
                    # models also divide transit skim values by 100. Since our skims
                    # aren't coming out of Cube, we multiply by 100 to negate the division.
                    # This only applies for travel times.
                    # EDIT: I don't think this is true for wait time
                    mtx = _build_square_matrix(df_[skimMeasure], num_taz, 'origin', 0.0)

                else:
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
                skims[name] = mtx
    skims.close()
    del df, df_


def _get_field_or_else_empty(skims: Optional[omx.File], field: str, num_taz: int):
    if skims is None:
        return np.full((num_taz, num_taz), np.nan, dtype=np.float32), True
    if field in skims.list_matrices():
        return np.array(skims[field], dtype=np.float32), False
    else:
        return np.full((num_taz, num_taz), np.nan, dtype=np.float32), True


def _fill_ridehail_skims(settings, input_skims, order, data_dir=None):
    logger.info("Merging ridehail omx skims.")

    ridehail_path_map = settings['ridehail_path_map']
    periods = settings['periods']
    measure_map = settings['beam_asim_ridehail_measure_map']

    num_taz = len(order)

    skims_fname = settings.get('skims_fname', False)
    beam_output_dir = settings['beam_local_output_folder']
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
    needToClose = True
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode='a')

    output_skim_tables = output_skims.list_matrices()

    # NOTE: time is in units of minutes
    for path, skimPath in ridehail_path_map.items():
        logger.info("Writing tables for path type {0}".format(path))
        for period in periods:
            completed_measure = '{0}_{1}__{2}'.format(path, "TRIPS", period)
            failed_measure = '{0}_{1}__{2}'.format(path, "FAILURES", period)

            temp_completed, createdTemp = _get_field_or_else_empty(output_skims, completed_measure, num_taz)
            temp_failed, createdTemp = _get_field_or_else_empty(output_skims, failed_measure, num_taz)

            if createdTemp:
                output_skims[completed_measure] = np.nan_to_num(np.array(temp_completed))
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period

            for measure in measure_map.keys():
                name = '{0}_{1}__{2}'.format(path, measure, period)
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    temp, createdTemp = _get_field_or_else_empty(input_skims, name, num_taz)
                    missing_values = np.isnan(temp)
                    if measure == "WAIT":
                        temp[missing_values] = 6.0
                    elif measure == "REJECTIONPROB":
                        temp[missing_values] = 0.2

                    if not inOutputSkims:
                        output_skims[name] = temp
                        output_skims[name].attrs.mode = path
                        output_skims[name].attrs.measure = measure
                        output_skims[name].attrs.timePeriod = period
                    else:
                        if np.any(missing_values):
                            output_skims[name][:] = temp
    if needToClose:
        output_skims.close()


def _fill_transit_skims(settings, input_skims, order, data_dir=None):
    logger.info("Merging transit omx skims.")

    transit_paths = settings['transit_paths']
    periods = settings['periods']
    measure_map = settings['beam_asim_transit_measure_map']

    num_taz = len(order)

    skims_fname = settings.get('skims_fname', False)
    beam_output_dir = settings['beam_local_output_folder']
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
    needToClose = True
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode='a')

    distance_miles = np.array(output_skims['DIST'], dtype=np.float32)
    output_skim_tables = output_skims.list_matrices()

    # NOTE: time is in units of minutes

    for path, measures in transit_paths.items():
        logger.info("Writing tables for path type {0}".format(path))
        for period in periods:
            completed_measure = '{0}_{1}__{2}'.format(path, "TRIPS", period)
            failed_measure = '{0}_{1}__{2}'.format(path, "FAILURES", period)

            temp_completed, createdTemp = _get_field_or_else_empty(output_skims, completed_measure, num_taz)
            temp_failed, createdTemp = _get_field_or_else_empty(output_skims, failed_measure, num_taz)

            if createdTemp:
                output_skims[completed_measure] = np.nan_to_num(np.array(temp_completed))
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period

            # tooManyFailures = temp_failed > temp_completed
            for measure in measures:
                name = '{0}_{1}__{2}'.format(path, measure, period)
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    temp, createdTemp = _get_field_or_else_empty(input_skims, name, num_taz)
                    missing_values = np.isnan(temp)
                    if np.any(missing_values):
                        if (measure == "IVT") | (measure == "TOTIVT") | (measure == "KEYIVT"):
                            temp[missing_values] = distance_miles[
                                                       missing_values] / default_speed_mph.get(
                                path.split("_")[1], 10.0) * 60 * 100  # Assume 20 mph average, with 100x multiplier
                            # temp[tooManyFailures] = 0
                        elif measure == "DIST":
                            temp[missing_values] = distance_miles[missing_values]
                        elif (measure == "WAIT") | (measure == "WACC") | (measure == "WEGR") | (measure == "IWAIT"):
                            temp[missing_values] = 1000.0
                        elif measure == "FAR":
                            temp[missing_values] = default_fare_dollars.get(path.split("_")[1], 4.0)
                        elif measure == "BOARDS":
                            temp[missing_values] = 2.0
                        elif (measure == "DDIST") & (path.endswith("DRV") | path.startswith("DRV")):
                            temp[missing_values] = 2.0
                        elif (measure == "DTIME") & (path.endswith("DRV") | path.startswith("DRV")):
                            temp[missing_values] = 1500.0
                        else:
                            temp[missing_values] = 0.0

                    output_skims[name] = temp
                    output_skims[name].attrs.mode = path
                    output_skims[name].attrs.measure = measure
                    output_skims[name].attrs.timePeriod = period

    if needToClose:
        output_skims.close()


def _fill_auto_skims(settings, input_skims, order, data_dir=None):
    logger.info("Merging drive omx skims.")

    # Open skims object
    periods = settings['periods']
    paths = settings['hwy_paths']
    measure_map = settings['beam_asim_hwy_measure_map']

    num_taz = len(order)

    skims_fname = settings.get('skims_fname', False)
    beam_output_dir = settings['beam_local_output_folder']
    mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
    needToClose = True
    if input_skims is not None:
        output_skims = input_skims
        needToClose = False
    else:
        output_skims = omx.open_file(mutable_skims_location, mode='a')

    distance_miles = np.array(output_skims['DIST'])
    output_skim_tables = output_skims.list_matrices()
    nSkimsCreated = 0

    # NOTE: time is in units of minutes

    for path in paths:
        logger.info("Writing tables for path type {0}".format(path))
        for period in periods:
            completed_measure = '{0}_{1}__{2}'.format(path, "TRIPS", period)
            failed_measure = '{0}_{1}__{2}'.format(path, "FAILURES", period)

            temp_completed, createdTemp = _get_field_or_else_empty(output_skims, completed_measure, num_taz)
            temp_failed, createdTemp = _get_field_or_else_empty(output_skims, failed_measure, num_taz)

            if createdTemp:
                nSkimsCreated += 1
                output_skims[completed_measure] = np.nan_to_num(np.array(temp_completed))
                output_skims[failed_measure] = np.nan_to_num(np.array(temp_failed))
                output_skims[completed_measure].attrs.mode = path
                output_skims[failed_measure].attrs.mode = path
                output_skims[completed_measure].attrs.measure = "TRIPS"
                output_skims[failed_measure].attrs.measure = "FAILURES"
                output_skims[completed_measure].attrs.timePeriod = period
                output_skims[failed_measure].attrs.timePeriod = period
            for measure in measure_map.keys():
                name = '{0}_{1}__{2}'.format(path, measure, period)
                inOutputSkims = name in output_skim_tables
                if not inOutputSkims:
                    nSkimsCreated += 1
                    temp, createdTemp = _get_field_or_else_empty(input_skims, name, num_taz)
                    missing_values = np.isnan(temp)
                    if np.any(missing_values):
                        if measure == "TIME":
                            temp[missing_values] = distance_miles[missing_values] / 35 * 60  # Assume 35 mph average
                        elif measure == "DIST":
                            temp[missing_values] = distance_miles[missing_values]
                        elif measure == "BTOLL":
                            temp[missing_values] = 0.0
                        elif measure == "VTOLL":
                            if path.endswith('TOLL'):
                                toll = 5.0
                            else:
                                toll = 0.0
                            temp[missing_values] = toll
                        else:
                            logger.error("Trying to read unknown measure {0} from BEAM skims".format(measure))

                    output_skims[name] = temp
                    output_skims[name].attrs.mode = path
                    output_skims[name].attrs.measure = measure
                    output_skims[name].attrs.timePeriod = period
    logger.info("Created {0} new skims in the omx object".format(nSkimsCreated))
    if needToClose:
        output_skims.close()


def _auto_skims(settings, auto_df, order, data_dir=None):
    logger.info("Creating drive skims.")

    # Open skims object
    periods = settings['periods']
    paths = settings['hwy_paths']
    measure_map = settings['beam_asim_hwy_measure_map']
    skims = read_skims(settings, mode='a', data_dir=data_dir)
    num_taz = len(order)
    beam_hwy_paths = settings['beam_simulated_hwy_paths']
    fill_na = np.nan

    groupBy = auto_df.groupby(level=[0, 1])

    with Pool(cpu_count() - 1) as p:
        ret_list = p.map(_build_od_matrix_parallel,
                         [(group.loc[name], measure_map, num_taz, order, fill_na) for name, group in groupBy])

    resultsDict = dict()

    for (period, path), processedDict in zip(groupBy.groups.keys(), ret_list):
        if path == "SOV":
            for measure, mtx in processedDict.items():
                for path_ in paths:
                    name = '{0}_{1}__{2}'.format(path_, measure, period)
                    resultsDict[name] = mtx

    for period in periods:
        for path in paths:
            for measure in measure_map.keys():
                name = '{0}_{1}__{2}'.format(path, measure, period)
                if name in resultsDict:
                    mtx = resultsDict[name]
                    missing = np.isnan(mtx)

                    if missing.any():
                        distances = np.array(skims['DIST'])
                        orig, dest = np.where(missing == True)
                        missing_measure = distances[orig, dest]

                        if measure == 'DIST':
                            mtx[orig, dest] = missing_measure
                        elif measure == 'TIME':
                            mtx[orig, dest] = missing_measure * (60 / 40)  # Assumes average speed of 40 miles/hour
                        else:
                            mtx[orig, dest] = 0  ## Assumes no toll or payment
                else:
                    mtx = np.zeros((num_taz, num_taz), dtype=np.float32)
                    logger.warning(
                        "Filling in default skim values for measure {0} because they're not in BEAM outputs".format(
                            name))
                # if ('TOLL' in path) & ('TOLL' in measure):
                #     # For now penalize toll routes because we don't simulate them
                #     mtx.fill(1.0e7)
                if np.any(np.isinf(mtx)):
                    logger.warning("Replacing {0} infs in skim {1}".format(np.isinf(mtx).sum().sum(), name))
                    mtx[np.isinf(mtx)] = np.nan
                skims[name] = mtx
    skims.close()


def _create_offset(settings, order, data_dir=None):
    logger.info("Creating skims offset keys")

    # Open skims object
    skims = read_skims(settings, mode='a', data_dir=data_dir)
    zone_id = np.arange(1, len(order) + 1)

    # Generint offset
    skims.create_mapping('zone_id', zone_id, overwrite=True)
    skims.close()


def create_skims_from_beam(settings, year,
                           output_dir=None,
                           overwrite=True):
    if not output_dir:
        output_dir = settings['asim_local_input_folder']

    # If running in static skims mode and ActivitySim skims already exist
    # there is no point in recreating them.
    static_skims = settings.get('static_skims', False)
    if static_skims:
        overwrite = False

    new, convertFromCsv, blankSkims = _create_skim_object(settings, overwrite, output_dir=output_dir)
    validation = settings.get('asim_validation', False)

    order = zone_order(settings, year)

    if new:
        tempSkims = _load_raw_beam_skims(settings, convertFromCsv, blankSkims)
        if isinstance(tempSkims, pd.DataFrame):
            skims = tempSkims.loc[tempSkims.origin.isin(order) & tempSkims.destination.isin(order), :]
            skims = _raw_beam_skims_preprocess(settings, year, skims)
            auto_df, transit_df = _create_skims_by_mode(settings, skims)
            ridehail_df = _load_raw_beam_origin_skims(settings)
            ridehail_df = _raw_beam_origin_skims_preprocess(settings, year, ridehail_df)

            # Create skims
            _distance_skims(settings, year, auto_df, order, data_dir=output_dir)
            _auto_skims(settings, auto_df, order, data_dir=output_dir)
            _transit_skims(settings, transit_df, order, data_dir=output_dir)
            _ridehail_skims(settings, ridehail_df, order, data_dir=output_dir)

            del auto_df, transit_df
        else:
            beam_output_dir = settings['beam_local_output_folder']
            _distance_skims(settings, year, tempSkims, order, data_dir=beam_output_dir)
            _fill_auto_skims(settings, tempSkims, order, data_dir=beam_output_dir)
            _fill_transit_skims(settings, tempSkims, order, data_dir=beam_output_dir)
            _fill_ridehail_skims(settings, tempSkims, order, data_dir=beam_output_dir)
            if isinstance(tempSkims, omx.File):
                tempSkims.close()
            final_skims_path = os.path.join(settings['asim_local_input_folder'], 'skims.omx')
            skims_fname = settings.get('skims_fname', False)
            mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
            shutil.copyfile(mutable_skims_location, final_skims_path)

    _create_offset(settings, order, data_dir=output_dir)

    if validation:
        order = zone_order(settings, year)
        skim_validations(settings, year, order, data_dir=output_dir)


def plot_skims(settings, zones,
               skims, order,
               random_sample=6,
               cols=3, name='DIST',
               units='in miles'):
    """
    Plot a map of skims for a random set zones to all other zones. For validation/debugging purposes. 
    
    Parameters:
    - settings:
    - zones : geopandas dataframe
    - skims : numpy array
        Skim measure. num_zone x num_zone ndarray. 
    - order : numpy array 
    - random_sample : int
        number of zone to plot the skims
    - cols : int
        number of columns in the resulting subplot. 
    - name : str
        name of the skim measure
    - units : str
        Unit of analysis of the skim measure
    """
    random_sample = random_sample
    cols = cols
    rows = int(random_sample / cols)
    zone_ids = list(zones.sample(random_sample).index.astype(int))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 20))

    counter = 0
    for row in range(rows):
        for col in range(cols):

            zone_id = int(zone_ids[counter])
            name_ = name + '_zone_id_' + str(zone_id)
            zone_measure = skims[zone_id - 1, :]
            empty = zone_measure.sum() == 0
            while empty:
                zone_id = int(list(zones.sample(1).index)[0])
                name_ = name + '_zone_id_' + str(zone_id)
                zone_measure = skims[zone_id - 1, :]
                empty = zone_measure.sum() == 0
            zones.loc[:, name_] = zone_measure
            zones.loc[:, name_] = zones.loc[:, name_].replace({999: np.nan, 0: np.nan})
            counter += 1
            bg_id = order[zone_id - 1]

            zones.plot(column=name_, legend=True, ax=axs[row][col])
            axs[row][col].set_title('{0} ({1}) from zone {2} \n block_group {3} '.format(name, units, zone_id, bg_id))

    # Saving plots to files.
    asim_validation = settings['asim_validation_folder']
    if not os.path.isdir(asim_validation):
        os.mkdir(asim_validation)

    save_path = os.path.join(asim_validation, 'skims_validation_' + name + '.pdf')
    fig.savefig(save_path)


def skim_validations(settings, year, order, data_dir=None):
    logger.info("Generating skims validation plots.")
    skims = read_skims(settings, mode='r', data_dir=data_dir)
    zone = read_zone_geoms(settings, year,
                           asim_zone_id_col='TAZ',
                           default_zone_id_col='zone_id')

    # Skims matrices 
    num_zones = len(order)
    distances = np.array(skims['DIST'])
    sov_time = np.array(skims['SOV_TIME__AM'])
    loc_time_list = ['WLK_LOC_WLK_TOTIVT__AM', 'WLK_LOC_WLK_XWAIT__AM',
                     'WLK_LOC_WLK_IWAIT__AM',
                     'WLK_LOC_WLK_WAUX__AM']
    PuT_time = np.zeros((num_zones, num_zones), dtype=np.float32)
    for measure in loc_time_list:
        time = np.array(skims[measure]) / 100
        PuT_time = PuT_time + time
    PuT_time[np.array(skims['WLK_LOC_WLK_TRIPS__AM']) == 0] = np.nan

    # Plots
    plot_skims(settings, zone, distances, order, 6, 3, 'DIST', 'in miles')
    plot_skims(settings, zone, sov_time, order, 6, 3, 'SOV_TIME', 'in minutes')
    plot_skims(settings, zone, PuT_time, order, 6, 3, 'WLK_LOC_WLK_TIME', 'in minutes')


#######################################
#### UrbanSim to ActivitySim tables ###
#######################################
def _get_full_time_enrollment(state_fips, year):
    base_url = (
        'https://educationdata.urban.org/api/v1/'
        '{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}')
    levels = ['undergraduate', 'graduate']
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t='college-university', so='ipeds', e='fall-enrollment',
            y=year, l=level, f='ftpt=1', s='sex=99',
            r='race=99', cl='class_level=99', ds='degree_seeking=99',
            fips='fips={0}'.format(state_fips))
        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(
            columns={'enrollment_fall': level})
        enroll[level].clip(0, inplace=True)
        enroll.set_index('unitid', inplace=True)
        enroll_list.append(enroll)
    full_time = pd.concat(enroll_list, axis=1).fillna(0)
    full_time['full_time'] = full_time['undergraduate'] + full_time['graduate']
    s = full_time.full_time
    assert s.index.name == 'unitid'
    return s


def _get_part_time_enrollment(state_fips):
    base_url = (
        'https://educationdata.urban.org/api/v1/'
        '{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}')
    levels = ['undergraduate', 'graduate']
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t='college-university', so='ipeds', e='fall-enrollment',
            y='2015', l=level, f='ftpt=2', s='sex=99',
            r='race=99', cl='class_level=99', ds='degree_seeking=99',
            fips='fips={0}'.format(state_fips))

        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(
            columns={'enrollment_fall': level})
        enroll[level].clip(0, inplace=True)
        enroll.set_index('unitid', inplace=True)
        enroll_list.append(enroll)

    part_time = pd.concat(enroll_list, axis=1).fillna(0)
    part_time['part_time'] = part_time['undergraduate'] + part_time['graduate']
    s = part_time.part_time
    assert s.index.name == 'unitid'
    return s


def _update_persons_table(persons, households, unassigned_households, blocks, asim_zone_id_col='TAZ'):
    # assign zones
    persons.index = persons.index.astype(int)
    unassigned_persons = persons.household_id.isin(unassigned_households)
    logger.info("Dropping {0} people from {1} households that haven't been assigned locations yet".format(
        unassigned_households.shape[0], unassigned_persons.sum()))
    persons = persons.loc[~unassigned_persons, :]
    persons.loc[:, asim_zone_id_col] = blocks[asim_zone_id_col].reindex(
        households['block_id'].reindex(persons['household_id']).values).values
    persons.loc[:, asim_zone_id_col] = persons.loc[:, asim_zone_id_col].astype(str)

    # create new column variables
    age_mask_1 = persons.age >= 18
    age_mask_2 = persons.age.between(18, 64, inclusive=True)
    age_mask_3 = persons.age >= 65
    work_mask = persons.worker == 1
    student_mask = persons.student == 1
    type_1 = ((age_mask_1) & (work_mask) & (~student_mask)) * 1  # Full time
    type_4 = ((age_mask_2) & (~work_mask) & (~student_mask)) * 4
    type_5 = ((age_mask_3) & (~work_mask) & (~student_mask)) * 5
    type_3 = ((age_mask_1) & (student_mask)) * 3
    type_6 = (persons.age.between(16, 17, inclusive=True)) * 6
    type_7 = (persons.age.between(6, 16, inclusive=True)) * 7
    type_8 = (persons.age.between(0, 5, inclusive=True)) * 8
    type_list = [
        type_1, type_3, type_4, type_5, type_6, type_7, type_8]
    for x in type_list:
        type_1.where(type_1 != 0, x, inplace=True)
    persons.loc[:, 'ptype'] = type_1

    pemploy_1 = ((persons.worker == 1) & (persons.age >= 16)) * 1
    pemploy_3 = ((persons.worker == 0) & (persons.age >= 16)) * 3
    pemploy_4 = (persons.age < 16) * 4
    type_list = [pemploy_1, pemploy_3, pemploy_4]
    for x in type_list:
        pemploy_1.where(pemploy_1 != 0, x, inplace=True)
    persons.loc[:, 'pemploy'] = pemploy_1

    pstudent_1 = (persons.age <= 18) * 1
    pstudent_2 = ((persons.student == 1) & (persons.age > 18)) * 2
    pstudent_3 = (persons.student == 0) * 3
    type_list = [pstudent_1, pstudent_2, pstudent_3]
    for x in type_list:
        pstudent_1.where(pstudent_1 != 0, x, inplace=True)
    persons.loc[:, 'pstudent'] = pstudent_1

    persons_w_res_blk = pd.merge(
        persons, households[['block_id']],
        left_on='household_id', right_index=True)
    persons_w_xy = pd.merge(
        persons_w_res_blk, blocks[['x', 'y']],
        left_on='block_id', right_index=True)
    persons.loc[:, 'home_x'] = persons_w_xy['x']
    persons.loc[:, 'home_y'] = persons_w_xy['y']

    del persons_w_res_blk
    del persons_w_xy

    persons.loc[:, "workplace_taz"] = pd.to_numeric(persons.loc[:, "work_zone_id"].copy(), errors='coerce').fillna(-1)
    persons.loc[:, "school_taz"] = pd.to_numeric(persons.loc[:, "school_zone_id"].copy(), errors='coerce').fillna(-1)
    persons.loc[:, "worker"] = pd.to_numeric(persons.loc[:, "worker"].copy(), errors='coerce').fillna(0)
    persons.loc[:, "student"] = pd.to_numeric(persons.loc[:, "student"].copy(), errors='coerce').fillna(0)

    # clean up dataframe structure
    # TODO: move this to annotate_persons.yaml in asim settings
    #     p_names_dict = {'member_id': 'PNUM'}
    #     persons = persons.rename(columns=p_names_dict)

    p_null_taz = persons[asim_zone_id_col].isnull()
    logger.info("Dropping {0} persons without TAZs".format(
        p_null_taz.sum()))
    p_newborn = persons['age'] < 1.0
    logger.info("Dropping {0} newborns from this iteration".format(
        p_newborn.sum()))
    p_badwork = (persons.worker == 1) & ~(persons.workplace_taz >= 0)
    p_badschool = (persons.student == 1) & ~(persons.school_taz >= 0)
    logger.warn(
        "Dropping {0} workers with undefined workplace and {1} students with undefined school".format(p_badwork.sum(),
                                                                                                      p_badschool.sum()))
    persons = persons.loc[(~p_null_taz) & (~p_newborn) & (~p_badwork) & (~p_badschool), :]
    persons = persons.dropna()
    persons.loc[:, 'member_id'] = persons.groupby('household_id')['member_id'].apply(np.argsort) + 1

    # Workers with school ID
    workers_with_school_id = persons[(persons.ptype == 1) & (persons.school_zone_id >= 0)].shape
    persons.loc[persons['ptype'] == 1, 'school_zone_id'] = -1
    persons.loc[persons['ptype'] == 1, 'school_taz'] = -1
    workers_with_school_id_post = persons[(persons.ptype == 1) & (persons.school_zone_id >= 0)].shape

    logger.info(f"Workers with School location: {workers_with_school_id[0]}")
    logger.info(f"Workers with School location after cleaning: {workers_with_school_id_post[0]}")

    # Make Sure non-workers and non-students dont't have a school location
    non_work_school_with_school_id = persons[(persons.ptype.isin([4, 5])) & (persons.school_zone_id > 0)].shape
    persons.loc[persons.ptype.isin([4, 5]), 'school_zone_id'] = -1
    persons.loc[persons.ptype.isin([4, 5]), 'school_taz'] = -1
    non_work_school_with_school_id_post = persons[(persons.ptype.isin([4, 5])) & (persons.school_zone_id > 0)].shape

    logger.info(f"Non-Workers and non-students with School location: {non_work_school_with_school_id[0]}")
    logger.info(
        f"Non-Workers and non-students with School location after cleaning: {non_work_school_with_school_id_post[0]}")

    # Make Sure non-workers and non-students dont't have a work location
    non_work_school_with_work_id = persons[(persons.ptype.isin([4, 5])) & (persons.work_zone_id > 0)].shape
    persons.loc[persons.ptype.isin([4, 5]), 'work_zone_id'] = -1
    persons.loc[persons.ptype.isin([4, 5]), 'workplace_taz'] = -1
    non_work_school_with_work_id_post = persons[(persons.ptype.isin([4, 5])) & (persons.school_zone_id > 0)].shape

    logger.info(f"Non-Workers and non-students with Work location: {non_work_school_with_work_id[0]}")
    logger.info(
        f"Non-Workers and non-students with Work location after cleaning: {non_work_school_with_work_id_post[0]}")

    assert (persons['work_zone_id'] == persons['workplace_taz']).all()
    assert (persons['school_zone_id'] == persons['school_taz']).all()

    return persons


def _update_households_table(households, blocks, asim_zone_id_col='TAZ'):
    # assign zones
    households.index = households.index.astype(int)
    households[asim_zone_id_col] = blocks[asim_zone_id_col].reindex(
        households['block_id']).values

    hh_null_taz = (~(households[asim_zone_id_col].astype("Int64") > 0)).fillna(True)

    households[asim_zone_id_col] = households[asim_zone_id_col].astype(str)
    logger.info('Dropping {0} households without TAZs'.format(
        hh_null_taz.sum()))
    hh_null_taz_id = households.index[hh_null_taz]
    households = households[~hh_null_taz]

    # create new column variables
    s = households.persons
    households.loc[:, 'HHT'] = s.where(s == 1, 4)

    # clean up dataframe structure
    # TODO: move this to annotate_households.yaml in asim settings
    #     hh_names_dict = {
    #         'persons': 'PERSONS',
    #         'cars': 'VEHICL'}
    #     households = households.rename(columns=hh_names_dict)
    if 'household_id' in households.columns:
        households.set_index('household_id', inplace=True)
    else:
        households.index.name = 'household_id'

    return households, hh_null_taz_id


def _update_jobs_table(
        jobs, blocks, state_fips, county_codes, local_crs,
        asim_zone_id_col='TAZ'):
    # assign zones
    jobs[asim_zone_id_col] = blocks[asim_zone_id_col].reindex(
        jobs['block_id']).values

    jobs[asim_zone_id_col] = jobs[asim_zone_id_col].astype(str)

    # make sure jobs are only assigned to blocks with land area > 0
    # so that employment density distributions don't contain Inf/NaN
    blocks = blocks[['square_meters_land']]
    jobs['square_meters_land'] = blocks.reindex(
        jobs['block_id'])['square_meters_land'].values
    jobs_w_no_land = jobs[jobs['square_meters_land'] == 0]
    blocks_to_reassign = jobs_w_no_land['block_id'].unique()
    num_reassigned = len(blocks_to_reassign)

    if num_reassigned > 0:

        logger.info("Reassigning jobs out of blocks with no land area!")
        blocks_gdf = get_block_geoms(state_fips, county_codes)
        blocks_gdf.set_index('GEOID', inplace=True)
        blocks_gdf['square_meters_land'] = blocks[
            'square_meters_land'].reindex(blocks_gdf.index)
        blocks_gdf = blocks_gdf.to_crs(local_crs)

        for block_id in tqdm(
                blocks_to_reassign,
                desc="Redistributing jobs from blocks:"):
            candidate_mask = (
                                     blocks_gdf.index.values != block_id) & (
                                     blocks_gdf['square_meters_land'] > 0)
            new_block_id = blocks_gdf[candidate_mask].distance(
                blocks_gdf.loc[block_id, 'geometry']).idxmin()

            jobs.loc[
                jobs['block_id'] == block_id, 'block_id'] = new_block_id

    else:
        logger.info("No block IDs to reassign in the jobs table!")

    return num_reassigned, jobs


def _update_blocks_table(settings, year, blocks,
                         households, jobs, zone_id_col):
    blocks['TOTEMP'] = jobs[['block_id', 'sector_id']].groupby(
        'block_id')['sector_id'].count().reindex(blocks.index).fillna(0)

    blocks['TOTPOP'] = households[['block_id', 'persons']].groupby(
        'block_id')['persons'].sum().reindex(blocks.index).fillna(0)

    blocks['TOTACRE'] = blocks['square_meters_land'] / 4046.86

    # update blocks (should only have to be run if asim is loading
    # raw urbansim data that has yet to be touched by pilates)
    geoid_to_zone_mapping_updated = False

    zone_type = settings['skims_zone_type']
    zone_id_col = "{}_{}".format(zone_type, zone_id_col)

    if zone_id_col not in blocks.columns:

        mapping = geoid_to_zone_map(settings, year)

        if zone_type == 'block':
            logger.info("Mapping block IDs")
            blocks[zone_id_col] = blocks.index.astype(str).replace(mapping)

        elif zone_type == 'block_group':
            logger.info("Mapping blocks to block group IDS")
            blocks[zone_id_col] = blocks.block_group_id.astype(str).replace(
                mapping)

        elif zone_type == 'taz':
            logger.info("Mapping block IDs to TAZ")
            blocks[zone_id_col] = blocks.index.astype(str)
            blocks[zone_id_col] = blocks[zone_id_col].replace(mapping)

        geoid_to_zone_mapping_updated = True

    else:
        logger.info(
            "Blocks table already has zone IDs. Make sure skim zones "
            "haven't changed.")

    blocks[zone_id_col] = blocks[zone_id_col].astype(str)

    return geoid_to_zone_mapping_updated, blocks


def _get_school_enrollment(state_fips, county_codes):
    logger.info(
        "Downloading school enrollment data from educationdata.urban.org!")
    base_url = 'https://educationdata.urban.org/api/v1/' + \
               '{topic}/{source}/{endpoint}/{year}/?{filters}'

    # at the moment you can't seem to filter results by county
    enroll_filters = 'fips={0}'.format(state_fips)
    enroll_url = base_url.format(
        topic='schools', source='ccd', endpoint='directory',
        year='2015', filters=enroll_filters)

    school_tables = []
    while True:
        response = requests.get(enroll_url).json()
        count = response['count']
        next_page = response['next']
        data = response['results']
        enroll = pd.DataFrame(data)
        school_tables.append(enroll)
        if next_page is not None:
            enroll_url = next_page
            time.sleep(2)
        else:
            break

    enrollment = pd.concat(school_tables, axis=0)
    assert len(enrollment) == count
    enrollment = enrollment[[
        'ncessch', 'county_code', 'latitude',
        'longitude', 'enrollment']].set_index('ncessch')
    enrollment['county_code'] = enrollment['county_code'].str[-3:]
    enrollment = enrollment[enrollment['county_code'].isin(county_codes)]
    enrollment.rename(
        columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)
    enrollment['enrollment'].clip(0, inplace=True)
    enrollment = enrollment[~enrollment.enrollment.isna()]

    return enrollment


def _get_college_enrollment(state_fips, county_codes):
    year = '2015'
    logger.info("Downloading college data from educationdata.urban.org!")
    base_url = 'https://educationdata.urban.org/api/v1/' + \
               '{topic}/{source}/{endpoint}/{year}/?{filters}'

    colleges_list = []
    total_count = 0
    for county in county_codes:
        county_fips = str(state_fips) + str(county)
        college_filters = 'county_fips={0}'.format(county_fips)
        college_url = base_url.format(
            topic='college-university', source='ipeds',
            endpoint='directory', year=year, filters=college_filters)
        response = requests.get(college_url).json()
        count = response['count']
        total_count += count
        college = pd.DataFrame(response['results'])
        colleges_list.append(college)
        time.sleep(2)

    colleges = pd.concat(colleges_list)
    assert len(colleges) == total_count
    colleges = colleges[[
        'unitid', 'inst_name', 'longitude',
        'latitude']].set_index('unitid')
    colleges.rename(
        columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)

    logger.info(
        "Downloading college full-time enrollment data from "
        "educationdata.urban.org!")
    fte = _get_full_time_enrollment(state_fips, year)
    colleges['full_time_enrollment'] = fte.reindex(colleges.index)

    logger.info(
        "Downloading college part-time enrollment data from "
        "educationdata.urban.org!")
    pte = _get_part_time_enrollment(state_fips)
    colleges['part_time_enrollment'] = pte.reindex(colleges.index)
    return colleges


def _get_park_cost(zones, weights, index_cols, output_cols):
    params = pd.Series(weights, index=index_cols)
    cols = zones[output_cols]
    s = cols @ params
    return s.where(s > 0, 0)


def _compute_area_type_metric(zones):
    """
    Because of the modifiable areal unit problem, it is probably a good
    idea to visually assess the accuracy of this metric when implementing
    in a new region. The metric was designed using SF Bay Area data and TAZ
    geometries. So what is considered "suburban" by SFMTC standard might be
    "urban" or "urban fringe" in less densesly developed regions, which
    can impact the results of the auto ownership and mode choice models.

    This issue should eventually resolve itself once we are able to re-
    estimate these two models for every new region/implementation. In the
    meantime, we expect that for regions less dense than the SF Bay Area,
    the area types classifications will be overly conservative. If anything,
    this bias results towards higher auto-ownership and larger auto-oriented
    mode shares. However, we haven't found this to be the case.
    """
    zones_df = zones[['TOTPOP', 'TOTEMP', 'TOTACRE']].copy()

    metric_vals = ((
                           1 * zones_df['TOTPOP']) + (
                           2.5 * zones_df['TOTEMP'])) / zones_df['TOTACRE']

    return metric_vals.fillna(0)


def _compute_area_type(zones):
    # Integer, 0=regional core, 1=central business district,
    # 2=urban business, 3=urban, 4=suburban, 5=rural
    area_types = pd.cut(
        zones['area_type_metric'],
        [0, 6, 30, 55, 100, 300, float("inf")],
        labels=['5', '4', '3', '2', '1', '0'],
        include_lowest=True).astype(str)
    return area_types


def enrollment_tables(settings, zones,
                      enrollment_type='schools',
                      asim_zone_id_col='TAZ'):
    region = settings['region']
    FIPS = settings['FIPS'][region]
    state_fips = FIPS['state']
    county_codes = FIPS['counties']
    local_crs = settings['local_crs'][region]

    zone_type = settings['skims_zone_type']
    path_to_schools_data = \
        "pilates/utils/data/{0}/{1}_{2}.csv".format(region, zone_type, enrollment_type)
    assert enrollment_type in ['schools', 'colleges'], "enrollemnt type one of ['schools', 'colleges']"

    if not os.path.exists(path_to_schools_data):
        if enrollment_type == 'schools':
            enrollment = _get_school_enrollment(state_fips, county_codes)
        elif enrollment_type == 'colleges':
            enrollment = _get_college_enrollment(state_fips, county_codes)
        else:
            raise KeyError("enrollemnt type one of ['schools', 'colleges']")
    else:
        logger.info("Reading school enrollment data from disk!")
        enrollment = pd.read_csv(path_to_schools_data, dtype={
            asim_zone_id_col: str})

    if asim_zone_id_col not in enrollment.columns:
        enrollment_df = enrollment[['x', 'y']].copy()
        enrollment_df.index.name = 'school_id'
        enrollment[asim_zone_id_col] = get_zone_from_points(
            enrollment_df, zones, local_crs)

        enrollment = enrollment.dropna(subset=[asim_zone_id_col])
        enrollment[asim_zone_id_col] = enrollment[asim_zone_id_col].astype(str)
        del enrollment_df
        logger.info("Saving {} enrollment data to disk!".format(enrollment_type))
        enrollment.to_csv(path_to_schools_data)

    return enrollment


def _create_land_use_table(
        settings, region, zones, state_fips, county_codes, local_crs,
        households, persons, jobs, blocks, asim_zone_id_col='TAZ'):
    logger.info('Creating land use table.')
    zone_type = settings['skims_zone_type']

    schools = enrollment_tables(settings, zones,
                                enrollment_type='schools',
                                asim_zone_id_col=asim_zone_id_col)
    colleges = enrollment_tables(settings, zones,
                                 enrollment_type='colleges',
                                 asim_zone_id_col=asim_zone_id_col)
    assert zones.index.name == 'TAZ'
    assert zones.index.inferred_type == 'string', "zone_id dtype should be str"
    for table in [households, persons, jobs, blocks, schools, colleges]:
        assert pd.api.types.is_string_dtype(table[asim_zone_id_col]), \
            "zone_id dtype in should be str"

    # create new column variables
    logger.info("Creating new columns in the land use table.")
    if zone_type != 'taz':
        if 'STATE' in zones.columns:
            zones.loc[:, 'STATE'] = zones['STATE'].astype(str)
        else:
            zones.loc[:, 'STATE'] = settings['FIPS'][settings['region']]['state']
        zones.loc[:, 'COUNTY'] = zones['COUNTY'].astype(str)
        try:
            zones.loc[:, 'TRACT'] = zones['TRACT'].astype(str)
        except:
            print("Skipping TRACT")
        try:
            zones.loc[:, 'BLKGRP'] = zones['BLKGRP'].astype(str)
        except:
            print("Skipping BLKGRP")

    zones.loc[:, 'TOTHH'] = households[asim_zone_id_col].groupby(households[asim_zone_id_col]).count().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'TOTPOP'] = persons[asim_zone_id_col].groupby(persons[asim_zone_id_col]).count().reindex(
        zones.index).fillna(
        0)
    zones.loc[:, 'EMPRES'] = households[[asim_zone_id_col, 'workers']].groupby(asim_zone_id_col)[
        'workers'].sum().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'HHINCQ1'] = \
        households.loc[households['income'] < 30000, [asim_zone_id_col, 'income']].groupby(asim_zone_id_col)[
            'income'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'HHINCQ2'] = \
        households.loc[households['income'].between(30000, 59999), [asim_zone_id_col, 'income']].groupby(
            asim_zone_id_col)[
            'income'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'HHINCQ3'] = \
        households.loc[households['income'].between(60000, 99999), [asim_zone_id_col, 'income']].groupby(
            asim_zone_id_col)[
            'income'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'HHINCQ4'] = \
        households.loc[households['income'] >= 100000, [asim_zone_id_col, 'income']].groupby(asim_zone_id_col)[
            'income'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE0004'] = \
        persons.loc[persons['age'].between(0, 4), [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
            'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE0519'] = \
        persons.loc[persons['age'].between(5, 19), [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
            'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE2044'] = \
        persons.loc[persons['age'].between(20, 44), [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
            'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE4564'] = \
        persons.loc[persons['age'].between(45, 64), [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
            'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE64P'] = persons.loc[persons['age'] >= 65, [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
        'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGE62P'] = persons.loc[persons['age'] >= 62, [asim_zone_id_col, 'age']].groupby(asim_zone_id_col)[
        'age'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'SHPOP62P'] = (zones.AGE62P / zones.TOTPOP).reindex(zones.index).fillna(0)
    zones.loc[:, 'TOTEMP'] = jobs[asim_zone_id_col].groupby(jobs[asim_zone_id_col]).count().reindex(zones.index).fillna(
        0)
    zones.loc[:, 'RETEMPN'] = \
        jobs.loc[jobs['sector_id'].isin(['44-45']), [asim_zone_id_col, 'sector_id']].groupby(asim_zone_id_col)[
            'sector_id'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'FPSEMPN'] = \
        jobs.loc[jobs['sector_id'].isin(['52', '54']), [asim_zone_id_col, 'sector_id']].groupby(asim_zone_id_col)[
            'sector_id'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'HEREMPN'] = \
        jobs.loc[jobs['sector_id'].isin(['61', '62', '71']), [asim_zone_id_col, 'sector_id']].groupby(asim_zone_id_col)[
            'sector_id'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'AGREMPN'] = \
        jobs.loc[jobs['sector_id'].isin(['11']), [asim_zone_id_col, 'sector_id']].groupby(asim_zone_id_col)[
            'sector_id'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'MWTEMPN'] = \
        jobs.loc[jobs['sector_id'].isin(['42', '31-33', '32', '48-49']), [asim_zone_id_col, 'sector_id']].groupby(
            asim_zone_id_col)['sector_id'].count().reindex(zones.index).fillna(0)
    zones.loc[:, 'OTHEMPN'] = jobs.loc[
        ~jobs['sector_id'].isin(['44-45', '52', '54', '61', '62', '71', '11', '42', '31-33', '32', '48-49']), [
            asim_zone_id_col, 'sector_id']].groupby(asim_zone_id_col)['sector_id'].count().reindex(zones.index).fillna(
        0)
    zones.loc[:, 'TOTACRE'] = blocks[['TOTACRE', asim_zone_id_col]].groupby(asim_zone_id_col)['TOTACRE'].sum().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'HSENROLL'] = schools[['enrollment', asim_zone_id_col]].groupby(asim_zone_id_col)[
        'enrollment'].sum().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'TOPOLOGY'] = 1  # FIXME
    zones.loc[:, 'employment_density'] = (zones.TOTEMP / zones.TOTACRE).fillna(0.0)
    zones.loc[:, 'pop_density'] = (zones.TOTPOP / zones.TOTACRE).fillna(0.0)
    zones.loc[:, 'hh_density'] = (zones.TOTHH / zones.TOTACRE).fillna(0.0)
    zones.loc[:, 'hq1_density'] = (zones.HHINCQ1 / zones.TOTACRE).fillna(0.0)
    zones.loc[:, 'PRKCST'] = _get_park_cost(
        zones, [-1.92168743, 4.89511403, 4.2772001, 0.65784643],
        ['pop_density', 'hh_density', 'hq1_density', 'employment_density'],
        ['employment_density', 'pop_density', 'hh_density', 'hq1_density'])
    zones.loc[:, 'OPRKCST'] = _get_park_cost(
        zones, [-6.17833544, 17.55155703, 2.0786466],
        ['pop_density', 'hh_density', 'employment_density'],
        ['employment_density', 'pop_density', 'hh_density'])
    zones.loc[:, 'COLLFTE'] = colleges[[
        asim_zone_id_col, 'full_time_enrollment']].groupby(
        asim_zone_id_col)['full_time_enrollment'].sum().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'COLLPTE'] = colleges[[
        asim_zone_id_col, 'part_time_enrollment']].groupby(
        asim_zone_id_col)['part_time_enrollment'].sum().reindex(
        zones.index).fillna(0)
    zones.loc[:, 'TERMINAL'] = 0
    zones.loc[:, 'area_type_metric'] = _compute_area_type_metric(zones)
    zones.loc[:, 'area_type'] = _compute_area_type(zones)
    zones.loc[:, 'TERMINAL'] = 0  # FIXME
    zones.loc[:, 'COUNTY'] = 1  # FIXME

    return zones


def copy_beam_geoms(settings, beam_geoms_location, asim_geoms_location):
    zone_type_column = {'block_group': 'BLKGRP', 'taz': 'TAZ', 'block': 'BLK'}
    beam_geoms_file = pd.read_csv(beam_geoms_location)
    zone_type = settings['skims_zone_type']
    zone_id_col = zone_type_column[zone_type]

    if 'TAZ' not in beam_geoms_file.columns:

        mapping = geoid_to_zone_map(settings, settings['start_year'])

        if zone_type == 'block':
            logger.info("Mapping block IDs")
            beam_geoms_file['TAZ'] = beam_geoms_file['GEOID'].astype(str).replace(mapping)

        elif zone_type == 'block_group':
            logger.info("Mapping block group IDs to TAZ ids")
            beam_geoms_file['TAZ'] = beam_geoms_file['GEOID'].astype(str).replace(mapping)

    beam_geoms_file.to_csv(asim_geoms_location)


def create_asim_data_from_h5(
        settings, year, warm_start=False, output_dir=None):
    # warm start: year = start_year
    # asim_no_usim: year = start_year
    # normal: year = forecast_year
    region = settings['region']
    region_id = settings['region_to_region_id'][region]
    FIPS = settings['FIPS'][region]
    state_fips = FIPS['state']
    county_codes = FIPS['counties']
    local_crs = settings['local_crs'][region]
    usim_local_data_folder = settings['usim_local_data_folder']
    zone_type = settings['skims_zone_type']

    if not output_dir:
        output_dir = settings['asim_local_input_folder']

    input_zone_id_col = 'zone_id'
    asim_zone_id_col = 'TAZ'

    # TODO: only call _get_zones_geoms if blocks or colleges or schools
    # don't already have a zone ID (e.g. TAZ). If they all do then we don't
    # need zone geoms and we can simply instantiate the zones table from
    # the unique zone ids in the blocks/persons/households tables.
    zones = read_zone_geoms(settings, year,
                            asim_zone_id_col=asim_zone_id_col,
                            default_zone_id_col=input_zone_id_col)

    store, table_prefix_yr = read_datastore(
        settings, year, warm_start=warm_start)

    logger.info("Loading UrbanSim data from .h5")
    households = store[os.path.join(table_prefix_yr, 'households')]
    persons = store[os.path.join(table_prefix_yr, 'persons')]
    blocks = store[os.path.join(table_prefix_yr, 'blocks')]
    jobs = store[os.path.join(table_prefix_yr, 'jobs')]

    # update blocks
    blocks_cols = blocks.columns.tolist()
    blocks_to_taz_mapping_updated, blocks = _update_blocks_table(
        settings, year, blocks, households, jobs, input_zone_id_col)
    input_zone_id_col = "{0}_zone_id".format(zone_type)
    if blocks_to_taz_mapping_updated:
        logger.info(
            "Storing blocks table with {} zone IDs to disk in .h5 datastore!".format(zone_type))
        blocks_cols += [input_zone_id_col]
        store[os.path.join(table_prefix_yr, 'blocks')] = blocks[blocks_cols]
    blocks.rename(
        columns={input_zone_id_col: asim_zone_id_col},
        inplace=True)  # Rename happens here.

    # update households
    households, unassigned_households = _update_households_table(households, blocks, asim_zone_id_col)

    # update persons
    persons = _update_persons_table(persons, households, unassigned_households, blocks, asim_zone_id_col)

    households.loc[:, 'persons'] = persons.groupby('household_id').size().reindex(households.index)
    households.loc[:, 'workers'] = persons.loc[persons.worker > 0].groupby('household_id').size().reindex(
        households.index).fillna(0)

    # update jobs
    jobs_cols = jobs.columns
    num_reassigned, jobs = _update_jobs_table(
        jobs, blocks, state_fips, county_codes, local_crs,
        asim_zone_id_col)

    if num_reassigned > 0:
        # update data store with new block_id's to avoid triggering
        # this process again in the future
        logger.info(
            "Storing jobs table with updated block IDs to disk "
            "in .h5 datastore!")
        store[os.path.join(str(table_prefix_yr), 'jobs')] = jobs[jobs_cols]

    store.close()

    # create land use table
    land_use = _create_land_use_table(
        settings, region, zones, state_fips, county_codes, local_crs,
        households, persons, jobs, blocks)

    households.to_csv(os.path.join(output_dir, 'households.csv'))
    persons.to_csv(os.path.join(output_dir, 'persons.csv'))
    land_use.to_csv(os.path.join(output_dir, 'land_use.csv'))
