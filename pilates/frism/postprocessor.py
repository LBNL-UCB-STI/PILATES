import logging
import os
import re
from itertools import groupby
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def copy_to_beam(settings, is_start_year: bool):
    logger.info('Copying frism results to beam input')
    frism_data_folder = settings['frism_data_folder']
    region = settings['region']
    frism_tour_plan_folder = os.path.join(frism_data_folder, region, 'Tour_plan') if is_start_year \
        else os.path.join(frism_data_folder, region, 'frism_light', 'Tour_plan')
    carrier_tour_payload_frames = [read_carrier_tour_payload_and_modify_ids(group) for group in grouped_paths(frism_tour_plan_folder)]
    unzipped = list(zip(*carrier_tour_payload_frames))
    carrier = pd.concat(unzipped[0], ignore_index=True)
    tour = pd.concat(unzipped[1], ignore_index=True)
    payload = pd.concat(unzipped[2], ignore_index=True)

    beam_freight_path = os.path.join(
        settings['beam_local_input_folder'],
        settings['region'],
        settings['beam_freight_folder'])
    carrier.rename(columns={
        'depot_zone': 'warehouseZone',
        'depot_zone_x': 'warehouseX',
        'depot_zone_y': 'warehouseY'
    }, inplace=True)
    columns_as_type_int(['warehouseZone'], carrier)
    carrier.to_csv(os.path.join(beam_freight_path, 'freight-carriers.csv'), index=False)

    tour.rename(columns={
        'tour_id': 'tourId',
        'departureLocation_zone': 'departureLocationZone',
        'departureLocation_x': 'departureLocationX',
        'departureLocation_y': 'departureLocationY'
    }, inplace=True)
    columns_as_type_int(['departureTimeInSec', 'maxTourDurationInSec', 'departureLocationZone'], tour)
    tour.to_csv(os.path.join(beam_freight_path, 'freight-tours.csv'), index=False)

    payload.loc[payload['weightInlb'] < 0, 'requestType'] = 1 # beam treats 1 as unloading
    payload.loc[payload['weightInlb'] > 0, 'requestType'] = 0 # beam treats 0 as loading
    payload['weightInlb'] = abs(payload['weightInlb'].astype(int)) * 0.4536
    payload.rename(columns={
        'arrivalTimeWindowInSec_lower': 'arrivalTimeWindowInSecLower',
        'arrivalTimeWindowInSec_upper': 'arrivalTimeWindowInSecUpper',
        'weightInlb': 'weightInKg',
        'locationZone_x': 'locationX',
        'locationZone_y': 'locationY'
    }, inplace=True)
    payload.drop(columns=['weightInlb', 'cummulativeWeightInlb'], inplace=True, errors='ignore')
    columns_as_type_int(['sequenceRank',
                         'payloadType',
                         'requestType',
                         'estimatedTimeOfArrivalInSec',
                         'arrivalTimeWindowInSecLower',
                         'arrivalTimeWindowInSecUpper',
                         'operationDurationInSec',
                         'locationZone',], payload)
    payload.to_csv(os.path.join(beam_freight_path, 'freight-payload-plans.csv'), index=False)
    return carrier


def grouped_paths(folder):
    def file_prefix(path: Path):
        name = path.name
        array = re.split(r'_carrier_|_freight_tours_|_payload_', name)
        return array[0] if len(array) == 2 else None

    paths = Path(folder).glob("*.csv")
    paths = [x for x in paths if file_prefix(x)]
    paths_sorted = sorted(paths, key=file_prefix)
    paths_grouped = [list(it) for k, it in groupby(paths_sorted, file_prefix)]
    return [x for x in paths_grouped if len(x) == 3]


def read_carrier_tour_payload_and_modify_ids(paths3):
    payload = pd.read_csv(find_path(paths3, 'payload'))
    tour = pd.read_csv(find_path(paths3, 'freight_tours'))
    carrier = pd.read_csv(find_path(paths3, 'carrier'))
    carrier['originalTourId'] = carrier['tourId']
    carrier['tourId'] = carrier['carrierId'] + '-' + carrier['originalTourId'].astype(str)
    carrier['vehicleId'] = carrier['carrierId'] + '-' + carrier['vehicleId'].astype(str)
    tour['tour_id'] = pd.merge(tour, carrier, left_on='tour_id', right_on='originalTourId')['tourId']
    payload['tourId'] = pd.merge(payload, carrier, left_on='tourId', right_on='originalTourId')['tourId_y']
    carrier.drop(columns='originalTourId', inplace=True)
    return carrier, tour, payload

def columns_as_type_int(columns, df):
    for column in columns:
        df[column] = df[column].astype(int)

def find_path(paths3, file_type):
    result = next(filter(lambda path: file_type in path.name, paths3))
    if result is None:
        raise ValueError(f"No {file_type} for {paths3[0]}")
    return result
