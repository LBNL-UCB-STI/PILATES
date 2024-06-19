import logging
import os

import numpy as np
import pandas as pd

# # Commented commands for only for debugging
# import yaml
# import argparse
# with open('settings.yaml') as file:
#     settings = yaml.load(file, Loader=yaml.FullLoader)


logger = logging.getLogger(__name__)


def _get_usim_datastore_fname(settings, io, year=None):
    # reference: asim postprocessor
    if io == 'output':
        datastore_name = settings['usim_formattable_output_file_name'].format(
            year=year)
    elif io == 'input':
        region = settings['region']
        region_id = settings['region_to_region_id'][region]
        usim_base_fname = settings['usim_formattable_input_file_name']
        datastore_name = usim_base_fname.format(region_id=region_id)
    return datastore_name


def atlas_update_h5_vehicle(settings, output_year, state: "WorkflowState", warm_start=False):
    # use atlas outputs in year provided and update "cars" & "hh_cars"
    # columns in urbansim h5 files
    logger.info('ATLAS is updating urbansim outputs for Year {}'.format(output_year))

    # read and format atlas vehicle ownership output
    atlas_output_path = os.path.join(state.full_path,
                                     settings['atlas_host_output_folder'])  # 'pilates/atlas/atlas_output'  #
    fname = 'householdv_{}.csv'.format(output_year)
    df = pd.read_csv(os.path.join(atlas_output_path, fname))
    df = df.rename(columns={'nvehicles': 'cars'}).set_index('household_id').sort_index(ascending=True)
    df['hh_cars'] = pd.cut(df['cars'], bins=[-0.5, 0.5, 1.5, np.inf], labels=['none', 'one', 'two or more'])

    # set which h5 file to update
    h5path = os.path.join(state.full_path, settings['usim_local_mutable_data_folder'])
    if warm_start:
        h5fname = _get_usim_datastore_fname(settings, io='input')
    else:
        h5fname = _get_usim_datastore_fname(settings, io='output', year=output_year)

    logger.info("Writing updated household vehicle info to h5 file {0}".format(h5fname))

    # read original h5 files
    with pd.HDFStore(os.path.join(h5path, h5fname), mode='r+') as h5:

        # if in main loop, update "model_data_*.h5", which has three layers ({$year}/households/cars)
        if not warm_start:
            key = '/{}/households'.format(output_year)
        # if in warm start, update "custom_mpo_***.h5", which has two layers (households/cars)
        else:
            key = 'households'

        olddf = h5[key]
        olddf.index = olddf.index.astype(int)
        olddf = olddf.reindex(df.index.astype(int))

        if olddf.shape[0] != df.shape[0]:
            logger.error('ATLAS household_id mismatch found - NOT update h5 datastore')
        else:
            olddf['cars'] = df['cars'].values
            olddf['hh_cars'] = df['hh_cars'].values
            for col in olddf.columns:
                if olddf[col].dtype == "category":
                    logger.info("Converting column {0} from category to str".format(col))
                    olddf[col] = olddf[col].astype(str)
            h5[key] = olddf
            logger.info('ATLAS update h5 datastore table {0} - done'.format(key))


def atlas_add_vehileTypeId(settings, output_year, state):
    # add a "vehicleTypeId" column in atlas output vehicles_{$year}.csv,
    # write as vehicles2_{$year}.csv
    # which will be read by beam preprocessor
    # vehicleTypeId = conc "bodytype"-"vintage_category"-"pred_power"

    atlas_output_path = os.path.join(state.full_path, settings['atlas_host_output_folder'])
    fname = 'vehicles_{}.csv'.format(output_year)

    # read original atlas output "vehicles_*.csv" as dataframe
    df = pd.read_csv(os.path.join(atlas_output_path, fname))

    # atlas:v1.0.6 can generate continuous modelyear
    df['modelyear'] = df['modelyear'].astype(int)

    # add "vehicleTypeId" column in dataframe for BEAM
    # for prior-2015-model vehicles, vehicleTypeId is *_*_2015
    df['vehicleTypeId'] = df[['bodytype', 'pred_power', 'modelyear']].astype(str).agg('_'.join, axis=1)
    df.loc[df['modelyear'] < 2015, 'vehicleTypeId'] = df.loc[df['modelyear'] < 2015, ['bodytype', 'pred_power']].astype(
        str).agg('_'.join, axis=1) + '_2015'

    # write to a new file vehicles2_*.csv 
    # because original file cannot be overwritten (root-owned)
    # may revise later
    df.to_csv(os.path.join(atlas_output_path, 'vehicles2_{}.csv'.format(output_year)), index=False)


def build_beam_vehicles_input(settings, output_year, state):
    atlas_output_path = os.path.join(state.full_path, settings['atlas_host_output_folder'])
    atlas_input_path = os.path.join(state.full_path, settings['atlas_host_input_folder'])
    vehicles = pd.read_csv(os.path.join(atlas_output_path, "vehicles_{0}.csv".format(output_year)),
                           dtype={"householdId": pd.Int64Dtype()})
    mapping = pd.read_csv(
        os.path.join(atlas_input_path, "vehicle_type_mapping_{0}.csv".format(settings['atlas_adscen'])))
    mapping['numberOfVehiclesCreated'] = 0
    mapping.set_index(["adopt_fuel", "bodytype", "modelyear", "vehicleTypeId"], inplace=True, drop=True)
    mapping = mapping.loc[~mapping.index.duplicated(), :]
    allCounts = mapping.copy()
    allVehicles = []
    for (fuelType, bodyType, modelYear), vehiclesSub in vehicles.groupby(["adopt_fuel", "adopt_veh", "modelyear"]):
        try:
            matched = mapping.loc[(fuelType, bodyType, modelYear, slice(None)), :]
        except KeyError:
            try:
                temp = mapping.loc[(fuelType, bodyType, slice(None), slice(None)), :]
                bestOption = (temp.reset_index()['modelyear'] - modelYear).abs().idxmin()
                bestYear = temp.reset_index().iloc[bestOption, :]["modelyear"]
                matched = mapping.loc[(fuelType, bodyType, bestYear, slice(None)), :]
            except KeyError:
                try:
                    matched = mapping.loc[(fuelType, slice(None), modelYear), :]
                except KeyError:
                    try:
                        temp = mapping.loc[(fuelType, slice(None), slice(None), slice(None)), :]
                        bestOption = (temp.reset_index()['modelyear'] - modelYear).abs().idxmin()
                        bestYear = temp.reset_index().iloc[bestOption, :]["modelyear"]
                        matched = mapping.loc[(fuelType, slice(None), bestYear), :]
                    except KeyError:
                        try:
                            temp = mapping.loc[(slice(None), bodyType, slice(None), slice(None)), :]
                            bestOption = (temp.reset_index()['modelyear'] - modelYear).abs().idxmin()
                            bestYear = temp.reset_index().iloc[bestOption, :]["modelyear"]
                            matched = mapping.loc[(slice(None), bodyType, bestYear), :]
                        except KeyError:
                            bestOption = (mapping.reset_index()['modelyear'] - modelYear).abs().idxmin()
                            bestYear = mapping.reset_index().iloc[bestOption, :]["modelyear"]
                            matched = mapping.loc[(slice(None), slice(None), bestYear), :]
        createdVehicles = matched.sample(vehiclesSub.shape[0], replace=True,
                                         weights=matched['sampleProbabilityWithinCategory'].values)
        createdVehicleCounts = createdVehicles.index.value_counts()
        allCounts.loc[createdVehicleCounts.index, 'numberOfVehiclesCreated'] += createdVehicleCounts.values
        vehiclesSub['vehicleTypeId'] = createdVehicles.index.get_level_values('vehicleTypeId')
        vehiclesSub['stateOfCharge'] = np.nan
        allVehicles.append(
            vehiclesSub[['household_id', 'vehicleTypeId']])
    outputVehicles = pd.concat(allVehicles).reset_index(drop=True)
    outputVehicles.rename(columns={"household_id": "householdId"}, inplace=True)
    outputVehicles.index.rename("vehicleId", inplace=True)
    outputVehicles.to_csv(os.path.join(atlas_output_path, 'vehicles_{0}.csv.gz'.format(output_year)))
    allCounts.loc[allCounts.numberOfVehiclesCreated > 0, :].sort_values(by="numberOfVehiclesCreated", ascending=False)[
        'numberOfVehiclesCreated'].to_csv(
        os.path.join(atlas_output_path, 'vehicles_by_type_{0}.csv'.format(output_year)))
