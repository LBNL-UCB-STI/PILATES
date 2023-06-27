import numpy as np
import pandas as pd
import os
import geopandas as gpd
import random
import statsmodels.api as sm


def update_fleet_files(trips_df, settings):
    FLEET_MIX_FILENAME = settings['TNCfleetMixFilename']
    TAZ_FILENAME = settings['tazFilename']
    ZIP_FILENAME = settings['zipFilename']
    AGG_MODEL_FILENAME = settings['aggModelFilename']
    DELTA_MODEL_FILENAME = settings['deltaModelFilename']
    FLEETS = settings['fleets']
    SCALE_FACTOR = settings['scaleFactor']
    BEAM_INPUT_PATH = settings['beamInputPath']

    zipShp = load_geoData(ZIP_FILENAME,TAZ_FILENAME)
    adf = trips_df.copy()
    adf.rename(columns={'depart':'Hour'})

    for f in FLEETS:
        df = adf.loc[(adf.trip_mode.isin(['SINGLE_{}'.format(f.upper()),'SHARED_{}'.format(f.upper())]))].copy()
        data = get_data(df,zipShp)
        data = scale_data(data, SCALE_FACTOR, columns=['Trip ends', 'Trip requests'])
        data['prediction_ActiveVeh'] = run_model(data, AGG_MODEL_FILENAME, 'ActiveVeh')/60
        data['prediction_DeltaVeh'] = run_model(data, DELTA_MODEL_FILENAME, 'DeltaVeh')/60
        data = scale_data(data, 1/SCALE_FACTOR, OVERLAP, columns = ['prediction_ActiveVeh','prediction_DeltaVeh'])

        # Adjust predicted change in vehicles to match the predicted number of vehicles per hour
        data['Delta_positive'] = (data['prediction_DeltaVeh']>0).astype(int)*data['prediction_DeltaVeh']
        data['Delta_negative'] = (data['prediction_DeltaVeh']<0).astype(int)*data['prediction_DeltaVeh']
        sum_data = data[['Hour','prediction_ActiveVeh','prediction_DeltaVeh','Delta_positive','Delta_negative']].groupby('Hour').sum().reset_index().sort_values('Hour')
        sum_data['Adjustment'] = sum_data['prediction_ActiveVeh'].shift(-1).fillna(0)-(sum_data['prediction_ActiveVeh'].fillna(0)+sum_data['prediction_DeltaVeh'].fillna(0))
        sum_data.loc[0,'Adjustment'] += sum_data.loc[0,'prediction_ActiveVeh']
        data = data.merge(sum_data, how='left', left_on='Hour', right_on = 'Hour',suffixes=('','_total'))

        data['adj_DeltaVeh'] = data['prediction_DeltaVeh'].copy()
        # if adjustment is positive in a particular hour (there are more active vehicles predicted than will result from deltaVeh)
        data.loc[(data.Adjustment>0)&(data.Delta_positive_total>0),'adj_DeltaVeh'] = data.loc[(data.Adjustment>0)&(data.Delta_positive_total>0),'adj_DeltaVeh'] + data.loc[(data.Adjustment>0)&(data.Delta_positive_total>0),'Adjustment']*(data.loc[(data.Adjustment>0)&(data.Delta_positive_total>0),'Delta_positive']/data.loc[(data.Adjustment>0)&(data.Delta_positive_total>0),'Delta_positive_total'])
        data.loc[(data.Adjustment<0)&(data.Delta_negative_total<0),'adj_DeltaVeh'] = data.loc[(data.Adjustment<0)&(data.Delta_negative_total<0),'adj_DeltaVeh'] + data.loc[(data.Adjustment<0)&(data.Delta_negative_total<0),'Adjustment']*(data.loc[(data.Adjustment<0)&(data.Delta_negative_total<0),'Delta_negative']/data.loc[(data.Adjustment<0)&(data.Delta_negative_total<0),'Delta_negative_total'])

        vehicleFile = generate_vehicles(data.fillna(0),f,FLEET_MIX_FILENAME)
        columns = ['id','rideHailManagerId','vehicleType','initialLocationX','initialLocationY','shifts',
               'geofenceX','geofenceY','geofenceRadius','geofenceTAZFile','fleetId','initialStateOfCharge']

        vehicleFile.to_csv('{}/FleetInitialization-{}.csv'.format(BEAM_INPUT_PATH,f),index=False)

def run_model(data, model, model_name):
    results = sm.load(model)
    model_columns = results.model.exog_names[1:]
    X_test = sm.add_constant(data[model_columns].fillna(0))
    y_pred = results.predict(X_test)
    return y_pred

def scale_data(df, scaleFactor, overlap, columns):
    for c in columns:
        if c in df.columns:
            df[c] = (df[c]/scaleFactor)/(1+overlap)
#         else:
# #            print("WARNING: {} NOT IN COLUMNS".format(c))
    return df

def get_data(df,zipShp):
    df['origin'] = df['origin'].astype(int)
    df['destination'] = df['destination'].astype(int)
    zipShp['Taz'] = zipShp['Taz'].astype(int)

    num_requests = df[['depart','origin','destination']].groupby(['depart','origin']).count().reset_index().rename(columns={'depart':'Hour','origin':'Taz','destination':'Trip requests'})
    num_ends = df[['depart','origin','destination']].groupby(['depart','destination']).count().reset_index().rename(columns={'depart':'Hour','destination':'Taz','origin':'Trip ends'})
    data = num_requests.merge(num_ends,how='outer',left_on=['Hour','Taz'],right_on=['Hour','Taz']).fillna(0)
    data = data.merge(zipShp[['Taz','City','County']],how='left',left_on='Taz',right_on='Taz')
    data = data.groupby(['Hour','City','County']).sum().reset_index()

    data = data.merge(
        data[['Hour','County','Trip requests','Trip ends']].groupby(['Hour','County']).sum().reset_index(),
        how='left',left_on=['Hour','County'],right_on=['Hour','County'],suffixes=('',' CNTY'))

    data['Night'] = (data['Hour']<3).astype(int)+(data['Hour']>=20).astype(int)
    data['Early Morning'] = ((data['Hour']>=3)&(data['Hour']<5)).astype(int)
    data['AM Peak'] = ((data['Hour']>=5)&(data['Hour']<10)).astype(int)
    data['Mid-Day'] = ((data['Hour']>=10)&(data['Hour']<15)).astype(int)
    data['PM Peak'] = ((data['Hour']>=15)&(data['Hour']<20)).astype(int)
    for c in ['San Mateo', 'San Francisco', 'Santa Clara', 'Alameda', 'Napa',
       'Contra Costa', 'Solano', 'Sonoma', 'Marin' ]:
        data[c] = (data.County==c).astype(int)

    Bounds = zipShp[['City','minx','miny']].groupby(['City']).min().reset_index().merge(
            zipShp[['City','maxx','maxy']].groupby(['City']).max().reset_index(),how='left',left_on=['City'],right_on=['City'])

    return data.merge(Bounds,how='left',left_on=['City'],right_on=['City']).rename(columns={'City':'CITY'})

def generate_vehicles(df,fleet,fleetMixFilename,geofenceFilename=''):
    df = df.sort_values(['Hour']).copy()
    vehCount = 0
    inactiveVehCount = 0
    vehList = []
    extraRemoval = 0
    for i,row in df.iterrows():
        numNew = np.max([0,int(np.round(row['adj_DeltaVeh'],0))])
        numEnd = -1 * np.min([0,int(np.ceil(row['adj_DeltaVeh']))])

        # initiate vehicles in this zip with unknown shiftEnd time
        for n in range(numNew):
            x,y = genPoint(row)
            vehList.append({'id': 'rideHailVehicle-{}-{}'.format(fleet,vehCount+n),
                            'rideHailManagerId':'RideHail',
                            'fleetId':fleet,
                            'vehicleType' : '',
                           'shifts':[int((row['Hour'])*3600),0],
                          'initialLocationX': x,
                           'initialLocationY': y,
                           'geofenceTAZFile':geofenceFilename,
                            'geofenceX':'','geofenceY':'','geofenceRadius':'',
                            'initialStateOfCharge':1.0
                           })
        vehCount += numNew

        # set shiftEnd time for vehicles according to FIFO
        for n in range(numEnd):
            if vehCount>inactiveVehCount:

                vehList[inactiveVehCount]['shifts'] = '{'+"{}:{}".format(vehList[inactiveVehCount]['shifts'][0],
                                                                    int((row['Hour']+1)*3600))+'}'
                inactiveVehCount+=1
            else:
                extraRemoval +=1
#                 print('Attempted to remove more vehicles than initiated @ hour {}; {} extra'.format(row['Hour'],extraRemoval))


    if inactiveVehCount<vehCount:
        for n in range(inactiveVehCount,vehCount):
            vehList[n]['shifts'] = '{'+"{}:{}".format(vehList[n]['shifts'][0],
                                                                    int(24*3600))+'}'
#     else:
#        print('{} vehicles initiated; {} vehicles removed'.format(vehCount,inactiveVehCount))
    vehDf = pd.DataFrame(vehList)
    vehDf['vehicleType'] = genVehType(vehDf.shape[0],fleetMixFilename)
    return vehDf


def genPoint(row):
    x = np.random.uniform(row['minx'], row['maxx'])
    y = np.random.uniform(row['miny'], row['maxy'])
    return x,y

def genVehType(n,fleetMixFilename):
    # load distribution
    distr = pd.read_csv(fleetMixFilename)
    # randomly generate a vehicle type
    return random.choices(distr['vehicleTypeId'].str.strip(), weights=distr['SampleProbability'], k=n)

def load_geoData(zip_filename,tazFilename):
    zipShp = gpd.read_file(zip_filename).to_crs('epsg:26910')
    zipShp['ZIP_CODE'] = zipShp['ZIP_CODE'].astype(int)
    zipShp.rename(columns={'ZIP_CODE':'Zip','PO_NAME':'City','POPULATION':'Population'},inplace=True)

    taz = gpd.read_file(tazFilename).rename(columns={'taz1454':'Taz','county':'County'})
    taz[['minx','miny','maxx','maxy']] = taz.bounds
    taz_join = gpd.sjoin(taz[['Taz','geometry', 'minx', 'miny', 'maxx','maxy','County']],
                    zipShp, how='right', op='intersects')

    zipShp = taz_join[['Zip','County','maxx','maxy', 'City','Taz']].groupby(['Zip','County', 'City','Taz']).max().reset_index()
    zipShp = zipShp.merge(taz_join[['Zip','County','minx','miny', 'City','Taz']].groupby(['Zip','County', 'City','Taz']).min().reset_index(),
                                     how='left',left_on=['Zip', 'County','City','Taz'], right_on=['Zip','County', 'City','Taz'])
    return zipShp