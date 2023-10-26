# %%
import pandas as pd
import numpy as np
import random
import config
from argparse import ArgumentParser
from os.path import exists as file_exists
import os
import VRP_light
import time
# %%
def main(args=None):
    # read input files
    parser = ArgumentParser()
    parser.add_argument("-sr", "--sample ratio", dest="sample_ratio",
                help="0-100", required=True, type=int)
    parser.add_argument("-md", "--main_folder", dest="mdir",
                help="main results folder", required=True, type=str)              
    parser.add_argument("-sn", "--scenario", dest="scenario",
                    help="scenario", required=True, type=str)                
    parser.add_argument("-yt", "--analysis year", dest="target_year",
                    help="20XX", required=True, type=int)  
    parser.add_argument("-cy", "--county-number", dest="county_num",
                        help="an integer indicating the county number", required=True, type=int)
    parser.add_argument("-st", "--ship_type", dest="ship_type",
                        help="B2B or B2C", required=True, type=str)
    parser.add_argument("-t", "--travel_time_file", dest="travel_file",
                        help="travel time file in gz format", required=True, type=str)
    parser.add_argument("-d", "--distance_file", dest="dist_file",
                        help="distance file in csv format", required=True, type=str)
    parser.add_argument("-ct", "--freigh_centroid_file", dest="CBGzone_file",
                        help="travel time file in geojson format", required=True, type=str)
    parser.add_argument("-vt", "--vehicle_type_file", dest="vehicleType_file",
                        help="vehicle type file in csv format", required=True, type=str)                                                                                                                                                
    args = parser.parse_args()

    # %%
    start_time=time.time()
    travel_file= args.travel_file
    dist_file=args.dist_file 
    CBGzone_file=args.CBGzone_file
    vehicleType_file= args.vehicleType_file 
    # %%
    sample_ratio = sample_ratio = args.sample_ratio
    mdir=args.mdir
    analysis_year=args.target_year
    scenario=args.scenario
    ship_type = args.ship_type #["B2B", "B2C"]
    county =args.county_num # [453, 491, 209, 55, 21, 53]
    fdir_tour= mdir +"frism_light/Tour_plan/"
    fdir_shipment= mdir +"frism_light/Shipment2Fleet/"

    out_fdir_tour= mdir +"frism_light/Tour_plan/"
    out_fdir_shipment= mdir +"frism_light/Shipment2Fleet/"

    # %%
    # read x% of sample 
    tour= pd.read_csv(fdir_tour+"{}_county{}_freight_tours_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
    carrier=pd.read_csv(fdir_tour+"{}_county{}_carrier_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
    payload= pd.read_csv(fdir_tour+"{}_county{}_payload_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
    list_tour = tour["tour_id"].unique().tolist()
    list_tour_sample= random.sample(list_tour, int(len(list_tour)*sample_ratio/100))
    # will be not updated
    org_tour=tour[~tour["tour_id"].isin(list_tour_sample)]
    org_carrier=carrier[~carrier["tourId"].isin(list_tour_sample)]
    org_payload=payload[~payload["tourId"].isin(list_tour_sample)]
    # will be updated
    sample_tour=tour[tour["tour_id"].isin(list_tour_sample)]
    sample_carrier=carrier[carrier["tourId"].isin(list_tour_sample)]
    sample_payload=payload[payload["tourId"].isin(list_tour_sample)]
    sample_payload_list = sample_payload["payloadId"].unique().tolist()


    shipment_payload= pd.read_csv(fdir_shipment+"{}_payload_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
    shipment_carrier= pd.read_csv(fdir_shipment+"{}_carrier_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
    org_shipment_payload = shipment_payload[~shipment_payload["payload_id"].isin(sample_payload_list)]
    org_shipment_carrier = shipment_carrier[shipment_carrier["carrier_id"].isin(org_shipment_payload["carrier_id"].unique().tolist())]
    sample_shipment_payload = shipment_payload[shipment_payload["payload_id"].isin(sample_payload_list)]
    sample_shipment_carrier = shipment_carrier[shipment_carrier["carrier_id"].isin(sample_shipment_payload["carrier_id"].unique().tolist())]

    ## Need reassignment? 
    # For B2C- reassignment B2B - delivery only (private fleet): no assignment pick-delivery (for-hire fleet)
    # Question how can back track shipments and ? 
    ##
    # %%
    update_tour,update_carrier,update_payload = VRP_light.main(county,travel_file,dist_file, CBGzone_file,sample_shipment_carrier, sample_shipment_payload, vehicleType_file, ship_type )

    # Combining org file (1-x sample) and update file (x sample) 
    ## reassign tour_ID
    dic_tourID = {}
    n=0
    for id in org_tour["tour_id"].unique():
        dic_tourID[id]=n
        n+=1
    org_tour["tour_id"]=org_tour["tour_id"].apply(lambda x:dic_tourID[x])
    org_carrier["tourId"] = org_carrier["tourId"].apply(lambda x:dic_tourID[x])
    org_payload["tourId"] = org_payload["tourId"].apply(lambda x:dic_tourID[x])

    dic_tourID = {}
    for id in update_tour["tour_id"].unique():
        dic_tourID[id]=n
        n+=1
    update_tour["tour_id"]=update_tour["tour_id"].apply(lambda x:dic_tourID[x])
    update_carrier["tourId"] = update_carrier["tourId"].apply(lambda x:dic_tourID[x])
    update_payload["tourId"] = update_payload["tourId"].apply(lambda x:dic_tourID[x])
    ## two file combine 
    update_tour= pd.concat([org_tour,update_tour], ignore_index=True).reset_index(drop=True)  
    update_carrier=pd.concat([org_carrier,update_carrier], ignore_index=True).reset_index(drop=True)  
    update_payload=pd.concat([org_payload,update_payload], ignore_index=True).reset_index(drop=True)

    update_shipment_payload= pd.concat([org_shipment_payload,sample_shipment_payload], ignore_index=True).reset_index(drop=True)  # "sample_shipment_payload" should be updated 
    update_shipment_carrier=pd.concat([org_shipment_carrier,sample_shipment_carrier], ignore_index=True).reset_index(drop=True)  

    # %%
    # Save files
    update_tour.to_csv(out_fdir_tour+"{}_county{}_freight_tours_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
    update_carrier.to_csv(out_fdir_tour+"{}_county{}_carrier_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
    update_payload.to_csv(out_fdir_tour+"{}_county{}_payload_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)

    update_shipment_payload.to_csv(out_fdir_shipment+"{}_payload_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
    update_shipment_carrier.to_csv(out_fdir_shipment+"{}_carrier_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)

    print ("Run time of %s: %s seconds" %(county, time.time()-start_time))    
# %%
if __name__ == "__main__":
    main()     