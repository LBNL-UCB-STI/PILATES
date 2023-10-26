# %%
import pandas as pd
import numpy as np
import random
import config
from argparse import ArgumentParser
from os.path import exists as file_exists
import os
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
    args = parser.parse_args()


    sample_ratio = args.sample_ratio 
    mdir=args.mdir
    analysis_year=args.target_year
    scenario=args.scenarios
    fdir_tour= mdir +"Tour_plan/"
    fdir_shipment= mdir +"Shipment2Fleet/"

    out_fdir_tour= mdir +"frism_light/Tour_plan/"
    out_fdir_shipment= mdir +"frism_light/Shipment2Fleet/"


    for ship_type in ["B2B", "B2C"]:
        for county in config.county_list:
            tour= pd.read_csv(fdir_tour+"{}_county{}_freight_tours_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
            carrier=pd.read_csv(fdir_tour+"{}_county{}_carrier_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
            payload= pd.read_csv(fdir_tour+"{}_county{}_payload_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
            list_tour = tour["tour_id"].unique().tolist()
            list_tour_sample= random.sample(list_tour, int(len(list_tour)*sample_ratio/100))
            sample_tour=tour[tour["tour_id"].isin(list_tour_sample)]
            sample_carrier=carrier[carrier["tourId"].isin(list_tour_sample)]
            sample_payload=payload[payload["tourId"].isin(list_tour_sample)]
            sample_payload_list = sample_payload["payloadId"].unique().tolist()
            if not file_exists(out_fdir_tour):
                os.makedirs(out_fdir_tour)        
            sample_tour.to_csv(out_fdir_tour+"{}_county{}_freight_tours_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
            sample_carrier.to_csv(out_fdir_tour+"{}_county{}_carrier_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
            sample_payload.to_csv(out_fdir_tour+"{}_county{}_payload_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)

            if ship_type == "B2B":
                shipment_payload= pd.read_csv(fdir_shipment+"{}_payload_county{}_shipall_A_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
                shipment_carrier= pd.read_csv(fdir_shipment+"{}_carrier_county{}_shipall_A_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
            else:
                shipment_payload= pd.read_csv(fdir_shipment+"{}_payload_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
                shipment_carrier= pd.read_csv(fdir_shipment+"{}_carrier_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), header=0, sep=',')
            sample_shipment_payload = shipment_payload[shipment_payload["payload_id"].isin(sample_payload_list)]
            sample_shipment_carrier = shipment_carrier[shipment_carrier["carrier_id"].isin(sample_shipment_payload["carrier_id"].unique().tolist())]
            if not file_exists(out_fdir_shipment):
                os.makedirs(out_fdir_shipment)        
            sample_shipment_payload.to_csv(out_fdir_shipment+"{}_payload_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)
            sample_shipment_carrier.to_csv(out_fdir_shipment+"{}_carrier_county{}_shipall_s{}_y{}.csv".format(ship_type,county,scenario,analysis_year), index = False, header=True)


if __name__ == "__main__":
    main()
