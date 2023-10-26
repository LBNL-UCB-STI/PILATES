import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-rt", "--run type", dest="run_type",
            help="initial or main ", required=True, type=str)
parser.add_argument("-md", "--directory", dest="main_dir",
            help="where the population restuls are located ", required=True, type=str)
parser.add_argument("-sn", "--scenario", dest="scenario",
                help="scenario", required=True, type=str)                
parser.add_argument("-yt", "--analysis year", dest="target_year",
            help="20XX", required=True, type=int) 
parser.add_argument("-isr", "--initial_sample_rate", dest="int_sample_rate",
            help="0-100 ", default=10, type=int)
parser.add_argument("-msr", "--main_sample_rate", dest="main_sample_rate",
            help="0-100 ", default=40, type=int)                             
args = parser.parse_args()

if args.run_type == "initial":
    '''
    This is to create x % (-sr x) of sample from the tour_plan and shipment results at the population level
    After running this, it will save x % of sample in "frism_light/Tour_plan/" and "frism_light/Shipment2Fleet/" under the mian result folder (-md )
    The results folder at the population level should be under the main folder (-md )
    '''
    os.system("python frism_light_initial_setting.py \
             -sr {0} \
               -md {1} \
                   -sn {2} \
                   -yt {3}".format(args.int_sample_rate, args.main_dir, args.scenario, args.target_year))
elif args.run_type == "main":
    '''
    For this, you should run "initial" just one time. Then, you can run this iteratively
    It is desinged to run each county and shipment type seperately. That means that the parallel run can generete results quickly.
    But for now, I set up for-loop.
    @ Haitam: you need to change the file directory regarding travel time and distance. "Austin_od_dist.csv" is back-up in case of failure to find travel time from tt_df_cbg.csv.gz
    tt_df_cbg.csv.gz is travel time table, thus it should be updated through interation.    

    '''
    ship_type= ["B2B", "B2C"]
    county_list = [453, 491, 209, 55, 21, 53]
    # ship_type= ["B2C"]
    # county_list = [21]


    veh_type_dir= args.main_dir+"Tour_plan/vehicle_types_s{}_y{}.csv".format(args.scenario, args.target_year)            
    for s in ship_type:
        for c in county_list:
            os.system("python frism_light_tour_update.py \
                        -sr {0} \
                        -md {1} \
                        -sn {2} \
                        -yt {3} \
                        -cy {4} \
                        -st {5} \
                        -t ../../../FRISM_input_output_AT/Sim_inputs/Geo_data/tt_df_cbg.csv.gz \
                        -d ../../../FRISM_input_output_AT/Sim_inputs/Geo_data/Austin_od_dist.csv\
                        -ct ../../../FRISM_input_output_AT/Sim_inputs/Geo_data/Austin_freight_centroids.geojson \
                        -vt {6}".format(args.main_sample_rate,args.main_dir, args.scenario, args.target_year, c,s, veh_type_dir))
else:
    print ("please provide a correct run type")    
   