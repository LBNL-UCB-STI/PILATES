## Configuring PILATES to run FRISM-light model

1. Add the following parameters to `settings.yaml`.
```yaml
# simulation platforms (leave blank to turn off)
commerce_demand_model: frism

# docker settings
docker_images:
    frism: dimaopen/frism-light:0.1.0
    beam: beammodel/beam:0.8.6.13
    
# BEAM
beam_freight_folder: freight/

# FRISM
frism_data_folder: pilates/frism/data
initialize_frism_light: False
```
2. Put your frism input data to `{frism_data_folder}/{region}`. For example `pilates/frism/data/austin` should contain
4 folders:
- frism_light
- Geo_data
- Shipment2Fleet
- Tour_plan
3. Beam config should contain the following parameters
```hocon
beam.exchange {
  output {
    activitySimSkimsEnabled = true
    # geo level different than TAZ (in beam taz-centers format)
    geo.filePath = ${beam.inputDirectory}"/frism-blockgroup-centers.csv.gz"
  }
}
beam.router.skim.activity-sim-skimmer.fileOutputFormat = "omx"
# This allows to write skims on each iteration
beam.router.skim.writeSkimsInterval = 1

beam.agentsim.agents.freight {
  enabled = true
  plansFilePath = ${beam.inputDirectory}"/freight/freight-payload-plans.csv"
  toursFilePath = ${beam.inputDirectory}"/freight/freight-tours.csv"
  carriersFilePath = ${beam.inputDirectory}"/freight/freight-carriers.csv"
  vehicleTypesFilePath = ${beam.inputDirectory}"/freight/freight-vehicles-types.csv"
  isWgs = true
}
```

- `beam.exchange.geo.filePath` must contain path to a csv file in beam taz-centers format with GEOIDs and coordinates of
geo units used in FRISM-light (if they are different to the ones BEAM is using).
- `beam.router.skim.activity-sim-skimmer.fileOutputFormat` must be "omx" because only this format is well tested in
PILATES.
- `plansFilePath`, `toursFilePath`, `carriersFilePath`, `vehicleTypesFilePath` should point to the same folder. And the
last file names should be `freight-payload-plans.csv`, `freight-tours.csv`, `freight-carriers.csv`, 
`freight-vehicles-types.csv` respectively. Because frism postprocessor writes resulting freight plans to the folder
defined by `beam_freight_folder` in `setting.yaml` and with exact those last file names.

### Data exchange during runtime
When PILATES is run with FRISM-light enabled frism preprocessor reads beam skims and merges them to
`{frism_data_folder}/{region}/Geo_data/tt_df_cbg.csv.gz`. At starting year it reads beam skims from
`beam_local_output_folder/skims_fname`. At all the subsequent years it reads them from the latest beam results that are
in `beam_local_output_folder`.

After FRISM model finishes frism postprocessor writes resulting freight plans to
`{beam_local_input_folder}/{region}/{beam_freight_folder}/` folder. And Beam should be configured to read freight plans
from that folders.

### FRISM-light initialization
At starting year FRISM-light init run happens in case
- `initialize_frism_light` parameter is set to True.

**OR**
- No beam skims were found.

Main FRISM-light run happens every iteration.