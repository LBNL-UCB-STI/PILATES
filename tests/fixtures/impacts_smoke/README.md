# IMPACTS smoke test fixture

- `beam/output` contains 1,404 emissions skim rows sampled from the saved
  IMPACTS San Francisco Bay Area example, with the 24 corresponding BEAM road
  links. The sample includes all 24 hours and both passenger and freight traffic.
- `beam/input` contains the matched BEAM vehicle types, sample assignments
  containing only vehicle type IDs for the replay, and the OSM ways and nodes
  referenced by the road links. It also contains ATLAS vehicle, household, and
  person rows and linked FRISM carriers, tours, and payloads for fleet
  generation. The ATLAS sample uses reassigned person, household, and vehicle
  IDs and rounds home coordinates to 0.001 degrees.
- The EMFAC archive contains 2018 Alameda County project analysis, black
  carbon, and annual inventory rows from the IMPACTS example. It retains the
  vehicle classes used by the sample and the PTO class used to build power
  takeoff rates. It also includes matching 2018 statewide fallback rows and
  the road dust tables. Presim derives the inventories and rate files at runtime.
- `beam/input/vehicle-tech/dispersions` includes the nine actual ISRM cells
  intersecting the sample network and the AERMOD patterns for the nearest
  original site, `KOAK_2017`. The grid retains the original cell identifiers
  in `original_isrm` and remaps the working identifiers to 0–8.
- `inmap` and the NOx-to-NO2 archive contain real ground layer transfer
  coefficients selected and reindexed to those same nine cells.
- `county` contains the simplified 2018 U.S. Census TIGER/Line Alameda County
  boundary, saved in the format IMPACTS uses for its local boundary cache.

The `sample.osm.pbf` data is © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)
and available under the Open Database License.
