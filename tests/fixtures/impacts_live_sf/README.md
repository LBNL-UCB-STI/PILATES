# San Francisco live IMPACTS fixture

This fixture covers part of SoMa and the Financial District in San Francisco
County, California (FIPS 06075). The test uses one BEAM iteration of road
traffic and generates its fleet, skims, concentrations, and exposure in Docker.

- The road extract uses the San Francisco portion of the BREATHE OpenStreetMap
  input between 37.779° and 37.794° N and 122.4115° and 122.3915° W. It contains
  426 nodes and 581 ways after removing ways that cross the extract boundary.
  The TAZ shapefile and parking data retain zones intersecting that area.
- The ATLAS exchange contains eight sampled people and households, 11 vehicles,
  and their corresponding plans. Person, household, vehicle, trip, and tour
  identifiers were reassigned, coordinates rounded to 0.001°, and household
  incomes rounded to the nearest thousand dollars. The FRISM exchange retains
  two complete San Francisco tours and six payloads, with reassigned identifiers.
- The EMFAC archive retains 2018 San Francisco records for the vehicle classes
  and fuels needed by the sample. The archive also includes the matching
  statewide fallback records and road-dust inputs used by IMPACTS to derive
  rates. The regional inventory files retain their original export names.
- The InMAP fixture contains the ground-level coefficients for the nine source
  and receptor cells covering the road extract, with matching NOx-to-NO₂
  coefficients. AERMOD uses the nearest available patterns, `KOAK_2017`, on
  the San Francisco receptor grid.
- The San Francisco County boundary comes from the 2018 U.S. Census TIGER/Line
  county data. BEAM uses it to look up the county-specific rates, and IMPACTS
  uses the same boundary for allocation.

The road extract contains data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License.
