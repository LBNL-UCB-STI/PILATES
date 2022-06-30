import numpy as np
import openmatrix as omx
import pandas as pd
import geopandas as gpd

skims_mtc = omx.open_file('../pilates/activitysim/data/skims-mtc.omx', mode = 'r')
skims_beam = omx.open_file('../pilates/activitysim/data/skims.omx', mode = 'r')
taz = gpd.read_file('/Users/zaneedell/Desktop/git/beam-data-sfbay/shape/sfbay-tazs-epsg-26910.shp')

def getRelationship(mtc, beam, idx):
    a = beam[idx, :]
    b = mtc[idx, :len(a)]
    shared = sum((a > 0) & (b > 0))
    if shared > 10:
        c, d, e, f = np.linalg.lstsq(a[(a > 0) & (b > 0), None], b[(a > 0) & (b > 0)])
        portionA = shared / sum((a > 0))
        portionB = shared / sum((b > 0))
        return np.array([c[0], d[0], portionA, portionB])
    else:
        return np.zeros(4) * np.nan


results = np.zeros((1454, 4))
col =  'WLK_LOC_WLK_WACC__AM'
mat_beam = np.array(skims_beam[col])
mat_mtc = np.array(skims_mtc[col])
for i in range(1454):
    results[i, :] = getRelationship(mat_mtc, mat_beam, i)
taz[col + '_corr'] = results[taz.taz1454-1, 0]
taz[col + '_pA'] = results[taz.taz1454-1, 2]
taz[col + '_pB'] = results[taz.taz1454-1, 3]
print(results[i, 0])