import sys

import openmatrix as omx
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 1:
        path = "pilates/activitysim/sfbay/data/skims.omx"
        dry_run = True
    else:
        path = sys.argv[1]
        if len(sys.argv) >= 3:
            dry_run = (sys.argv[2].lower() == 'true')
        else:
            print("Doing a try run. Add another argument of `false` to do a real run.")
            dry_run = True

    if dry_run:
        method = 'r'
    else:
        method = 'a'

    sk = omx.open_file(path, method)
    tables = sk.list_matrices()

    distSOV = np.array(sk['SOV_DIST__AM'])

    distRaw = np.array(sk['DIST'])

    dist = distRaw.copy()

    # dist = distSOV.copy()
    # dist[~(distSOV > 0)] = distRaw[~(distSOV > 0)]

    sz = dist.size

    for table in tables:
        if ('TOTIVT' in table) | ('WLK_TRN_WLK_IVT' in table):
            if ("HVY" in table) or ("COM" in table):
                modeMaxSpeed = 75.
            elif "LOC" in table:
                modeMaxSpeed = 50.
            else:
                modeMaxSpeed = 60.
            print("------------------")
            print("Looking at skim {0}".format(table))
            ivt_minutes = np.array(sk[table]) / 100.0
            spd_raw = dist / (ivt_minutes / 60.0)
            ignore = ivt_minutes <= 0
            sz_valid = sz - ignore.sum().sum()
            print("Finding {:0.2%} of speeds are 0, leaving them alone".format(ignore.sum() / sz))
            print("Finding {:0.2%} of remaining speeds are nan, replacing them with 30 mph".format(
                np.isnan(spd_raw[~ignore]).sum() / sz_valid))
            spd_raw[np.isnan(spd_raw)] = 30.0
            print("Finding {0:0.2%} of remaining speeds are too fast, replacing them with {1} mph".format(
                (spd_raw[~ignore] > modeMaxSpeed).sum() / sz_valid, modeMaxSpeed))
            spd_raw[(spd_raw > modeMaxSpeed) & ~ignore] = modeMaxSpeed
            print("Finding {:0.2%} of remaining speeds are too slow, replacing them with 3 mph".format(
                (spd_raw[~ignore] < 3).sum() / sz_valid))
            spd_raw[(spd_raw < 3) & ~ignore] = 3.0
            print("New mean speed {0}".format(spd_raw[~ignore].mean()))
            new_tt_minutes = dist / spd_raw * 60.0
            if not dry_run:
                print("Updating table!!")
                sk[table][~ignore] = new_tt_minutes[~ignore] * 100.0

        if (table.startswith("SOV") | table.startswith("HOV")) & ("TIME" in table):
            print("------------------")
            print("Looking at skim {0}".format(table))
            ivt_minutes = np.array(sk[table])
            spd_raw = dist / (ivt_minutes / 60.0)
            bad = ~(ivt_minutes >= 0)
            print("Finding {:0.2%} of speeds are 0, replacing them with 30".format(bad.sum() / sz))
            spd_raw[bad] = 35.0
            print("Finding {:0.2%} of remaining speeds are too fast, replacing them with 75 mph".format(
                (spd_raw[~bad] > 75).sum() / sz))
            spd_raw[(spd_raw > 75)] = 75.0
            print("Finding {:0.2%} of remaining speeds are too slow, replacing them with 5 mph".format(
                (spd_raw[~bad] < 5).sum() / sz))
            spd_raw[(spd_raw < 5)] = 5.0
            print("New mean speed {0}".format(spd_raw[~bad].mean()))
            new_tt_minutes = dist / spd_raw * 60.0
            if not dry_run:
                print("Updating table!!")
                sk[table][:] = new_tt_minutes

        if "_VTOLL_" in table:
            toll = np.array(sk[table])
            if "TOLL_VTOLL" in table:
                newToll = np.zeros_like(toll) + 7.0
            else:
                newToll = np.zeros_like(toll)
            if not dry_run:
                print("Updating table {0}!!".format(table))
                sk[table][:] = newToll

    sk.close()
