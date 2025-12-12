import openmatrix as omx
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Log to console
)
logger = logging.getLogger(__name__)

# --- File Paths ---
file1_path = "pilates/beam/production/seattle/as-base-skims-seattle-bg.omx"
file2_path = "pilates/beam/production/seattle/as-base-skims-seattle-bg-new.omx"

# --- Skims to Copy ---
# This list is generated *strictly* based on the analysis of
# compare_skims.log, excluding those with _EXP_ or _HVY_
# as per the user's criteria for the target file.
skims_to_copy = [
    "BIKE_TRIPS",
    "DISTBIKE",
    "DISTWALK",
    "HOV2TOLL_BTOLL__AM",
    "HOV2TOLL_BTOLL__EA",
    "HOV2TOLL_BTOLL__EV",
    "HOV2TOLL_BTOLL__MD",
    "HOV2TOLL_BTOLL__PM",
    "HOV2TOLL_DIST__AM",
    "HOV2TOLL_DIST__EA",
    "HOV2TOLL_DIST__EV",
    "HOV2TOLL_DIST__MD",
    "HOV2TOLL_DIST__PM",
    "HOV2TOLL_TIME__AM",
    "HOV2TOLL_TIME__EA",
    "HOV2TOLL_TIME__EV",
    "HOV2TOLL_TIME__MD",
    "HOV2TOLL_TIME__PM",
    "HOV2TOLL_VTOLL__AM",
    "HOV2TOLL_VTOLL__EA",
    "HOV2TOLL_VTOLL__EV",
    "HOV2TOLL_VTOLL__MD",
    "HOV2TOLL_VTOLL__PM",
    "HOV2_BTOLL__AM",
    "HOV2_BTOLL__EA",
    "HOV2_BTOLL__EV",
    "HOV2_BTOLL__MD",
    "HOV2_BTOLL__PM",
    "HOV2_DIST__AM",
    "HOV2_DIST__EA",
    "HOV2_DIST__EV",
    "HOV2_DIST__MD",
    "HOV2_DIST__PM",
    "HOV2_TIME__AM",
    "HOV2_TIME__EA",
    "HOV2_TIME__EV",
    "HOV2_TIME__MD",
    "HOV2_TIME__PM",
    "HOV2_VTOLL__AM",
    "HOV2_VTOLL__EA",
    "HOV2_VTOLL__EV",
    "HOV2_VTOLL__MD",
    "HOV2_VTOLL__PM",
    "HOV3TOLL_BTOLL__AM",
    "HOV3TOLL_BTOLL__EA",
    "HOV3TOLL_BTOLL__EV",
    "HOV3TOLL_BTOLL__MD",
    "HOV3TOLL_BTOLL__PM",
    "HOV3TOLL_DIST__AM",
    "HOV3TOLL_DIST__EA",
    "HOV3TOLL_DIST__EV",
    "HOV3TOLL_DIST__MD",
    "HOV3TOLL_DIST__PM",
    "HOV3TOLL_VTOLL__AM",
    "HOV3TOLL_VTOLL__EA",
    "HOV3TOLL_VTOLL__EV",
    "HOV3TOLL_VTOLL__MD",
    "HOV3TOLL_VTOLL__PM",
    "HOV3_TIME__AM",
    "HOV3_TIME__EA",
    "HOV3_TIME__EV",
    "HOV3_TIME__MD",
    "HOV3_TIME__PM",
    "HOV3_VTOLL__AM",
    "HOV3_VTOLL__EA",
    "HOV3_VTOLL__EV",
    "HOV3_VTOLL__MD",
    "HOV3_VTOLL__PM",
    "RH_POOLED_REJECTIONPROB__AM",
    "RH_POOLED_REJECTIONPROB__EA",
    "RH_POOLED_REJECTIONPROB__EV",
    "RH_POOLED_REJECTIONPROB__MD",
    "RH_POOLED_REJECTIONPROB__PM",
    "RH_POOLED_TRIPS__AM",
    "RH_POOLED_TRIPS__EA",
    "RH_POOLED_TRIPS__EV",
    "RH_POOLED_TRIPS__MD",
    "RH_POOLED_TRIPS__PM",
    "RH_POOLED_WAIT__AM",
    "RH_POOLED_WAIT__EA",
    "RH_POOLED_WAIT__EV",
    "RH_POOLED_WAIT__MD",
    "RH_POOLED_WAIT__PM",
    "RH_SOLO_REJECTIONPROB__AM",
    "RH_SOLO_REJECTIONPROB__EA",
    "RH_SOLO_REJECTIONPROB__EV",
    "RH_SOLO_REJECTIONPROB__MD",
    "RH_SOLO_REJECTIONPROB__PM",
    "RH_SOLO_TRIPS__AM",
    "RH_SOLO_TRIPS__EA",
    "RH_SOLO_TRIPS__EV",
    "RH_SOLO_TRIPS__MD",
    "RH_SOLO_TRIPS__PM",
    "RH_SOLO_WAIT__AM",
    "RH_SOLO_WAIT__EA",
    "RH_SOLO_WAIT__EV",
    "RH_SOLO_WAIT__MD",
    "RH_SOLO_WAIT__PM",
    "SOVTOLL_BTOLL__AM",
    "SOVTOLL_BTOLL__EA",
    "SOVTOLL_BTOLL__EV",
    "SOVTOLL_BTOLL__MD",
    "SOVTOLL_BTOLL__PM",
    "SOVTOLL_DIST__AM",
    "SOVTOLL_DIST__EA",
    "SOVTOLL_DIST__EV",
    "SOVTOLL_DIST__MD",
    "SOVTOLL_DIST__PM",
    "SOVTOLL_TIME__AM",
    "SOVTOLL_TIME__EA",
    "SOVTOLL_TIME__EV",
    "SOVTOLL_TIME__MD",
    "SOVTOLL_TIME__PM",
    "SOVTOLL_VTOLL__AM",
    "SOVTOLL_VTOLL__EA",
    "SOVTOLL_VTOLL__EV",
    "SOVTOLL_VTOLL__MD",
    "SOVTOLL_VTOLL__PM",
    "SOV_BTOLL__AM",
    "SOV_BTOLL__EA",
    "SOV_BTOLL__EV",
    "SOV_BTOLL__MD",
    "SOV_BTOLL__PM",
    "SOV_DIST__AM",
    "SOV_DIST__EA",
    "SOV_DIST__EV",
    "SOV_DIST__MD",
    "SOV_DIST__PM",
    "SOV_TIME__AM",
    "SOV_TIME__EA",
    "SOV_TIME__EV",
    "SOV_TIME__MD",
    "SOV_TIME__PM",
    "SOV_TRIPS__AM",
    "SOV_TRIPS__EA",
    "SOV_TRIPS__EV",
    "SOV_TRIPS__MD",
    "SOV_TRIPS__PM",
    "SOV_VTOLL__AM",
    "SOV_VTOLL__EA",
    "SOV_VTOLL__EV",
    "SOV_VTOLL__MD",
    "SOV_VTOLL__PM",
    "WALK_TRIPS",
    "WLK_TRN_WLK_BOARDS__AM",
    "WLK_TRN_WLK_BOARDS__EA",
    "WLK_TRN_WLK_BOARDS__EV",
    "WLK_TRN_WLK_BOARDS__MD",
    "WLK_TRN_WLK_BOARDS__PM",
    "WLK_TRN_WLK_FAILURES__AM",
    "WLK_TRN_WLK_FAILURES__EA",
    "WLK_TRN_WLK_FAILURES__EV",
    "WLK_TRN_WLK_FAILURES__MD",
    "WLK_TRN_WLK_FAILURES__PM",
    "WLK_TRN_WLK_FAR__AM",
    "WLK_TRN_WLK_FAR__EA",
    "WLK_TRN_WLK_FAR__EV",
    "WLK_TRN_WLK_FAR__MD",
    "WLK_TRN_WLK_FAR__PM",
    "WLK_TRN_WLK_IVT__AM",
    "WLK_TRN_WLK_IVT__EA",
    "WLK_TRN_WLK_IVT__EV",
    "WLK_TRN_WLK_IVT__MD",
    "WLK_TRN_WLK_IVT__PM",
    "WLK_TRN_WLK_IWAIT__AM",
    "WLK_TRN_WLK_IWAIT__EA",
    "WLK_TRN_WLK_IWAIT__EV",
    "WLK_TRN_WLK_IWAIT__MD",
    "WLK_TRN_WLK_IWAIT__PM",
    "WLK_TRN_WLK_KEYIVT__AM",
    "WLK_TRN_WLK_KEYIVT__EA",
    "WLK_TRN_WLK_KEYIVT__EV",
    "WLK_TRN_WLK_KEYIVT__MD",
    "WLK_TRN_WLK_KEYIVT__PM",
    "WLK_TRN_WLK_TRIPS__AM",
    "WLK_TRN_WLK_TRIPS__EA",
    "WLK_TRN_WLK_TRIPS__EV",
    "WLK_TRN_WLK_TRIPS__MD",
    "WLK_TRN_WLK_TRIPS__PM",
    "WLK_TRN_WLK_WACC__AM",
    "WLK_TRN_WLK_WACC__EA",
    "WLK_TRN_WLK_WACC__EV",
    "WLK_TRN_WLK_WACC__MD",
    "WLK_TRN_WLK_WACC__PM",
    "WLK_TRN_WLK_WAUX__AM",
    "WLK_TRN_WLK_WAUX__EA",
    "WLK_TRN_WLK_WAUX__EV",
    "WLK_TRN_WLK_WAUX__MD",
    "WLK_TRN_WLK_WAUX__PM",
    "WLK_TRN_WLK_WEGR__AM",
    "WLK_TRN_WLK_WEGR__EA",
    "WLK_TRN_WLK_WEGR__EV",
    "WLK_TRN_WLK_WEGR__MD",
    "WLK_TRN_WLK_WEGR__PM",
    "WLK_TRN_WLK_XWAIT__AM",
    "WLK_TRN_WLK_XWAIT__EA",
    "WLK_TRN_WLK_XWAIT__EV",
    "WLK_TRN_WLK_XWAIT__MD",
    "WLK_TRN_WLK_XWAIT__PM",
]


def copy_skims(src_path, dest_path, skims_list):
    """
    Copies a specified list of skims from a source OMX file to a destination OMX file,
    overwriting existing skims in the destination.

    Parameters
    ----------
    src_path : str
        Path to the source OMX file.
    dest_path : str
        Path to the destination OMX file.
    skims_list : list
        A list of skim names (strings) to copy.
    """
    if not os.path.exists(src_path):
        logger.critical(f"Source file not found: {src_path}")
        return
    if not os.path.exists(dest_path):
        logger.critical(f"Destination file not found: {dest_path}")
        return  # Or create it, depending on desired behavior. Sticking to copy to existing here.

    skims_src = None
    skims_dest = None

    try:
        logger.info(f"Opening source file: {src_path}")
        skims_src = omx.open_file(src_path, "r")

        logger.info(f"Opening destination file for append/write: {dest_path}")
        # Open in append mode to be able to write/overwrite
        skims_dest = omx.open_file(dest_path, "a")

        logger.info(f"Attempting to copy {len(skims_list)} specified skims.")

        for skim_name in skims_list:
            if skim_name in skims_src.list_matrices():
                logger.info(f"Processing skim: '{skim_name}'")
                try:
                    # If the skim exists in the destination, delete it first
                    if skim_name in skims_dest.list_matrices():
                        logger.info(
                            f"  Removing existing skim '{skim_name}' from destination."
                        )
                        del skims_dest[skim_name]

                    # Read data from source file
                    data_to_copy = skims_src[skim_name][
                        :
                    ]  # Use [:] to get the actual numpy array

                    # Write data to destination file
                    skims_dest[skim_name] = data_to_copy

                    logger.info(f"  Successfully copied '{skim_name}'.")

                except Exception as e:
                    logger.error(f"  Error copying skim '{skim_name}': {e}")
            else:
                logger.warning(
                    f"  Skim '{skim_name}' not found in source file '{src_path}'. Skipping."
                )

        logger.info("\n--- Copy Operation Summary ---")
        logger.info(f"Attempted to copy {len(skims_list)} skims.")
        # You could add counters here for success/failure if desired
        logger.info("----------------------------")

    except FileNotFoundError as e:
        logger.critical(f"File not found error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
    finally:
        if skims_src:
            try:
                skims_src.close()
                logger.info(f"Closed source file: {src_path}")
            except Exception as e:
                logger.error(f"Error closing source file {src_path}: {e}")
        if skims_dest:
            try:
                skims_dest.close()
                logger.info(f"Closed destination file: {dest_path}")
            except Exception as e:
                logger.error(f"Error closing destination file {dest_path}: {e}")


if __name__ == "__main__":
    # Note: The file paths are hardcoded at the top of the script,
    # making this a one-time script tailored to the log analysis.
    # You will need to manually run this script.
    logger.info("Starting skim copying script.")
    copy_skims(file1_path, file2_path, skims_to_copy)
    logger.info("Script finished.")
