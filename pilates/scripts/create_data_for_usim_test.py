__test__ = False

import os
import yaml

from pilates.utils.settings_helper import get as get_setting

WARM_START_ACTIVITIES = True

if __name__ == "__main__":

    os.chdir("../..")
    from run import warm_start_activities, initialize_docker_client
    from pilates.urbansim import preprocessor as usim_pre

    with open("settings.yaml") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    settings.update({"docker_stdout": True})
    year = get_setting(settings, "run.start_year")

    client = initialize_docker_client(settings)

    if WARM_START_ACTIVITIES:
        warm_start_activities(settings, year, client)

    usim_pre.add_skims_to_model_data(settings)
