#!/usr/bin/env python3
import re
import csv
import sys


def parse_slow_routing_log(input_file, output_file):
    # Pre-filter pattern - matches lines starting with timestamp and containing WARN and SLOW-ROUTING
    prefilter = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+WARN.*\[SLOW-ROUTING\]")

    # Detailed extraction pattern
    pattern = r"duration=(\d+)ms.*?origin=\(([-\d.]+),\s*([-\d.]+)\).*?dest=\(([-\d.]+),\s*([-\d.]+)\).*?distanceInMiles=\(([-\d.]+)\).*?withTransit=(\w+).*?requestedMode=(\S+).*?departureTime=(\d+).*?numVehicles=(\d+).*?numOptions=(\d+)"

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(
            [
                "duration_ms",
                "origin_x",
                "origin_y",
                "dest_x",
                "dest_y",
                "distanceInMiles",
                "withTransit",
                "requestedMode",
                "departureTime",
                "numVehicles",
                "numOptions",
            ]
        )

        for line in infile:
            # Quick pre-filter: only process lines with the right format
            if True:
                match = re.search(pattern, line)
                if match:
                    writer.writerow(
                        [
                            match.group(1),  # duration
                            match.group(2),  # origin_x
                            match.group(3),  # origin_y
                            match.group(4),  # dest_x
                            match.group(5),  # dest_y
                            match.group(6),  # distanceInMiles
                            match.group(7),  # withTransit
                            match.group(8),  # requestedMode
                            match.group(9),  # departureTime
                            match.group(10),  # numVehicles
                            match.group(11),  # numOptions
                        ]
                    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_logs.py <input_log_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    parse_slow_routing_log(input_file, output_file)
    print(f"Parsing complete. Output written to {output_file}")
