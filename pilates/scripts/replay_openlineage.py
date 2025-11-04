"""
Utility to replay OpenLineage events from a .jsonl file to a Marquez HTTP endpoint.

This script is useful for capturing OpenLineage events on a system without direct
access to a Marquez instance (e.g., an HPC environment) and then replaying them
locally for visualization and debugging.

Usage:
    python replay_openlineage.py --file /path/to/your/openlineage.jsonl --url http://localhost:5002
"""

import argparse
import json
import logging
import requests
import time

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def replay_events(file_path: str, marquez_url: str):
    """Reads a .jsonl file and sends each line as a POST request to the Marquez API.

    Args:
        file_path (str): The path to the openlineage.jsonl file.
        marquez_url (str): The URL of the Marquez API endpoint (e.g., http://localhost:5002/api/v1/lineage).
    """
    logging.info(f"Starting replay of events from '{file_path}' to '{marquez_url}'")
    
    headers = {"Content-Type": "application/json"}
    event_count = 0
    success_count = 0

    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line_num = i + 1
                if not line.strip():
                    continue

                try:
                    # Validate that the line is valid JSON
                    event_data = json.loads(line)
                    event_count += 1

                    # Send the event to Marquez
                    response = requests.post(marquez_url, data=line, headers=headers)

                    if response.status_code == 201 or response.status_code == 200:
                        logging.info(f"Successfully sent event {line_num} (RunID: {event_data.get('run', {}).get('runId')})")
                        success_count += 1
                    else:
                        logging.error(
                            f"Failed to send event {line_num}. Status: {response.status_code}, Response: {response.text}"
                        )

                except json.JSONDecodeError:
                    logging.error(f"Skipping line {line_num}: Invalid JSON.")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Failed to connect to Marquez at {marquez_url}. Aborting. Error: {e}")
                    # Stop replay on connection error
                    break
                
                # Add a small delay to avoid overwhelming the Marquez instance
                time.sleep(0.1)

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info(f"Replay finished. Sent {success_count}/{event_count} events successfully.")

def main():
    """Parses command-line arguments and starts the replay process."""
    parser = argparse.ArgumentParser(description="Replay OpenLineage events to a Marquez instance.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the openlineage.jsonl file."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5002/api/v1/lineage",
        help="The URL of the Marquez API endpoint. Defaults to http://localhost:5002/api/v1/lineage"
    )
    args = parser.parse_args()

    replay_events(args.file, args.url)

if __name__ == "__main__":
    main()
