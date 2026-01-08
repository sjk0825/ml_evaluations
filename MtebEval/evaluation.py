import argparse
import yaml
import os
import mteb
import json

from typing import Dict
from mteb.cache import ResultCache

def main(config: Dict):

    encoder = mteb.get_model(config["model_name"])
    task = mteb.get_task(config["task"])

    prediction_folder = config["prediction_folder"]

    res = mteb.evaluate(
        encoder,
        task,
        prediction_folder=prediction_folder,
    )

    for result in res.task_results:
        print(result)
    
    return


if __name__ == '__main__':
    CONFIG_PATH='config_mteb.yml'

    parser = argparse.ArgumentParser(description="Load and print YAML configuration.")
    parser.add_argument('--config-path', type=str, default=CONFIG_PATH, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config_path = args.config_path

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("======================")
    print("Loaded configuration:")
    print("======================")

    print(json.dumps(config, indent=4))

    main(config)