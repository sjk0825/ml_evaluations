import argparse
import yaml
import os
import json

from typing import Dict
from manager.DeepevalManager import DeepevalManager
from utils.utils import groupby_conversation_id


def main(config: Dict):

    config_deepeval = config["deepeval"]
    evaluation_model = config_deepeval["evaluation_model"]
    evaluation_threshold = config_deepeval["evaluation_threshold"]
    evaluation_metrics = config_deepeval["evaluation_metrics"]

    deepevalManager = DeepevalManager(
        evaluation_model=evaluation_model,
        evaluation_threshold=evaluation_threshold,
        evaluation_metrics=evaluation_metrics
    )

    config_data = config["data"]
    conversation_id_grouped_data = groupby_conversation_id(config_data["input"])

    for key, value in conversation_id_grouped_data.items():
        results = deepevalManager.evaluate_application(eval_data_list=value)
        print(results)

    return


if __name__ == '__main__':
    CONFIG_PATH='config_deepeval.yml'

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