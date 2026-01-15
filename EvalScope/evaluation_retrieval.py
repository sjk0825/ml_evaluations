import argparse
import yaml
import os
import json

from evalscope import TaskConfig, run_task
from evalscope.api.model import ModelOutput, GenerateConfig
from evalscope.models.mockllm import MockLLM


def main(config):

    task_cfg = {
        "work_dir": config["work_dir"],
        "eval_backend": config["eval_backend"],
        "eval_config": {
            "tool": config["eval_config"]["tool"],
            "model": config["eval_config"]["model"],
            "eval": config["eval_config"]["eval"]
        },
    }
    run_task(task_cfg=task_cfg)

    print(task_cfg)


if __name__ == '__main__':
    CONFIG_PATH='config_evalscope_retrieval.yml'

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