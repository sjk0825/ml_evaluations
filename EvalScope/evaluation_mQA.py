import argparse
import yaml
import os
import json

from evalscope import TaskConfig, run_task
from evalscope.api.model import ModelOutput, GenerateConfig
from evalscope.models.mockllm import MockLLM

def load_custom_outputs(prediction_path):
    outputs = []
    with open(prediction_path) as f:
        for line in f:
            obj = json.loads(line)
            outputs.append(
                ModelOutput.from_content(
                    model="mockllm",
                    content=obj["prediction"]
                )
            )
    return outputs

def main(config):

    task_config = config["task_config"]
    prediction_path = config["prediction_path"]

    custom_outputs = load_custom_outputs(prediction_path)

    mock_model = MockLLM(
        model_name="mockllm",
        custom_outputs=custom_outputs,
        config=GenerateConfig()
    )

    task_cfg = TaskConfig(
        model=mock_model,
        datasets=task_config["datasets"],
        dataset_args=task_config["dataset_args"],
    )
    run_task(task_cfg=task_cfg)

    print(task_cfg)


if __name__ == '__main__':
    CONFIG_PATH='config_evalscope.yml'

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