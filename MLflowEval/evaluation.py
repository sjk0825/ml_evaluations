import argparse
import yaml
import os
import json

from typing import Dict
from manager.MLFlowManager import MLflowLogger
from openai import OpenAI


def main(config: Dict):

    # set mlflow
    config_setting = config["mlflow_setting"]
    config_evaluation = config["mlflow_evaluation"]

    mlflowManager = MLflowLogger(experiment_name=config_setting["experiment_name"],
                                tracking_uri=config_setting["tracking_uri"],
                                run_name=config_setting["run_name"],
                                tags=None,
                                scorers=config_evaluation["scorers"])
    
    # load data
    config_data = config["data"]
    with open(config_data["input"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    # run evaluate
    mlflowManager.evaluate(dataset=data, answer_generator=qa_predict_fn)
    
    return

client = OpenAI()
def qa_predict_fn(question: str) -> str:
    """Simple Q&A prediction function using OpenAI"""
    global client

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely (smaller than 10 character). 그리고 반드시 한글로 대답해야해.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    CONFIG_PATH='config_mlflow.yml'

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