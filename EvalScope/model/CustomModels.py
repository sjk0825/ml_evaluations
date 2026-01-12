import json
from typing import Dict
from evalscope.models import BaseModel

class PrecomputedModel(BaseModel):
    def __init__(self, prediction_path: str):

        self.pred_map = {}
        with open(prediction_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.pred_map[item["id"]] = item["prediction"]

    def generate(self, inputs: json, **kwargs) -> str:
        sample_id = inputs["id"]
        return self.pred_map[sample_id]