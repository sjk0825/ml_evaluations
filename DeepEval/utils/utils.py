import json
from typing import List, Dict, Any
from collections import defaultdict


def load_jsonl_to_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{path} is empty")
        data = json.loads(content)
    return data



def groupby_conversation_id(path: str) -> List[List[Dict[str, Any]]]:
    grouped = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            conversation_id = item["conversation_info"]["conversation_id"]
            grouped[conversation_id].append(item)
            
    return grouped