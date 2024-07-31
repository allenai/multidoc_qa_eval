import hashlib
import json
from tqdm import tqdm

annotations = []
with open("data/output.jsonl", "r") as f:
    for line in f:
        annotations.append(json.loads(line))

test_cases = []

for d in tqdm(annotations):
    qn = {
        'initial_prompt': d['question'],
        'metric_config':
            {
                "name": "rubric_corpusqa_generic",
                "config": {
                    "question": d['question'],
                    "low_length": 300,
                    "high_length": 600,
                    "length_weight": 0.05,
                    "expertise_weight": 0.05,
                    "citations_weight": 0.2,
                    "excerpts_weight": 0.1,
                    "other_properties": [],
                }
            },
        'case_id': hashlib.md5(str((d['question'], d['spreadsheet']['id'])).encode('utf-8')).digest().hex(),
    }

    if len(d['ingredients']['nice_to_have']) != 0 and len(d['ingredients']['most_important']) != 0:
        base_weight = 0.6 / (2 * len(d['ingredients']['most_important']) + len(d['ingredients']['nice_to_have']))
        mostimp_weight = 2 * base_weight
        niceimp_weight = base_weight
    elif len(d['ingredients']['most_important']) != 0:
        mostimp_weight = 0.6 / len(d['ingredients']['most_important'])
    elif len(d['ingredients']['nice_to_have']) != 0:
        print(d['spreadsheet'], d['ingredients_doc_link'])
        niceimp_weight = 0.6 / len(d['ingredients']['nice_to_have'])
    else:
        print("No rubric items")

    for item_idx, item in enumerate(d['ingredients']['most_important']):
        qn['metric_config']['config']['other_properties'].append({
            "name": f"most_important_item_{item_idx}",
            "criterion": item,
            "weight": mostimp_weight,
        })

    for item_idx, item in enumerate(d['ingredients']['nice_to_have']):
        qn['metric_config']['config']['other_properties'].append({
            "name": f"nice_to_have_item_{item_idx}",
            "criterion": item,
            "weight": niceimp_weight,
        })
    qn["annotator"] = d["spreadsheet"]["name"]
    qn["agreement"] = d["agreement"]
    test_cases.append(qn)

with open("data/test_configs.json", "w") as f:
    json.dump(test_cases, f, indent=2)