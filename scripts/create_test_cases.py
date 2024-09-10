import hashlib
import json
from typing import Any, Dict, List
from run_utils import extract_json_from_response, run_chatopenai
from litellm.caching import Cache
import litellm

from tqdm import tqdm

annotations = []
with open("data/output_snippets.jsonl", "r") as f:
    for line in f:
        annotations.append(json.loads(line))

test_cases = []

litellm.cache = Cache(type="disk", disk_cache_dir="./data/litellm_cache/")


def gpt_filter(query: str, ingredients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ingredients:
        return ingredients
    context = "\n\n".join([f"{i + 1}. Criterion: {x['text']}\nSupporting quotes: {x['snippets']}" for i, x in enumerate(ingredients)])
    system_prompt = f"""You will be given a question someone asked to a scientific assistant (in <question></question> tags).
    You will then be provided with a list of criterion numbered from 1 to {len(ingredients)} alon with supporting quotes from publications, 
    that the response to the question should satisfy in (in <criterion></criterion> tags).
    Your job is to filter out the criterion that are not directly relevant to answering the question.
    Output the required criterion index/ordinal as a JSON: {{"criterion": [1, 2, 4,...]}}."""
    user_prompt = f"""<question>{query}</question>\n<criterion>{context}</criterion>"""
    resp = run_chatopenai("gpt-4-turbo", system_prompt, user_prompt, json_mode=True)
    obj = extract_json_from_response(resp)
    if not obj:
        return ingredients
    obj["criterion"].sort()
    return [ingredients[i - 1] for i in obj["criterion"] if 0 <= (i - 1) < len(ingredients)]


for d in tqdm(annotations):
    qn = {
        "initial_prompt": d["question"],
        "metric_config": {
            "name": "rubric_corpusqa_generic",
            "config": {
                "question": d["question"],
                "low_length": 300,
                "high_length": 600,
                "length_weight": 0.05,
                "expertise_weight": 0.05,
                "citations_weight": 0.2,
                "excerpts_weight": 0.1,
                "other_properties": [],
            },
        },
        "case_id": hashlib.md5(
            str((d["question"], d["spreadsheet"]["id"])).encode("utf-8")
        )
        .digest()
        .hex(),
    }

    d["ingredients"]["most_important"] = gpt_filter(d["question"], d["ingredients"]["most_important"])
    d["ingredients"]["nice_to_have"] = gpt_filter(d["question"], d["ingredients"]["nice_to_have"])

    if (
            len(d["ingredients"]["nice_to_have"]) != 0
            and len(d["ingredients"]["most_important"]) != 0
    ):
        base_weight = 0.6 / (
                2 * len(d["ingredients"]["most_important"])
                + len(d["ingredients"]["nice_to_have"])
        )
        mostimp_weight = 2 * base_weight
        niceimp_weight = base_weight
    elif len(d["ingredients"]["most_important"]) != 0:
        mostimp_weight = 0.6 / len(d["ingredients"]["most_important"])
    elif len(d["ingredients"]["nice_to_have"]) != 0:
        print(d["spreadsheet"], d["ingredients_doc_link"])
        niceimp_weight = 0.6 / len(d["ingredients"]["nice_to_have"])
    else:
        raise ValueError("No rubric items")

    for item_idx, item in enumerate(d["ingredients"]["most_important"]):
        qn["metric_config"]["config"]["other_properties"].append(
            {
                "name": f"most_important_item_{item_idx}",
                "criterion": item["text"],
                "weight": mostimp_weight,
                "evidence": item["snippets"],
            }
        )

    for item_idx, item in enumerate(d["ingredients"]["nice_to_have"]):
        qn["metric_config"]["config"]["other_properties"].append(
            {
                "name": f"nice_to_have_item_{item_idx}",
                "criterion": item["text"],
                "weight": niceimp_weight,
                "evidence": item["snippets"],
            }
        )
    qn["annotator"] = d["spreadsheet"]["name"]
    qn["agreement"] = d["agreement"]
    test_cases.append(qn)

with open("data/test_configs_snippets_gpt.json", "w") as f:
    json.dump(test_cases, f, indent=2)
