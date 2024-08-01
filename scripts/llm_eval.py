import argparse
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import litellm
import scipy.stats
from corpusqa_rubric import RubricCorpusQaGenericMetric
from litellm.caching import Cache
from pydantic.v1 import BaseModel, Field
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class TestCase(BaseModel):
    case_id: str = Field(description="The ID of the case.")
    annotator: str = Field(description="Annotator id for the question.")
    agreement: bool = Field(
        description="Indicator whether the question is annotated by multiple annotators."
    )

    initial_prompt: str = Field(
        description="The initial query from the user to the system."
    )
    metric_config: Dict[str, Any] = Field(
        description="The metric to use to score the response."
    )
    response: str = Field(description="The response from the system.")

    def run(self) -> Dict[str, Any]:
        metric = RubricCorpusQaGenericMetric(self.metric_config["config"])
        resp = dict()
        resp["scores"] = metric.score_output(self.response)
        resp["case_id"] = self.case_id
        resp["annotator"] = self.annotator
        return resp


class LlmEval:
    def __init__(self, case_file: str, qa_file: str):
        self.case_file = case_file
        self.qa_file = qa_file

    def make_test_cases(
        self, src: str = None, skip_duplicate_annotations=True
    ) -> List[TestCase]:
        with open(self.case_file) as f:
            configs = json.load(f)

        responses = dict()

        print("Reading responses...")
        with open(self.qa_file) as f:
            for line in f:
                qa = json.loads(line)
                if src == "single":
                    response = qa["sources"][0]["answer_txt"]
                else:
                    for src_ans in qa["sources"]:
                        if src_ans["name"] == src:
                            response = src_ans["answer_txt"]
                            break
                responses[qa["case_id"]] = response
        print(f"{len(responses)} responses obtained for eval...")
        test_cases = []
        seen_agreements = set()
        for conf in configs:
            if conf["case_id"] not in responses:
                continue
            conf["response"] = responses[conf["case_id"]]
            if conf["agreement"]:
                if (
                    conf["initial_prompt"] in seen_agreements
                    and skip_duplicate_annotations
                ):
                    continue
                seen_agreements.add(conf["initial_prompt"])
            test_cases.append(TestCase(**conf))

        print(f"Created {len(test_cases)} tests for src: {src}...")
        return test_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-file",
        type=str,
        required=True,
        help="Jsonl file containing queries and responses from system(s) to be evaluated",
    )
    parser.add_argument(
        "--rubrics",
        type=str,
        required=True,
        help="Json file containing rubrics for all the questions to be evaluated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"./data/results_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json",
        help="output file to store the results of the evaluation",
    )
    parser.add_argument(
        "--agreement",
        action="store_true",
        help="Calculate agreement between annotators",
        default=False,
    )

    parser.add_argument(
        "--src-names",
        type=str,
        help="names of the sources to evaluate (comma separated)",
        default=None,
    )

    args = parser.parse_args()

    litellm.cache = Cache(type="disk", disk_cache_dir="./data/litellm_cache/")

    srcs = args.src_names.split(",") if args.src_names else []
    llm_evals = dict()
    if not srcs:
        llm_eval = LlmEval(args.rubrics, args.qa_file)
        llm_evals["single"] = llm_eval
    else:
        for src in srcs:
            llm_eval = LlmEval(args.rubrics, args.qa_file)
            llm_evals[src] = llm_eval

    results_by_src = dict()
    qn_by_case = dict()
    for src, llm_eval in llm_evals.items():
        print(f"Creating test cases for src: {src}...")
        test_cases = llm_eval.make_test_cases(
            src, skip_duplicate_annotations=(not args.agreement)
        )
        for test_case in test_cases:
            qn_by_case[test_case.case_id] = test_case.initial_prompt
        results_by_src[src] = []
        print(f"Running test cases for src: {src}...")
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_test_case = {
                executor.submit(test_case.run): test_case for test_case in test_cases
            }
            for future in tqdm(
                as_completed(future_to_test_case), total=len(test_cases)
            ):
                results_by_src[src].append(future.result())

    with open(args.output, "w") as f:
        json.dump(results_by_src, f)
        print(f"Results written to {args.output}")

    for src, results in results_by_src.items():
        print(
            f'Avg score for src={src}: {statistics.mean([res["scores"]["score"] for res in results])}'
        )

    if args.agreement:
        print("Calculating agreement...")
        results_by_annotator = dict()
        for src, results in results_by_src.items():
            for res in results:
                res["src"] = src
                if res["annotator"] not in results_by_annotator:
                    results_by_annotator[res["annotator"]] = []
                results_by_annotator[res["annotator"]].append(res)
        if len(results_by_annotator) != 2:
            raise ValueError(
                f"Need exactly 2 annotators to calculate agreement, got {len(results_by_annotator)}"
            )

        anno1 = results_by_annotator[list(results_by_annotator.keys())[0]]
        anno2 = results_by_annotator[list(results_by_annotator.keys())[1]]

        def casekey(x):
            # return (x["case_id"], x["src"])
            return (qn_by_case[x["case_id"]], x["src"])

        union_ids = set([casekey(x) for x in anno1]) | set([casekey(x) for x in anno2])
        overlapping_ids = set([casekey(x) for x in anno1]) & set(
            [casekey(x) for x in anno2]
        )
        if len(overlapping_ids) != len(union_ids):
            print(
                f"Pruned {len(union_ids) - len(overlapping_ids)} cases from one or both annotators"
            )

        anno1 = [x for x in anno1 if casekey(x) in overlapping_ids]
        anno2 = [x for x in anno2 if casekey(x) in overlapping_ids]

        anno1.sort(key=casekey)
        anno2.sort(key=casekey)

        if [casekey(x) for x in anno1] != [casekey(x) for x in anno2]:
            raise ValueError(f"Got different cases for the two annotators")

        scores1 = [x["scores"]["score"] for x in anno1]
        scores2 = [x["scores"]["score"] for x in anno2]
        print(f"Agreement metrics across {len(scores1)} cases:")
        print(f"        Pearson corr: {scipy.stats.pearsonr(scores1, scores2)}")
        print(f"         Kendall tau: {scipy.stats.kendalltau(scores1, scores2)}")


if __name__ == "__main__":
    main()
