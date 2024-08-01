import argparse
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import litellm
import numpy as np
import scipy.stats
from corpusqa_rubric import RubricCorpusQaGenericMetric
from litellm.caching import Cache
from pydantic.v1 import BaseModel, Field
from tqdm import tqdm
import glob

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
        resp["agreement"] = self.agreement
        return resp


class LlmEval:
    def __init__(self, test_config: List[Dict[str, Any]], responses: Dict[str, str]):
        self.test_configs = test_config
        self.responses = responses

    def make_test_cases(
            self, skip_duplicate_annotations=True
    ) -> List[TestCase]:

        test_cases = []
        seen_agreements = set()
        for conf in self.test_configs:
            if conf["case_id"] not in self.responses:
                continue
            conf["response"] = self.responses[conf["case_id"]]
            if (
                    conf["initial_prompt"] in seen_agreements
                    and skip_duplicate_annotations
            ):
                continue
            seen_agreements.add(conf["initial_prompt"])
            test_cases.append(TestCase(**conf))
        return test_cases


def calculate_icc(scores1, scores2):
    """Calculate the intraclass correlation"""
    score_pairs = list(zip(scores1, scores2))
    n = len(score_pairs)
    grand_mean = np.mean(scores1 + scores2)

    sum1 = sum((x - grand_mean) ** 2 for x in scores1)
    sum2 = sum((x - grand_mean) ** 2 for x in scores2)
    s2 = (sum1 + sum2) / (2 * n - 1)

    icc = sum((x - grand_mean) * (y - grand_mean) for x, y in score_pairs) / (
            (n - 1) * s2
    )
    return icc


def load_sys_responses(qa_files):
    sys_responses = dict()
    for qa_file in qa_files:
        curr_responses = dict()
        print(f"Reading responses for source {qa_file}...")
        with open(qa_file) as f:
            for line in f:
                qa = json.loads(line)
                curr_responses[qa["case_id"]] = qa["answer_text"]
        print(f"{len(curr_responses)} responses obtained for eval...")
        sys_responses[qa_file] = curr_responses
    return sys_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-dir",
        type=str,
        required=True,
        help="Directory containing *.jsonl files with queries and responses from system(s) to be evaluated. "
             "(All the files will be picked, to filter to specific files provide src-names param)",
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
        help="names of the source files to evaluate (comma separated with .jsonl extension)",
        default=None,
    )

    args = parser.parse_args()

    litellm.cache = Cache(type="disk", disk_cache_dir="./data/litellm_cache/")

    if args.src_names:
        srcs = [s.strip() for s in args.src_names.split(",")]
        qa_files = [f"{args.qa_dir}/{src}.jsonl" for src in srcs]
    else:
        qa_files = glob.glob(f"{args.qa_dir}/*.jsonl")
    qa_files.sort()
    print(f"{len(qa_files)} src files found: {qa_files}")

    sys_responses = load_sys_responses(qa_files)
    test_config = json.load(open(args.rubrics, "r"))

    results_by_src = dict()
    qn_by_case = dict()
    for src, responses in sys_responses.items():
        llm_eval = LlmEval(test_config, responses)
        print()
        print(f"Creating test cases for src: {src}...")
        test_cases = llm_eval.make_test_cases(
            skip_duplicate_annotations=(not args.agreement)
        )
        print(f"Created {len(test_cases)} tests for src: {src}...")
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
        print(f"Results written to {args.output}\n")

    for src, results in results_by_src.items():
        print(
            f'Avg score for src={src}: {statistics.mean([res["scores"]["score"] for res in results])}'
        )

    # Note: "agreement" here is not actually agreement in the annotators'
    # labels, but rather the agreement in the scores their rubrics assign to
    # the same response.
    if args.agreement:
        print("\nCalculating agreement...")
        results_by_annotator = dict()
        for src, results in results_by_src.items():
            for res in results:
                res["src"] = src
                if res["agreement"]:
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
            return (qn_by_case[x["case_id"]], x["src"])

        # union_ids = set([casekey(x) for x in anno1]) | set([casekey(x) for x in anno2])
        # overlapping_ids = set([casekey(x) for x in anno1]) & set(
        #     [casekey(x) for x in anno2]
        # )
        # if len(overlapping_ids) != len(union_ids):
        #     print(
        #         f"Pruned {len(union_ids) - len(overlapping_ids)} cases from one or both annotators"
        #     )
        #
        # anno1 = [x for x in anno1 if casekey(x) in overlapping_ids]
        # anno2 = [x for x in anno2 if casekey(x) in overlapping_ids]

        anno1.sort(key=casekey)
        anno2.sort(key=casekey)

        if [casekey(x) for x in anno1] != [casekey(x) for x in anno2]:
            raise ValueError(f"Got different cases for the two annotators")

        scores1 = [x["scores"]["score"] for x in anno1]
        scores2 = [x["scores"]["score"] for x in anno2]

        print(f"\nAgreement metrics across {len(scores1)} cases:")
        print(f"   Pearson corr: {scipy.stats.pearsonr(scores1, scores2)}")
        print(f"    Kendall tau: {scipy.stats.kendalltau(scores1, scores2)}")
        print(f"Intraclass corr: {calculate_icc(scores1, scores2):.4f}")


if __name__ == "__main__":
    main()
