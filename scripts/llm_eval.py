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
        """Run the test case and return the results."""
        metric = RubricCorpusQaGenericMetric(self.metric_config["config"])
        resp = dict()
        resp["scores"] = metric.score_output(self.response)
        resp["case_id"] = self.case_id
        resp["annotator"] = self.annotator
        resp["agreement"] = self.agreement
        resp["question"] = self.initial_prompt
        return resp


class LlmEval:
    def __init__(self, test_config: List[Dict[str, Any]], responses: Dict[str, str], use_rubrics: bool,
                 use_snippets: bool):
        self.test_configs = test_config
        self.responses = responses
        self.use_snippets = use_snippets
        self.use_rubrics = use_rubrics

    def make_test_cases(
            self, skip_duplicate_annotations: bool = True
    ) -> List[TestCase]:
        """Create a list of ``TestCase`` objects by mapping each system response to each test query.
        param: skip_duplicate_annotations: Skip duplicate annotations for the same query.
        return: list of test cases
        """
        test_cases = []
        seen_agreements = set()
        for conf in self.test_configs:
            if conf["case_id"] not in self.responses:
                continue
            if not self.use_rubrics:
                for prop in conf["metric_config"]["config"]["other_properties"]:
                    prop["criterion"] = ""
            if not self.use_snippets:
                for prop in conf["metric_config"]["config"]["other_properties"]:
                    prop["evidence"] = []
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


def load_sys_responses(qa_files: List[str]) -> Dict[str, Dict[str, str]]:
    """Load system answers for each test query by iterating over the ``qa_files`` jsonl files.
        :param qa_files: list of paths to the jsonl files containing system responses
        :returns sys_response dict with keys as the qa_files names and values as the dict of test case_id to response
    """
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
        "--test-config",
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
        "--snippets",
        action="store_true",
        help="Score system for answer snippets",
        default=False,
    )
    parser.add_argument(
        "--rubrics",
        action="store_true",
        help="Score system for answer rubrics",
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

    # If src_names is provided, only evaluate those files, else all the jsonl files under args.qa_dir directory
    if args.src_names:
        srcs = [s.strip() for s in args.src_names.split(",")]
        qa_files = [f"{args.qa_dir}/{src}.jsonl" for src in srcs]
    else:
        qa_files = glob.glob(f"{args.qa_dir}/*.jsonl")
    qa_files.sort()
    print(f"{len(qa_files)} src files found: {qa_files}")

    sys_responses = load_sys_responses(qa_files)

    # Load the scoring rubrics and weights for each test case
    test_config = json.load(open(args.test_config, "r"))

    results_by_src = dict()
    qn_by_case = dict()

    # Evaluate each system under consideration for its responses to each test case
    for src, responses in sys_responses.items():
        llm_eval = LlmEval(test_config, responses, args.rubrics, args.snippets)
        print(f"Creating test cases for src: {src}...")
        test_cases = llm_eval.make_test_cases(
            skip_duplicate_annotations=(not args.agreement)
        )
        print(f"Created {len(test_cases)} tests for src: {src}...")

        for test_case in test_cases:
            qn_by_case[test_case.case_id] = test_case.initial_prompt
        results_by_src[src] = []

        # Evaluate all the test cases in parallel
        print(f"Running test cases for src: {src}...")
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_test_case = {
                executor.submit(test_case.run): test_case for test_case in test_cases
            }
            for future in tqdm(
                    as_completed(future_to_test_case), total=len(test_cases)
            ):
                results_by_src[src].append(future.result())
        results_by_src[src].sort(key=lambda x: (x["annotator"], x["case_id"]))

    with open(args.output, "w") as f:
        json.dump(results_by_src, f)
        print(f"Results written to {args.output}\n")

    for src, results in results_by_src.items():
        print(
            f'Avg score for src={src}: {round(statistics.mean([res["scores"]["score"] for res in results]), 3)}'
        )

    # Note: "agreement" here is not actually agreement in the annotators'
    # labels, but rather the agreement in the scores their rubrics assign to
    # the same response.
    if args.agreement:
        qn_results = dict()
        print("\nCalculating agreement...")
        for src, results in results_by_src.items():
            for res in results:
                if res["agreement"]:
                    if res["question"] not in qn_results:
                        qn_results[res["question"]] = [[], []]
                    if res["annotator"] == "Annotator 1 Assignments":
                        qn_results[res["question"]][0].append(res["scores"]["ann_score"])
                    else:
                        qn_results[res["question"]][1].append(res["scores"]["ann_score"])
        ktaus, pcorr = [], []
        for qn, scores in qn_results.items():
            ann1, ann2 = [x for x in scores[0]], [x for x in scores[1]]
            if len(set(ann1)) ==1 or len(set(ann2)) == 1:
                ktaus.append(0.0)
                pcorr.append(0.0)
            else:
                ktaus.append(np.abs(scipy.stats.kendalltau(ann1, ann2)[0]))
                pcorr.append(np.abs(scipy.stats.pearsonr(ann1, ann2)[0]))
        print(f"\nAgreement metrics across {len(ktaus)} cases:")
        print(f"   Pearson corr: {round(np.mean(pcorr), 3)}")
        print(f"   Kendall tau: {round(np.mean(ktaus), 3)}")


if __name__ == "__main__":
    main()
