import argparse
import json
from typing import Optional, Dict, Any, List
from pydantic.v1 import BaseModel, Field
from corpusqa_rubric import RubricCorpusQaGenericMetric
import logging
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class TestCase(BaseModel):
    case_id: str = Field(description="The ID of the case.")
    annotator: str = Field(description="Annotator id for the question.")
    agreement: bool = Field(description="Indicator whether the question is annotated by multiple annotators.")

    initial_prompt: str = Field(
        description="The initial query from the user to the system."
    )
    metric_config: Dict[str, Any] = Field(
        description="The metric to use to score the response."
    )
    response: str = Field(description="The response from the system.")

    def run(self) -> Dict[str, Any]:
        metric = RubricCorpusQaGenericMetric(self.metric_config["config"])
        return metric.score_output(self.response)


class LlmEval:
    def __init__(self, case_file: str, qa_file: str):
        self.case_file = case_file
        self.qa_file = qa_file

    def make_test_cases(self, src: str = None) -> List[TestCase]:
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
                if conf["initial_prompt"] in seen_agreements:
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
        "--agreement",
        action='store_true',
        help="Calculate agreement between annotators",
        default=False,
    )

    parser.add_argument(
        "--src-names",
        type=str,
        help="names of the sources to evaluate (comma separated)",
        default=None
    )

    args = parser.parse_args()

    if not args.agreement:
        srcs = args.src_names.split(",") if args.src_names else []
        llm_evals = dict()
        if not srcs:
            llm_eval = LlmEval(args.rubrics, args.qa_file)
            llm_evals["single"] = llm_eval
        else:
            for src in srcs:
                llm_eval = LlmEval(args.rubrics, args.qa_file)
                llm_evals[src] = llm_eval

        results = dict()
        for src, llm_eval in llm_evals.items():
            print(f"Creating test cases for src: {src}...")
            test_cases = llm_eval.make_test_cases(src)
            results[src] = []
            print(f"Running test cases for src: {src}...")
            for test_case in tqdm(test_cases):
                results[src].append(test_case.run())
            print(f"Results for {src} source: {results[src]}")


if __name__ == "__main__":
    main()
