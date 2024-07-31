import argparse
import json
from typing import Optional, Dict, Any, List
from pydantic.v1 import BaseModel, Field
from corpusqa_rubric import RubricCorpusQaGenericMetric
import logging

LOGGER = logging.getLogger(__name__)


class TestCase(BaseModel):
    case_id: str = Field(description="The ID of the case.")
    annotator: str = Field(description="Annotator id for the question.")
    agreement: bool = Field(description="Indicator whether the question is annotated by multiple annotators.")

    initial_prompt: str = Field(
        description="The initial query from the user to the system."
    )
    metric: RubricCorpusQaGenericMetric = Field(
        description="The metric to use to score the response."
    )
    response: str = Field(description="The response from the system.")

    def run(self) -> Dict[str, Any]:
        return self.metric.score_output(self.response)


class LlmEval:
    def __init__(self, case_file: str, qa_file: str):
        self.case_file = case_file
        self.qa_file = qa_file

    def make_test_cases(self, src: str = None) -> List[TestCase]:
        with open(self.case_file) as f:
            configs = json.load(f)

        responses = dict()
        with open(self.qa_file) as f:
            for line in f:
                qa = json.loads(line)
                if not src:
                    response = qa["sources"][0]["ans_text"]
                else:
                    for src_ans in qa["sources"]:
                        if src_ans["source"] == src:
                            response = src_ans["ans_text"]
                            break
                responses[qa["case_id"]] = response
        test_cases = []
        for conf in configs:
            conf["response"] = responses[conf["case_id"]]
            conf["metric"] = RubricCorpusQaGenericMetric(**conf["metric_config"])
            del conf["metric_config"]
            test_cases.append(TestCase(**conf))
        return test_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa_file",
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
        type=bool,
        help="Calculate agreement between annotators",
        default=False,
    )

    parser.add_argument(
        "--src_names",
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
                llm_eval = LlmEval(args.rubrics, args.qa_file, src)
                llm_evals[src] = llm_eval

        results = dict()
        for src, llm_eval in llm_evals.items():
            test_cases = llm_eval.make_test_cases()
            results[src] = []
            for test_case in test_cases:
                results[src].append(test_case.run())
            LOGGER.info(f"Results for {src} source: {results[src]}")


if __name__ == "__main__":
    main()
