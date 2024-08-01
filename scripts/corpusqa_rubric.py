import itertools
import logging
from typing import Any, Dict, List

from pydantic.v1 import BaseModel, Field
from run_utils import extract_json_from_response, run_chatopenai

LOGGER = logging.getLogger(__name__)


class CorpusQaRubricPropertyConfig(BaseModel):
    name: str
    criterion: str
    weight: float


class CorpusQaRubricConfig(BaseModel):
    question: str
    low_length: int = 300
    high_length: int = 600
    length_weight: float = 0.05
    expertise_weight: float = 0.05
    citations_weight: float = 0.2
    excerpts_weight: float = 0.1
    other_properties: List[CorpusQaRubricPropertyConfig] = Field(default_factory=list)
    model_name: str = "gpt-4-turbo-2024-04-09"


class RubricCorpusQaGenericMetric:
    def __init__(self, config: Dict[str, Any]):
        self.config = CorpusQaRubricConfig.parse_obj(config)

    def _score_length(self, response: str, low_length, high_length) -> float:
        word_count = len(response.split())
        return 1 - (
            (max(min(high_length, word_count), low_length) - low_length)
            / (high_length - low_length)
        )

    def _score_property(self, response: str, question: str, prop: str) -> float:
        resp = run_chatopenai(
            self.config.model_name,
            system_prompt="""You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.  You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).

Return a score on a scale of 0 to 10 indicating how appropriate the response is based on the given criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON in the format: {{"score": x}}.""",
            user_prompt=f"""<question>{question}</question>\n<response>{response}</response>\n<criterion>{prop}</criterion>""",
            json_mode=True,
            max_tokens=100,
        )

        obj = extract_json_from_response(resp)
        if not obj:
            return 0.0

        return obj["score"] / 10.0

    def _score_citations_excerpts(self, response: str) -> Dict[str, float]:

        try:
            score_components = self._score_citations_excerpts_inner(response)
        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Could not extract citations and excerpts: {e}")
            score_components = {"citations": 0.0, "excerpts": 0.0}

        return score_components

    def _score_citations_excerpts_inner(self, response: str) -> Dict[str, float]:
        resp = run_chatopenai(
            self.config.model_name,
            system_prompt="You are a helpful assistant.",
            user_prompt=f"""Here is a response to a question that includes several claims and citations:
Response: {response}

Split the response into individual claims, citations, and excerpts from the citations, in JSON format: """
            '{"claims": [{"claim_text": "...", "citations": [{"citation_text": "...", "excerpts": ["...", ...]}, ...]}, ...]}'
            "\n\nIf a claim is missing citations or a citation is not accompanied by excerpts, some lists may be empty in your output.",
            json_mode=True,
        )

        extracted_json = extract_json_from_response(resp)
        if not extracted_json:
            return {"citations": 0.0, "excerpts": 0.0}
        citation_score = sum(
            1 for claim in extracted_json["claims"] if claim["citations"]
        ) / max(len(extracted_json["claims"]), 1)

        all_citations = list(
            itertools.chain.from_iterable(
                claim["citations"] for claim in extracted_json["claims"]
            )
        )
        excerpt_score = sum(
            1 for citation in all_citations if citation["excerpts"]
        ) / max(len(all_citations), 1)

        return {"citations": citation_score, "excerpts": excerpt_score}

    def score_output(self, response: str) -> Dict[str, Any]:
        score_weights = {
            "length": self.config.length_weight,
            "expertise": self.config.expertise_weight,
            "citations": self.config.citations_weight,
            "excerpts": self.config.excerpts_weight,
        }
        score_weights.update({x.name: x.weight for x in self.config.other_properties})
        assert abs(sum(score_weights.values()) - 1.0) < 1e-6

        score_components = dict()

        score_components["length"] = self._score_length(
            response,
            low_length=self.config.low_length,
            high_length=self.config.high_length,
        )
        score_components["expertise"] = self._score_property(
            response,
            self.config.question,
            "The level of expertise required to understand the answer should be roughly aligned with the estimated expertise of a typical person who would ask the question.",
        )
        score_components.update(self._score_citations_excerpts(response))

        for x in self.config.other_properties:
            score_components[x.name] = self._score_property(
                response, self.config.question, x.criterion
            )

        assert set(score_components.keys()) == set(score_weights.keys())
        score = sum(score_weights[key] * score_components[key] for key in score_weights)
        return {"score": score, **score_components}
