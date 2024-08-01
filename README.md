# multidoc_qa_eval

### Setup

```python
conda create -n qa_eval python=3.10.0
conda activate qa_eval
pip install -r requirements.txt
```

### Data Files

- ``output.jsonl`` : Contains the questions with system responses for eval (and some other metadata used to generate the test cases but not required for subsequent runs). Should not require further modification.
- `test_configs.json` : A collection of test cases in json format with associated rubrics for each question. Each question has its own test case and rubrics. Should not require further modification.
- `qa_metadata.jsonl` : Metadata file that was used to bootstrap this utility. Should not require further modification.
- `src_answers`: Directory containing sample system responses from 4 systems.
### Eval

To run eval for your system, first setup the prediction file with the system answers to be evaluated as per following requirement:

A jsonl file with fields `case_id` and `sources` -

- `case_id` corresponds to the identifier of the question for which the response is to be evaluated, map the question text with the case_id in `test_configs.json`
- `sources` is a list of system responses to be evaluated. Each list element should be a dict with keys `name` and `answer_text`

Once the prediction json file is ready, save it a new directory run the eval script as follows (You can save as many system response files under a directory, they can be picked together for eval):

```python
export OPENAI_API_KEY=<openai key>
python scripts/llm_eval.py --qa-dir <prediction jsonl file directory> --rubrics data/test_configs.json --src-names <optional comma separated src names prefixes of prediction files with .jsonl, if not given all the files will be picked>
```
