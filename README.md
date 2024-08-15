# multidoc_qa_eval

**Update** 

Aug 16 2024: Incorporated the snippets/quotes from the annotations along with the rubrics for fine-grained eval. 
Changed parameter `--rubrics` to `--test-config` in `scripts/llm_eval.py`. Also, new `test_configs_snippets.json` and `output_snippets.jsonl` files.

### Setup

```python
conda create -n qa_eval python=3.10.0
conda activate qa_eval
pip install -r requirements.txt
```

### Data Files

- ``output_snippets.jsonl`` : Contains the questions with system responses for eval (and some other metadata used to generate the test cases but not required for subsequent runs). Should not require further modification.
- `test_configs_snippets.json` : A collection of test cases in json format with associated rubrics for each question. Each question has its own test case and rubrics. Should not require further modification.
- `qa_metadata_all.jsonl` : Metadata file that was used to bootstrap this utility. Should not require further modification.
- `src_answers`: Directory containing sample system responses from 4 systems.
### Eval

To run eval for your system, first setup the prediction file with the system answers to be evaluated as per following requirement:

A jsonl file with fields `case_id` and `answer_text` (See [example file](https://github.com/allenai/multidoc_qa_eval/blob/main/data/src_answers/gpt.jsonl)) -

- `case_id` corresponds to the identifier of the question for which the response is to be evaluated, map the question text with the case_id in `test_configs_snippets.json`
- `answer_text` is the system answer (along with citations and excerpts, if applicable) in plain text

Once the prediction json file is ready, save it a new directory run the eval script as follows (You can save as many system response files under a directory, they will be picked together for eval):

```python
export OPENAI_API_KEY=<openai key>
python scripts/llm_eval.py --qa-dir <prediction jsonl file directory> --test-config data/test_configs_snippets.json --rubrics --snippets --src-names <optional comma separated src names prefixes of prediction files with .jsonl, if not given all the files will be picked>
```
**Note** To evaluate only using rubrics, remove `--snippets` parameter and vice-versa to use only snippets. 

## License
The aggregate test cases, sample system answers under `data/src_answers` and other files under data directory are released under [ODC-BY](https://opendatacommons.org/licenses/by/1.0/) license. By downloading this data you acknowledge that you have read and agreed to all the terms in this license.
For constituent datasets, also go through the individual licensing requirements, as applicable. 
