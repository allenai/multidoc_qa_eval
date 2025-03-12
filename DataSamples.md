### Test Case Sample
[All Test cases](https://github.com/allenai/multidoc_qa_eval/blob/main/data/test_configs_snippets.json)

```json
  {
    "initial_prompt": "What publicly available datasets are typically used for evaluating type inference systems in python?",
    "metric_config": {
      "name": "rubric_corpusqa_generic",
      "config": {
        "question": "What publicly available datasets are typically used for evaluating type inference systems in python?",
        "low_length": 300,
        "high_length": 600,
        "length_weight": 0.05,
        "expertise_weight": 0.05,
        "citations_weight": 0.2,
        "excerpts_weight": 0.1,
        "other_properties": [
          {
            "name": "most_important_item_0",
            "criterion": "Near the beginning, the answer should briefly define what is the goal of using a type inference system for programming languages in general.",
            "weight": 0.13333333333333333,
            "evidence": [
              "Goal of type inference: Automatically deduce the most general type for each expression. Two key points: 1. Automatically inferring types: This means the programmer has to write no types, but still gets all the benefit from static typing 2. Inferring the most general type: This means we want to infer polymorphic types whenever possible"
            ]
          },
          {
            "name": "most_important_item_1",
            "criterion": "The answer should emphasize on the importance of an automatic type inference system for Python.",
            "weight": 0.13333333333333333,
            "evidence": [
              " its dynamic type system can lead to potential type errors, leading researchers to explore automatic type inference approaches for Python programs.",
              "In Python, which is dynamically typed, this determination takes place at runtime. To address potential ambiguities, developers can utilize type annotations, which explicitly specifies the expected data types of variables or function returns. As the complexity of software projects increases, programmers find it increasingly challenging to maintain consistent data types. In response to this challenge, both industry and academia have developed type inference tools and static type checkers."
            ]},
          {
            "name": "nice_to_have_item_0",
            "criterion": "The answer could explain different categories of methods for type inference in Python such as rule-based and ML-based approaches.",
            "weight": 0.06666666666666667,
            "evidence": [
              "Existing type inference approaches can be generally grouped into three categories, i.e., rule-based, supervised, and cloze-style approaches. The rule-based type inference approaches can ensure the accuracy of predicted variable types, but they suffer from low coverage problems caused by dynamic features and external calls. Supervised type inference approaches, while feature-agnostic and able to mitigate the low coverage problem, require large, high quality annotated datasets and are limited to pre-defined types. As zero-shot approaches, the cloze-style approaches reformulate the type inference problem into a fill-in-the-blank problem by leveraging the general knowledge in powerful pre-trained code models. However, their performance is limited since they ignore the domain knowledge from static typing rules which reflect the inference logic."
            ]
          }
        ]
      }
    },
    "case_id": "d44280651a6fb71d56ee96834e180fa6",
    "annotator": "Annotator 1 Assignments",
    "agreement": true
  },
```
Field descriptions:

- **initial_prompt/metric_config.config.question**: The user literature search query to a scientific assistant
- **low_length/high_length**: Avg expected length for the system response
- **length_weight**: Length is 5% of the overall score.
- **expertise_weight**: LLM alloted score for whether the response is expertise appropriate per the query. It accounts for 5% of the score.
- **citations_weight** and excerpts_weight: Each claim should be attributed with a citation and the citation should have excerpts for verification. They each account for 20% and 10% of the score respectively.
- **other_properties**: 60% of the score is alloted based on human annotated rubric ingredients, i.e. criterion that a good answer to the query should meet.
- **most_important_item_x**: Critical ingredient or criteria necessary to answer the query. The most important ingredients are weighed 2X compared to nice to have ingredients (described below). The associated human annotated evidence is used for scoring as well.
- **nice_to_have_item_x**: Helpful information that adds valuable context to the response.
- **case_id**: Unique id for the test case
- **annotator**: annotator id
- **agreement**: whether this query was double annotated for agreement calculation


### Expected System Response Sample

Helpful Pydantic Classes:
```python
class Citation(BaseModel):
    id: str
    snippets: List[str]
    metadata: Optional[Dict[str, Any]]    

class CQASection(BaseModel):
    title: str
    text: str
    citations: List[Citation]

class CQAResponse(BaseModel):
    sections: List[CQASection]
```
```json
{
    "sections":
    [
        {
            "title": "Introduction to Type Inference in Python",
            "text": "\nType inference is the process of automatically determining the data types of expressions in a programming language without requiring explicit type annotations. In Python, which is dynamically typed by nature, type inference systems have become increasingly important as the language has been adopted for larger, more complex codebases where type-related bugs can become costly and difficult to detect.\n\nPython 3.5 introduced type hints through PEP 484, allowing developers to optionally annotate their code with type information. However, manually adding these annotations to existing codebases can be labor-intensive. This is where type inference systems come in - they can automatically suggest or infer types for variables, function parameters, and return values in Python code.\n\nSeveral tools have emerged to support type checking and inference in Python, including mypy (developed by Dropbox), pytype (by Google), Pyre (by Facebook/Meta), and pyright (by Microsoft). These systems use various static analysis techniques to determine types, ranging from simple local inference to more complex whole-program analysis.\n\nThe evaluation of these type inference systems requires representative datasets that reflect real-world Python usage. Researchers and tool developers need these datasets to measure accuracy, precision, recall, and performance of their inference algorithms across diverse coding patterns and styles. <Model name=\"Anthropic\" version=\"claude-3-7-sonnet-20250219\">",
            "citations":
            []
        },
        {
            "title": "Major Publicly Available Datasets",
            "text": "\nHere are the key publicly available datasets used for evaluating Python type inference systems:\n\n- **ManyTypes4Py**: Currently the most widely used benchmark dataset, containing 5,382 Python projects with more than 869,000 type annotations. It was specifically designed for machine learning-based type inference and includes a pipeline for extracting type information from abstract syntax trees. The dataset has been cleaned to remove duplicate source code files, which helps eliminate biases in evaluation. <Paper corpusId=\"233210280\" paperTitle=\"(Mir et al., 2021)\" isShortName></Paper> <Paper corpusId=\"259951409\" paperTitle=\"(Peng et al., 2023)\" isShortName></Paper> <Paper corpusId=\"270878649\" paperTitle=\"(Wang et al., 2024)\" isShortName></Paper>\n\n- **Typilus Dataset**: Created by Allamanis et al., this dataset contains 600 Python projects from GitHub. The files are converted to graph representations that were originally designed for training the Typilus graph-based neural model. <Paper corpusId=\"216056383\" paperTitle=\"(Allamanis et al., 2020)\" isShortName></Paper> <Paper corpusId=\"246680113\" paperTitle=\"(Peng et al., 2021)\" isShortName></Paper>\n\n- **TypeWriter OSS**: A dataset released by Pradel et al. from their TypeWriter tool, collected from GitHub projects. The dataset focuses on Python3 projects with substantial public interest (at least 50 stars) and includes projects that use mypy as a dependency. <Paper corpusId=\"208909790\" paperTitle=\"(Pradel et al., 2019)\" isShortName></Paper> <Paper corpusId=\"248157108\" paperTitle=\"(Fried et al., 2022)\" isShortName></Paper>\n\n- **ProbPY**: A dataset published by Xu et al. that combines results from static analysis (PySonar2) and dynamic analysis. It provides variable names, annotations, and source code to generate contextual information. <Paper corpusId=\"235658605\" paperTitle=\"(Cui et al., 2021)\" isShortName></Paper> <Paper corpusId=\"11548488\" paperTitle=\"(Xu et al., 2016)\" isShortName></Paper>\n\n- **Typeshed**: A human-labeled dataset containing type annotations for Python standard libraries and third-party packages. It's considered more reliable due to its human-created annotations but only covers function parameters and return types. It also lacks contextual information due to frequent code updates. <Paper corpusId=\"235658605\" paperTitle=\"(Cui et al., 2021)\" isShortName></Paper>\n\n- **CrossDomainTypes4Py**: A more recent dataset designed specifically for cross-domain experiments in type inference. It's often used alongside ManyTypes4Py in evaluations. <Paper corpusId=\"251710434\" paperTitle=\"(Gruner et al., 2022)\" isShortName></Paper> <Paper corpusId=\"260512650\" paperTitle=\"(Elkobi et al., 2023)\" isShortName></Paper>\n\n- **BetterTypes4Py**: A high-quality subset derived from the ManyTypes4Py dataset, used for training more specialized models. <Paper corpusId=\"257623048\" paperTitle=\"(Wei et al., 2023)\" isShortName></Paper>\n\n- **InferTypes4Py**: A test set specifically created to evaluate models that may have been pre-trained on existing datasets. It's derived from the source code of type inference tools themselves (Typilus, Type4Py) and therefore contains code not used in pre-training data for models like CodeT5. <Paper corpusId=\"257623048\" paperTitle=\"(Wei et al., 2023)\" isShortName></Paper>\n\n- **TypePY**: A dataset collected from 4,577 top-starred GitHub repositories. <Paper corpusId=\"235658605\" paperTitle=\"(Cui et al., 2021)\" isShortName></Paper>\n\n- **CodeXGLUE-derived dataset**: Some researchers have developed benchmarks for type inference using the Python portion of the CodeXGLUE dataset, originally created for code-to-text tasks. <Paper corpusId=\"248157108\" paperTitle=\"(Fried et al., 2022)\" isShortName></Paper>",
            "citations":
            [
                {
                    "id": "(Mir et al., 2021)",
                    "snippets":
                    [
                        "Recently, Allamanis et al. [8] proposed the Typilus model, which is a graphbased neural model that predicts type annotations for Python. The Typilus model [8] is accompanied by a dataset that contains 600 Python projects. Moreover, the source code files of Typilus' dataset are converted to graph representations that are only suitable for training the Typilus model. The Many-Types4Py dataset provides JSON-formatted analyzed source [16] is not collected solely for the ML-based type inference task, meaning that a large number of projects in the dataset may not have type annotations at all, especially given the time that the dataset was created. Allamanis [12] showed that the Python-150K dataset suffers from code duplication despite the removal of project forks."
                    ],
                    "metadata":
                    {
                        "paper":
                        {
                            "corpus_id": 233210280,
                            "title": "ManyTypes4Py: A Benchmark Python Dataset for Machine Learning-based Type Inference",
                            "authors":
                            [
                                {
                                    "authorId": "143909797",
                                    "name": "Amir M. Mir"
                                },
                                {
                                    "authorId": "2046786069",
                                    "name": "Evaldas Latoskinas"
                                },
                                {
                                    "authorId": "2211882",
                                    "name": "Georgios Gousios"
                                }
                            ],
                            "year": 2021,
                            "venue": "IEEE Working Conference on Mining Software Repositories",
                            "n_citations": 22
                        },
                        "score": 0.90625
                    }
                }
            ]
        }
    ]
}
```
