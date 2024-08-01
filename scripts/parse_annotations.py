import argparse
import json
import os
import re
from typing import Callable

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tqdm import tqdm

S2_API_KEY = os.environ.get("S2_API_KEY")
# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]

CLIENT_SECRET_FILE = "cred.json"
TOKEN_FILE = "token.json"


def get_credentials():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


def format_nora_ans(nora_ans, question):
    nora_text = ""
    for section in nora_ans:
        header = f"{section['title']}\nTLDR: {section['tldr']}"
        text = section["text"]
        citations = section["citations"]
        cite_text = ""
        if citations:
            r = requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "externalIds,title"},
                json={
                    "ids": [
                        f"CorpusID:{citation['corpus_id']}" for citation in citations
                    ]
                },
                headers={"x-api-key": S2_API_KEY},
            )
            title_dict = {}
            if r.status_code == 200:
                rjson = r.json()
                for paper in rjson:
                    title_dict[int(paper["externalIds"]["CorpusId"])] = paper["title"]

            for j, citation in enumerate(citations):
                if citation["corpus_id"] in title_dict:
                    cite_text += f"{j + 1}. [{citation['id']} | n_citations: {citation['n_citations']} | {title_dict[citation['corpus_id']]} ]: \n{'... '.join(citation['snippets'])}\n\n"
            if not cite_text:
                print(question)
            text += "\n\nReferences:\n" + cite_text
        nora_text += f"{header}\n\n{text}\n\n"
    return nora_text


def list_spreadsheets(service, folder_id):
    results = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and name contains 'Annotator'",
            spaces="drive",
            fields="nextPageToken, files(id, name)",
        )
        .execute()
    )
    return results.get("files", [])


def read_spreadsheet(service, spreadsheet_id):
    # Assuming data starts after two header rows and you need columns A, B, and C
    ranges = ["A4:C28", "A31:C55", "A58:C62"]
    cumul_result = []
    for range_name in ranges:
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
        cumul_result += result.get("values", [])
    return cumul_result


def download_doc_content(docs_service, doc_id):
    doc = docs_service.documents().get(documentId=doc_id).execute()
    return doc


def para2txt(para: dict, map_fn: Callable[[str], str] = lambda x: x):
    return "".join([map_fn(x["textRun"]["content"]) for x in para["elements"]])


def element_to_markdown(elem: dict):
    elem_txt = elem["textRun"]["content"]
    if elem_txt.strip() == "":
        return elem_txt

    elem_style = elem["textRun"].get("textStyle", dict())

    if elem_style.get("fontSize", dict()).get("magnitude", 12) >= 15:
        if not elem_txt.strip().startswith("#"):
            elem_txt = f"## {elem_txt}"
        return elem_txt

    if elem_style.get("bold", False):
        if not elem_txt.strip().startswith("*"):
            pieces = ["", elem_txt, ""]
            m1 = re.search(r"^\s+", pieces[1])
            if m1:
                pieces[0] = m1.string[: m1.start()]
                pieces[1] = m1.string[m1.start() :]
            m2 = re.search(r"\s+$", pieces[1])
            if m2:
                pieces[1] = m2.string[: m2.start()]
                pieces[2] = m2.string[m2.start() :]
            pieces[1] = f"**{pieces[1]}**"
            elem_txt = "".join(pieces)

    if "url" in elem_style.get("link", dict()):
        url = elem_style["link"]["url"]
        if elem_txt.strip() != url.strip():
            elem_txt = f"[{elem_txt}]({url})"

    return elem_txt


def paragraph_to_markdown(para: dict, map_fn: Callable[[str], str] = lambda x: x):
    markdown = "".join([map_fn(element_to_markdown(x)) for x in para["elements"]])
    if "bullet" in para:
        markdown = f"- {markdown}"
    return markdown


def parse_ingredients_from_doc(doc):
    paragraphs = [x["paragraph"] for x in doc["body"]["content"] if "paragraph" in x]
    most_important_paras = []
    nice_to_have_paras = []
    other_paras = []
    cur_lst = other_paras

    for para in paragraphs:
        if len(para.get("elements", [])) >= 1:
            if "textRun" not in para["elements"][0]:
                continue
            if (
                para["elements"][0]["textRun"]["content"].lower().strip()
                == "most important"
            ):
                cur_lst = most_important_paras
            elif (
                para["elements"][0]["textRun"]["content"].lower().strip()
                == "nice to have"
            ):
                cur_lst = nice_to_have_paras
            elif "bullet" in para:
                cur_lst.append(para)

    return {
        "most_important": [para2txt(p, str.strip) for p in most_important_paras],
        "nice_to_have": [para2txt(p, str.strip) for p in nice_to_have_paras],
    }


def parse_sources_from_doc(doc):
    paragraphs = [x["paragraph"] for x in doc["body"]["content"] if "paragraph" in x]
    sources = []

    for para in paragraphs:
        if ("headingId" in para.get("paragraphStyle", dict())) or (
            para.get("elements")
            and para["elements"][0]["textRun"]
            .get("textStyle", dict())
            .get("fontSize", dict())
            .get("magnitude", 12)
            >= 15
        ):
            source_name = para2txt(para, lambda x: x.strip().strip(":"))
            if source_name.strip() == "":
                continue
            sources.append(
                {
                    "name": source_name,
                    "answer_paras": [],
                }
            )
        elif len(sources) != 0:
            sources[-1]["answer_paras"].append(para)

    sources = [x for x in sources if len(x["answer_paras"]) != 0]

    for source in sources:
        source["answer_txt"] = "".join(
            map(paragraph_to_markdown, source["answer_paras"])
        ).strip()
    return sources


def extract_doc_id_from_url(url):
    # Extract the document ID from the URL
    return url.split("/d/")[1].split("/")[0]


def get_sheet_data(service, spreadsheet_id, range_name):
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    values = result.get("values", [])
    return values


def get_nora_answer(sources_answers, question):
    for src_ans in sources_answers:
        if src_ans["name"] == "Nora":
            src_ans["answer_txt"] = format_nora_ans(src_ans["answer_txt"], question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-id",
        type=str,
        required=True,
        help="Google Drive folder ID containing the spreadsheets",
    )
    parser.add_argument(
        "--json-meta",
        type=str,
        help="Metadata JSON file containing the questions, links and source answers",
        default="data/qa_metadata_all.jsonl",
    )
    args = parser.parse_args()

    qa_metadata = []
    with open(args.json_meta, "r") as f:
        for line in f:
            qa_metadata.append(json.loads(line))

    qa_rev_idx = {
        qa_meta["question"].strip(): i for i, qa_meta in enumerate(qa_metadata)
    }
    agreement_qidx = [i for i in range(5)] + [i for i in range(25, 30, 1)]
    creds = get_credentials()
    drive_service = build("drive", "v3", credentials=creds)
    sheets_service = build("sheets", "v4", credentials=creds)
    docs_service = build("docs", "v1", credentials=creds)

    spreadsheets = list_spreadsheets(drive_service, args.folder_id)
    data = []

    print(f"Found {len(spreadsheets)} spreadsheets")

    for spreadsheet in spreadsheets:
        print(f"Processing spreadsheet: {spreadsheet['id']}")
        rows = read_spreadsheet(sheets_service, spreadsheet["id"])
        for row in tqdm(rows):
            if len(row) >= 3:
                question, doc_link, sources_link = row
                qidx = qa_rev_idx[question.strip()]
                print(f"Processing ingredients doc: {doc_link}")
                qmeta = qa_metadata[qidx]
                if qidx not in agreement_qidx:
                    qmeta["key_ingredients"] = [doc_link]
                ingredients_doc = download_doc_content(
                    docs_service,
                    extract_doc_id_from_url(doc_link),
                )
                try:
                    ingredients = parse_ingredients_from_doc(ingredients_doc)
                except:
                    print(f"Error parsing ingredients doc: {doc_link}")
                    ingredients = {"most_important": [], "nice_to_have": []}

                # print(f"Processing sources doc: {sources_link}")
                # sources_doc = download_doc_content(
                #     docs_service,
                #     extract_doc_id_from_url(sources_link),
                # )
                # sources_answers = parse_sources_from_doc(sources_doc)
                sources_answers = [
                    {"name": qsrc, "answer_txt": qans}
                    for qsrc, qans in qmeta["src_answers"].items()
                ]
                get_nora_answer(sources_answers, question)
                data.append(
                    {
                        "spreadsheet": spreadsheet,
                        "question": question,
                        "ingredients": ingredients,
                        "ingredients_doc_link": doc_link,
                        "sources": sources_answers,
                        "sources_doc_link": sources_link,
                        "agreement": False if qidx not in agreement_qidx else True,
                    }
                )

    with open("data/output.jsonl", "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
