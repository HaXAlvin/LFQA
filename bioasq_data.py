import json
import sys
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.insert(0,str(path_root)+"/haystack")
from haystack.document_stores import FAISSDocumentStore
from haystack.schema import Document
from haystack.nodes import DensePassageRetriever
from tqdm import tqdm


def make_retriever(document_store, devices=["mps"]):
    return DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        devices=devices
    )

def get_bioasq_ori_data():
    with open("training10b.json", "r") as f:
        data = json.load(f)["questions"]
    all_snippets = []
    question = []
    answer = []
    for d in data:
        question.append(d["body"])
        answer.append(d["ideal_answer"])
        all_snippets += d["snippets"]
    return all_snippets, question, answer


def convert_bioasq_to_doc(article):
    content = article.pop("text")
    article["name"] = article.pop("document")
    return Document(
        content=content,
        meta=article
    )


def get_bioasq_data():
    all_snippets, question, answer = get_bioasq_ori_data()
    if Path("./db/bioasq/bioasq_data_faiss").exists():
        document_store = FAISSDocumentStore.load(index_path="./db/bioasq/bioasq_data_faiss")
        retriever = make_retriever(document_store)
    else:
        document_store = FAISSDocumentStore(sql_url="sqlite:///db/bioasq/doc_store.db", embedding_dim=128, faiss_index_factory_str="Flat", duplicate_documents="overwrite")
        all_docs = [convert_bioasq_to_doc(snippet) for snippet in tqdm(all_snippets, desc="Converting BioASQ data to Documents")]
        document_store.write_documents(all_docs)
        retriever = make_retriever(document_store)
        document_store.update_embeddings(retriever=retriever)
        document_store.save("./db/bioasq/bioasq_data_faiss")
    return document_store, retriever, question, answer

if __name__ == "__main__":
    from haystack.pipelines import DocumentSearchPipeline
    from haystack.utils import print_documents
    document_store, retriever, _, _ = get_bioasq_data()
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Is the protein Papilin secreted?", params={"Retriever": {"top_k": 10}})
    print_documents(res, max_text_len=512)