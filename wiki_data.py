from datasets import load_dataset
from tqdm import tqdm
import sys
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.insert(0,str(path_root)+"/haystack")
from haystack.document_stores import FAISSDocumentStore
from haystack.schema import Document
from haystack.nodes import DensePassageRetriever

def make_retriever(document_store, devices=["mps"]):
    return DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        devices=devices
    )

def convert_wiki_to_doc(article):
  meta = {"name": article['article_title'], "title": article['section_title']}
  return Document(content=article['passage_text'], meta=meta)

def create_wiki_data(batch_size=10000, max_docs=20000):
    document_store = FAISSDocumentStore(sql_url="sqlite:///db/wiki/doc_store.db", embedding_dim=128, faiss_index_factory_str="Flat", progress_bar=False, duplicate_documents="overwrite")
    print("Loading dataset...")
    wiki_snippets = load_dataset('wiki_snippets', name='wiki40b_en_100_0', cache_dir=".cache")["train"] #100k Wiki snippets, only train data
    print("Converting docs...")
    doc_buffer = []
    doc_count = 0
    with tqdm(total=max_docs) as pbar:
        for snippet in wiki_snippets:
            if doc_count >= max_docs:
                break
            doc_buffer.append(convert_wiki_to_doc(snippet))
            if len(doc_buffer) % batch_size == 0 or doc_count+len(doc_buffer) == max_docs:
                document_store.write_documents(doc_buffer)
                doc_count = document_store.get_document_count()
                pbar.n = doc_count
                pbar.refresh()
                doc_buffer.clear()
    print(f"Total document count: {document_store.get_document_count()}")
    print("Updating embeddings...")
    retriever = make_retriever(document_store)
    document_store.update_embeddings(retriever)
    print("Saving document store...")
    document_store.save("./db/wiki/wiki_data_faiss")
    return document_store, retriever

def get_wiki_data():
    if Path("./db/wiki/wiki_data_faiss").exists():
        document_store = FAISSDocumentStore.load(index_path="./db/wiki/wiki_data_faiss")
        retriever = make_retriever(document_store)
        return document_store, retriever
    else:
        return create_wiki_data()


if __name__ == "__main__":
    from haystack.pipelines import DocumentSearchPipeline
    from haystack.utils import print_documents
    document_store, retriever = get_wiki_data()
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Is the protein Papilin secreted?", params={"Retriever": {"top_k": 10}})
    print_documents(res, max_text_len=512)