import logging
from pathlib import Path
import sys
from bioasq_data import get_bioasq_data

path_root = Path(__file__).parents[0]
sys.path.insert(0,str(path_root)+"/haystack")

from haystack.utils import print_documents, print_answers
from haystack.pipelines import DocumentSearchPipeline, GenerativeQAPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


# class MyConverter: #converter for t5 model
#     def __call__(self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None) -> BatchEncoding:
#         conditioned_doc = "<s> " + " </s> ".join([d.content for d in documents])
#         query_and_docs = "question: {} context: {}".format(query, conditioned_doc)
#         return tokenizer([query_and_docs], truncation=True, padding=True, return_tensors="pt")

# generator = Seq2SeqGenerator(model_name_or_path="wyu1/FiD-TQA", min_length=80,input_converter=MyConverter()) #T5 too big

def test_retriever(retriever):
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Is Hirschsprung disease a mendelian or a multifactorial disorder?", params={"Retriever": {"top_k": 3}})
    print_documents(res)

document_store, retriever, question,answer = get_bioasq_data()
test_retriever(retriever)

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa", min_length=80) # better than Fit-TQA

pipe = GenerativeQAPipeline(generator, retriever)

res = pipe.run(query="Is Hirschsprung disease a mendelian or a multifactorial disorder?", params={"Retriever": {"top_k": 3}})

print_answers(res)


