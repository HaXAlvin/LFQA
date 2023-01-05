from transformers import AutoTokenizer, AutoModel, logging, DPRReader, DPRReaderTokenizer, T5Model, T5Tokenizer, LongT5Model, LongT5ForConditionalGeneration, LongT5Config
import torch
logging.set_verbosity_error()
import faiss  # https://faiss.ai/index.html


from transformers import DPRContextEncoder
from datasets import load_dataset
import json



if False:
    with open("/Users/alvin/Projects/Courses/NLP/LFQA/ELI5/data_creation/processed_data/public_examples/examples_qda_long.json", "r") as f:
        d = json.load(f)

    # careful about unicode
    ids, questions, documents, answers = zip(*[i.values() for i in d])

class DPR():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=".cache/")
        # try facebook/dpr-question_encoder-multiset-base
        self.model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=".cache/")

    def search_documents(self, questions, documents):
        # sentences = ["'What Is Love' is a song recorded by the artist Haddaway", "Breathing in air when close to an infected person who is exhaling small droplets and particles that contain the virus. Having these droplets and particles land on the eyes, nose, or mouth. Touching the eyes, nose, and mouth with hands that have the virus on them."]
        # questions=["What is love ?","Why covid-19 is dangerous ?"]
        encoded_questions = self.tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
        encoded_documents = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        print(encoded_questions.input_ids.shape)
        print(encoded_documents.input_ids.shape)
        # Compute token embeddings
        with torch.no_grad():
            question_embedding = self.model(**encoded_questions)
            document_embedding = self.model(**encoded_documents)


        index = faiss.IndexFlatL2(768) # docs are 768-dimensional
        index.add(document_embedding.pooler_output.numpy()) # read all docs
        print(index.ntotal)
        D,I = index.search(question_embedding.pooler_output.numpy(), 2) # find top-k
        print(D) # distances
        print(I) # indexes
        
if False:
    DPR().search_documents(questions[:2], documents[:2])

if False:
    tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base", cache_dir=".cache/")
    model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", cache_dir=".cache/") # Better than bm-25 and tf-idf https://towardsdatascience.com/how-to-create-an-answer-from-a-question-with-dpr-d76e29cc5d60
    encoded_inputs = tokenizer(
        questions=["What is love ?","Why covid-19 is dangerous ?"],
        titles=["Haddaway","9 Things Everyone Should Know About the Coronavirus Outbreak"],
        texts=["'What Is Love' is a song recorded by the artist Haddaway", "Breathing in air when close to an infected person who is exhaling small droplets and particles that contain the virus. Having these droplets and particles land on the eyes, nose, or mouth. Touching the eyes, nose, and mouth with hands that have the virus on them."],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    print(encoded_inputs)
    outputs = model(**encoded_inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    relevance_logits = outputs.relevance_logits
    print(start_logits)
    print(end_logits)
    print(relevance_logits)

if True:
    tokenizer = AutoTokenizer.from_pretrained("wyu1/FiD-TQA", cache_dir=".cache/")  # https://huggingface.co/wyu1/FiD-TQA #TriviaQA
    # config = LongT5Config.from_pretrained("wyu1/FiD-TQA", cache_dir=".cache/")
    # print(config)
    model = LongT5ForConditionalGeneration.from_pretrained("wyu1/FiD-TQA", cache_dir=".cache/")  # https://huggingface.co/wyu1/FiD-TQA
    # print(model)
    print(model.config.task_specific_params.keys())  # prefix

    # exit()
    inputs = tokenizer(["question: How old is Alvin? context: Alvin is handsome.</s>Alvin is 18 yeats old."], padding=True, truncation=True, return_tensors="pt")
    print(inputs.tokens(0))
    print(inputs)

    labels = tokenizer(["18 years"], padding=True, truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100 so it's ignored by the loss

    # decoder_input_ids = tokenizer("I am 18 years old.", padding=True, truncation=True, return_tensors="pt")
    # print(decoder_input_ids)
    # decoder_input_ids = model._shift_right(decoder_input_ids.input_ids)

    print(inputs)
    # print(decoder_input_ids)


    # outputs = model(**inputs,labels=labels, output_hidden_states=True)
    # print(outputs.logits.shape)
    # print(outputs.keys())
    # print(outputs.decoder_hidden_states[-1].shape)
    # print(outputs.encoder_last_hidden_state.shape)

    outputs = model.generate(**inputs, output_hidden_states=True)
    print(outputs)
    print(tokenizer.decode(outputs[0]))
