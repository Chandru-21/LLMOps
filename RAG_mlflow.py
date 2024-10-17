import os
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda,RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import mlflow
import pandas as pd
from datasets import Dataset 
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas import evaluate


prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
llm = ChatOpenAI(model="gpt-4o-mini")


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)


loader = PyPDFDirectoryLoader("./LLMOps/data")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

document_chunks = splitter.split_documents(docs)

embedding_generator = OpenAIEmbeddings()
faiss_database = FAISS.from_documents(document_chunks, embedding_generator)
faiss_database.save_local("./LLMOps/faissrag_index")

retriever = faiss_database.as_retriever()

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: (x["context"])))
    | PROMPT
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

rag_chain_with_source.invoke("what is this document all about?")

persist_dir = os.path.join("C:/Users/Chandru/.spyder-py3/LLMOps/LLMOps", "faissrag_index")

def load_retriever(persist_directory):
   
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings = OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,  
    )
    return vectorstore.as_retriever()

mlflow.set_experiment("RAG_MLFlow")

mlflow.langchain.autolog(log_models=True, log_input_examples=True)

with mlflow.start_run() as run:
    model_info = mlflow.langchain.log_model(
        rag_chain_with_source,
        artifact_path="rag_chain_with_source",
        loader_fn=load_retriever,
        persist_dir=persist_dir,
        input_example="hi"
    )
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/rag_chain_with_source"
    print(f"Unique identifier for the model location for loading: {model_uri}")


loaded_model = mlflow.langchain.load_model(model_uri)
print(loaded_model.invoke("what is the topic of this document?"))



###Load your evaluation data

eval_data=pd.read_csv("evaluation_dataset.csv")

results = []
contexts = []
query=[]
ground_truth=[]

for queries in eval_data['question']:
    result = loaded_model.invoke(queries)
    #print(result)
    query.append(queries)
    results.append(result['answer'])
    sources = result["context"]
    contents = []
    for i in range(len(sources)):
        contents.append(sources[i].page_content)
    contexts.append(contents)
        

for i in  eval_data['ground_truth']:
    ground_truth.append(i)

d = {
    "question": query,
    "answer": results,
    "contexts": contexts,
    "ground_truth": ground_truth
}

dataset = Dataset.from_dict(d)

score = evaluate(dataset,metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness])
score_df = score.to_pandas()
score_df
score_df[['faithfulness','answer_relevancy', 'context_precision', 'context_recall',
       'context_entity_recall', 'answer_similarity', 'answer_correctness']].mean(axis=0)

score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)

##Log Parameters in to the mlflow model

scores=score_df[['faithfulness','answer_relevancy', 'context_precision', 'context_recall',
       'context_entity_recall', 'answer_similarity', 'answer_correctness']].mean(axis=0).to_dict()

with mlflow.start_run(run_id=run_id):
    for metric_name, metric_value in scores.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print(f"RAGAS metrics logged successfully to run: {run_id}")

mlflow.end_run()

