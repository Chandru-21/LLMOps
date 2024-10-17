from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlflow


app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("RAG_MLFlow")
mlflow.langchain.autolog()
model_uri='runs:/9976886f819f4400816192bbf73fcb25/rag_chain_with_source'

loaded_model = mlflow.langchain.load_model(model_uri)

class Query(BaseModel):
    question: str

@app.post("/predict")
async def predict(query: Query):
    try:
        result = loaded_model.invoke(query.question)
      
        answer = result.get('answer', 'No answer found')
        
        return {"question": query.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8005)