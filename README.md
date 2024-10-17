# Retrieval Augmented Generation (RAG)

## RAG LLMOps using OpenAI,MLflow,FastAPI and RAGAS(Evaluation)

Step I : Install all the requirements and execute the RAG_mlflow.py file

Now your LLM model run is registered in MLFlow with its evaluations metrices. 

![image](https://github.com/user-attachments/assets/d2114340-6f4b-44fb-a772-38e1d283a157)

RAGAS Evaluation metrics are logged in MLFlow, refer code for further details.

Step II : Execute the app.py (python app.py) in cmd navigate to the FastAPI UI in browser,

![image](https://github.com/user-attachments/assets/535549fc-3595-448c-b30c-2f4cd7a42638)

Click on try it out and enter your question

Step III : Model Monitoring/tracing

Navigate back to the MLflow UI and click on the 'Traces' tab. There, you'll find the question you asked in FastAPI is logged along with its response.

![image](https://github.com/user-attachments/assets/8f4037fb-7533-4ec1-bf20-70319418f252)

Click on the Request id to see more details about the run,

![image](https://github.com/user-attachments/assets/84fe7bdf-db3d-4872-ba70-233088442ef5)
