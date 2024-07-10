import os

from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"), base_url="https://api.llama-api.com")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


class ModelInfoRequest(BaseModel):
    content: str
    model_names: List[str]
    model_types: List[str]


def model_query(content: str, model_names: str, model_types: str) -> dict:
    """Function to query the model and process the response.
    Ask the question to the model and extract the answer from the model.
    """

    try:
        return (
            client
            .chat
            .completions
            .create(
                model="llama-13b-chat",
                messages=[
                    {"role": "system", "content": "Tu es un expert en machine learning."},
                    {
                        "role": "user",
                        "content": f"""
                            Voici le contenu d'un fichier Python: {content}
                            Voici une liste de nom de modèle parmi lesquels choisir: {model_names}
                            Voici une liste de types de modèles parmi lesquels choisir: {model_types}
                            Quel modèle est utilisé dans ce fichier et quels sont les metrics les plus adaptés pour ce code.
                            Je veux une sortie uniquement dans ce format exact rien d'autre: 'model_name: XXX; model_type: XXX;'.
                            Ne retire pas les espaces entre les mots des valeurs XXX pour model_name et model_type. 
                        """,
                    },
                ],
            )
            .choices[0]
            .message
            .content
            .strip()
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/model_info")
def get_model_info(request: ModelInfoRequest) -> dict:
    if "error" in (
        resultat := {
            pair.split(":")[0].strip(): pair.split(":")[1].strip()
            for pair in model_query(
                request.content,
                ", ".join(request.model_names),
                ", ".join(request.model_types),
            ).replace("\"", "").split(";")
            if pair.strip()
        }
    ):
        raise HTTPException(status_code=400, detail=resultat["error"])

    return resultat
