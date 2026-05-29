from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.graph import graph

    app_state["graph"] = graph
    yield
    app_state.clear()


app = FastAPI(lifespan=lifespan)


class DocResponse(BaseModel):
    source: str
    content: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    generation: str
    relevance_score: str
    hallucination_score: str
    answer_score: str
    retries: int
    docs: List[DocResponse] = []


@app.post("/chat")
async def query(request: QueryRequest):
    try:
        result = app_state["graph"].invoke(
            {
                "question": request.question,
                "generation": "",
                "documents": [],
                "relevance_score": "",
                "hallucination_score": "",
                "answer_score": "",
                "retries": 0,
            }
        )

        response = QueryResponse(
            generation=result["generation"],
            documents=result["documents"],
            relevance_score=result["relevance_score"],
            hallucination_score=result["hallucination_score"],
            answer_score=result["answer_score"],
            retries=result["retries"],
            docs=[
                DocResponse(
                    source=doc.metadata.get("source", "unknown"),
                    content=doc.page_content,
                )
                for doc in result["documents"]
            ],
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def check_health():
    return {"status": "Server is healthy"}
