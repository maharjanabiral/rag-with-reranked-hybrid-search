from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from retriever import build_retriever


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    relevance_store: str
    hallucination_score: str
    answer_score: str
    retries: int


def retrieve(state: GraphState) -> StateGraph:

    retriever = build_retriever()

    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}


graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_edge(START, "retrieve")
app = graph.compile()
app.invoke({"question": "What is Hybrid search"})
