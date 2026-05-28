from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from graders.answer_grader import answer_grader
from graders.hallucination_grader import hallucination_grader
from graders.relevance_grader import document_grader
from rag import rag_chain
from retriever import get_retriever
from query_rewriter import query_rewriter


max_retries = 2


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    relevance_score: str
    hallucination_score: str
    answer_score: str
    retries: int


def retrieve(state: GraphState) -> dict:
    question = state["question"]
    docs = get_retriever().invoke(question)
    return {"documents": docs, "question": question}


def generate(state: GraphState) -> dict:

    documents = "\n\n".join(doc.page_content for doc in state["documents"])
    question = state["question"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}


def grade_documents(state: GraphState) -> dict:

    documents = state["documents"]
    question = state["question"]

    filtered = []
    for doc in documents:
        score = document_grader.invoke(
            {"question": question, "documents": doc.page_content}
        )

        if score.binary_score == "yes":
            filtered.append(doc)

    relevance_score = "relevant" if filtered else "irrelevant"
    return {"documents": filtered, "relevance_score": relevance_score}


def grade_hallucination(state: GraphState) -> dict:

    generation = state["generation"]
    documents = state["documents"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    hallucination_score = "grounded" if score.binary_score == "yes" else "hallucinated"
    return {"hallucination_score": hallucination_score}


def grade_answer(state: GraphState) -> dict:

    question = state["question"]
    generation = state["generation"]

    score = answer_grader.invoke({"question": question, "generation": generation})

    answer_score = "useful" if score.binary_score == "yes" else "not useful"
    return {"answer_score": answer_score}


def rewrite_query(state: GraphState) -> dict:
    question = state["question"]
    new_question = query_rewriter.invoke({"question": question})
    return {"question": new_question, "retries": state.get("retries", 0) + 1}


def decide_after_relevance_grading(state: GraphState):

    if state["relevance_score"] == "relevant":
        return "generate"
    return "generate"


def decide_after_hallucination_grading(state: GraphState):
    if state["hallucination_score"] == "not grounded":
        return "generate"
    return "grade_answer"


def decide_after_answer_grading(state: GraphState):

    if state["answer_score"] == "useful":
        return "end"
    if state["relevance_score"] == "irrelevant":
        return "end"
    if state["retries"] > max_retries:
        return "end"

    return "rewrite_query"


graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate)
graph.add_node("grade_hallucination", grade_hallucination)
graph.add_node("grade_answer", grade_answer)
graph.add_node("rewrite_query", rewrite_query)


graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_edge("generate", "grade_hallucination")
graph.add_edge("rewrite_query", "retrieve")

graph.add_conditional_edges(
    "grade_documents",
    decide_after_relevance_grading,
    {"generate": "generate"},
)

graph.add_conditional_edges(
    "grade_hallucination",
    decide_after_hallucination_grading,
    {"generate": "generate", "grade_answer": "grade_answer"},
)

graph.add_conditional_edges(
    "grade_answer",
    decide_after_answer_grading,
    {"end": END, "rewrite_query": "rewrite_query"},
)

graph = graph.compile()
result = graph.invoke({"question": "Explain the architecture of transformers model"})
print(result)
print(result["generation"])
