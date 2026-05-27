from graders.classes.grade_classes import GradeDocuments
from langchain_core.prompts import ChatPromptTemplate


def build_document_grader(llm):

    llm = llm.with_structured_output(GradeDocuments)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing relevance of a retrieved document to a user question.
                    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                    Give a binary score 'yes' or 'no'.""",
            ),
            ("human", "Retrieved documents:\n{documents}\n\nUser question: {question}"),
        ]
    )

    document_grader = prompt | llm
    return document_grader
