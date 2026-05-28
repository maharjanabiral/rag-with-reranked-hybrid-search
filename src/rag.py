from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from shared_llm import llm


def build_rag_chain():

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks.
                Use the following retrieved context to answer the question.
                If you don't know the answer, say you don't know.
                Keep the answer concise and grounded in the context.
                If you do not find any context related to the question  Briefly explain what the documents do cover and why they don't fully answer the question.
                Also do not use your own knowledge to answer the question""",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain


rag_chain = build_rag_chain()
