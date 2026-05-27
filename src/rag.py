from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_rag_chain(llm):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks.
                Use the following retrieved context to answer the question.
                If you don't know the answer, say you don't know.
                Keep the answer concise and grounded in the context.""",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain
