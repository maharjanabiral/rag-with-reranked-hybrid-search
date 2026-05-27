from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


def build_query_rewriter():
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                """You are a question rewriter that improves queries for vector store retrieval.
                Look at the input question and rewrite it to be more precise and retrieval-friendly.""",
            ),
            ("human", "Original question: {question}\nRewrite it:"),
        ]
    )

    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter
