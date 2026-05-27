from graders.classes.grade_classes import GradeHallucinations
from langchain_core.prompts import ChatPromptTemplate


def build_hallucination_grader(llm):

    llm = llm.with_structured_output(GradeHallucinations)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing whether an LLM generation is grounded in retrieved facts.
                    Give a binary score 'yes' if grounded, 'no' if it contains hallucinations.""",
            ),
            ("human", "Facts:\n{documents}\n\nGeneration: {generation}"),
        ]
    )

    hallucination_grader = prompt | llm
    return hallucination_grader
