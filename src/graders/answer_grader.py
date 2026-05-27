from graders.classes.grade_classes import GradeAnswer
from langchain_core.prompts import ChatPromptTemplate


def build_answer_grader(llm):

    llm = llm.with_structured_output(GradeAnswer)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing whether an answer addresses the question.
                    Give a binary score 'yes' if it addresses it, 'no' if it does not.""",
            ),
            ("human", "Question:\n{question}\n\nAnswer: {generation}"),
        ]
    )

    answer_grader = prompt | llm
    return answer_grader
