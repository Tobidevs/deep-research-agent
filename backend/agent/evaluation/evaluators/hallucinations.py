from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from ..prompts import BRIEF_HALLUCINATION_PROMPT
from langchain_core.messages import HumanMessage

model = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
)


class HallucinationEvaluation(BaseModel):
    """
    Structured output model for evaluating hallucinations in the agent's responses.
    """

    is_hallucination: bool = Field(
        description="Indicates whether the response contains hallucinated information."
    )
    reasoning: str = Field(
        description="The reasoning behind why the response is or isn't considered to contain hallucinations in one short sentence."
    )


def evaluate_hallucinations(outputs: dict, reference_outputs: dict):
    """
    Evaluates the agent's response for hallucinations by comparing it against reference outputs.

    @param outputs: Dict containing the agent's response to evaluate.
    @param reference_outputs: Dict containing the reference information to compare against.

    @return: Dict containing the overall hallucination score and reasoning.
    """

    response = outputs["research_brief"]
    success_criteria = reference_outputs["success_criteria"]

    structured_output_model = model.with_structured_output(HallucinationEvaluation)

    responses = structured_output_model.batch(
        [
            [
                HumanMessage(
                    content=BRIEF_HALLUCINATION_PROMPT.format(
                        research_brief=response, criterion=criterion
                    )
                )
            ]
            for criterion in success_criteria
        ]
    )

    individual_evaluations = [
        HallucinationEvaluation(
            is_hallucination=resp.is_hallucination,
            reasoning=resp.reasoning
        )
        for resp in responses
    ]

    # Calculate overall score as percentage of criteria captured
    total_count = len(success_criteria)
    passed_count = sum(
        1 for eval in individual_evaluations if not eval.is_hallucination
    )
    # Local testing
    print(f"Passed {passed_count} out of {total_count} criteria. \n\n")

    return {
        "key": "hallucination_evaluation",
        "score": passed_count / total_count if total_count > 0 else 0.0,
    }
