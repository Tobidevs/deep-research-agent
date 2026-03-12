from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from ..prompts import BRIEF_CRITERIA_PROMPT


model = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
)

class Criteria(BaseModel):
    """
    Individual criteria for evaluating the model's performance in scoping research topics based on user input.
    """

    criteria_text: str = Field(
        description="The specific criterion to evaluate the model's performance on."
    )
    reasoning: str = Field(
        description="The reasoning behind why this criteria is or isn't captured in the research brief."
    )
    is_captured: bool = Field(
        description="Binary assessment of whether the criterion is captured in the research brief."
    )

# Evaluator Function
def evaluate_success_criteria(outputs: dict, reference_outputs: dict):
    """
    Evaluates each criterion individually to provide focused assessment
    and detailed reasoning for each evaluation decision.

    @param outputs: Dict containing the research brief to evaluate.
    @param reference_outputs: Dict containing the success criteria to evaluate against from dataset["outputs"].

    @return: List of Criteria objects with detailed reasoning for each criterion.
    """
    research_brief = outputs["research_brief"]
    success_criteria = reference_outputs["success_criteria"]

    structured_output_model = model.with_structured_output(Criteria)

    # Run Evals using LLM as Judge for each criterion and collect detailed reasoning
    responses = structured_output_model.batch(
        [
            [
                HumanMessage(
                    content=BRIEF_CRITERIA_PROMPT.format(
                        criterion=criterion, research_brief=research_brief
                    )
                )
            ]
            for criterion in success_criteria
        ]
    )

    # Aggregate results into structured format with detailed reasoning for each criterion
    individual_evaluations = [
        Criteria(
            reasoning=response.reasoning,
            criteria_text=criterion,
            is_captured=response.is_captured,
        )
        for criterion, response in zip(success_criteria, responses)
    ]

    # Calculate overall score as percentage of criteria captured
    total_count = len(success_criteria)
    captured_count = sum(1 for eval in individual_evaluations if eval.is_captured)
    print(f"Captured {captured_count} out of {total_count} criteria. \n\n")

    return {
        "key": "success_criteria_evaluation",
        "score": captured_count / total_count if total_count > 0 else 0.0,
    }