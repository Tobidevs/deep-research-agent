import os
from langsmith import Client
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from backend.agent.evaluation.prompts import BRIEF_CRITERIA_PROMPT
from backend.agent.evaluation.datasets import ReferenceDataset
from pydantic import BaseModel, Field

client = Client()
model = init_chat_model(
    model="openai:gpt-4.1",
    temperature=0,
)
dataset_name = "deep-research-scoping"
reference_dataset = ReferenceDataset()


if not client.has_dataset(dataset_name=dataset_name):
    # Create dataset for evaluation
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Dataset for training the deep research agent to scope "
            "research topics based on user input."
        ),
    )
else:
    dataset = client.get_dataset(dataset_name=dataset_name)

# Materialize examples from the reference dataset so they stay
# in sync with whatever is configured in `datasets.py`.
examples = []
for name in reference_dataset.list_datasets():
    data = reference_dataset.get_dataset(name)
    if not data:
        continue
    examples.append(
        {
            "inputs": {"messages": data["conversation"]},
            "outputs": {"criteria": data["criteria"]},
        }
    )

if examples:
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
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


def evaluate_success_criteria(outputs: dict, reference_outputs: dict):
    """
    Evaluates each criterion individually to provide focused assessment
    and detailed reasoning for each evaluation decision.

    @param outputs: Dict containing the research brief to evaluate.
    @param reference_outputs: Dict containing the specific criteria to evaluate against.

    @return: List of Criteria objects with detailed reasoning for each criterion.
    """
    research_brief = outputs["research_brief"]
    success_criteria = reference_outputs["criteria"]

    structured_output_model = model.with_structured_output(Criteria)

    # Run Evals
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

    # Verify criteria_text is populated correctly
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

    return {
        "key": "success_criteria_evaluation",
        "score": captured_count / total_count if total_count > 0 else 0.0, 
        "individual_evaluations": [
            { 
                "criteria": eval_result.criteria_text,
                "is_captured": eval_result.is_captured,
                "reasoning": eval_result.reasoning,
            }
            for eval_result in individual_evaluations
        ],
    }
