from langsmith import Client
import uuid
from .datasets import (
    conversation_1,
    conversation_2,
    criteria_1,
    criteria_2,
)
from ..agent import scope_research
from .evaluators.success_criteria import evaluate_success_criteria

ls_client = Client()
dataset_name = "deep-research-scoping"


if not ls_client.has_dataset(dataset_name=dataset_name):
    # Create dataset for evaluation
    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Dataset for training the deep research agent to scope "
            "research topics based on user input."
        ),
    )
    # Pre populate dataset with conversations and success criteria
    ls_client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"messages": conversation_1},
                "outputs": {"success_criteria": criteria_1},
            },
            {
                "inputs": {"messages": conversation_2},
                "outputs": {"success_criteria": criteria_2},
            },
        ],
    )
else:
    dataset = ls_client.read_dataset(dataset_name=dataset_name)



"""
Evaluation of the deep research agent's ability to scope research topics based on user input.

`dataset` is passed into the evaluation loop, where each example's `inputs` are fed into the `scope_research` function,
and the output is evaluated against the `success_criteria` using the `evaluate_success_criteria` function.

dataset["inputs"] -> scope_research.invoke(inputs) -> outputs -> evaluate_success_criteria(outputs, reference_outputs=dataset["outputs"])
"""
def target_func(inputs):
    return scope_research.invoke(inputs)

ls_client.evaluate(
    target_func,
    data=dataset, 
    evaluators=[evaluate_success_criteria],
    experiment_prefix="deep-research-scope-eval",
)
