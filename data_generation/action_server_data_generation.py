#!/usr/bin/env python3

import json
import logging
import os
import time
import glob
from typing import List, Optional

from openai import AzureOpenAI
from openai import BadRequestError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from dotenv import find_dotenv, load_dotenv

from action_server_constants import action_server_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

if find_dotenv():
    load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("azure_key"),
    api_version=os.getenv("AZURE_API_VERSION"),
)


class Annotation(BaseModel):
    token: str
    label: str


class DataPoint(BaseModel):
    sentence: str
    annotations: List[Annotation]
    action: str


class Dataset(BaseModel):
    dataset: List[DataPoint]


def generate_split(num_examples: int, split: str, **kwargs):
    logging.info(f"Using model: {kwargs['model']}")
    total_examples = 0
    all_data = []
    batch_size = max(1, num_examples // 5 if num_examples > 100 else num_examples // 2)

    num_batches = (num_examples + batch_size - 1) // batch_size
    logging.info(
        f"{num_batches} batches where each butch has size: {batch_size} for a total of {num_examples} examples of the {split} split"
    )

    for batch in range(num_batches):
        examples_to_generate = min(batch_size, num_examples - total_examples)

        messages = [
            {"role": "system", "content": action_server_prompt},
            {
                "role": "user",
                "content": f"Generate {examples_to_generate} diverse examples of robot commands with their BIO annotations.",
            },
        ]

        logging.info(
            f"Generating batch {batch + 1}/{num_batches} with {examples_to_generate} examples"
        )

        result = parse_with_retry(
            client=client,
            model=kwargs["model"],
            messages=messages,
            response_format=Dataset,
        )

        if result is None:
            logging.error(f"Failed to generate batch {batch + 1}")
            continue

        all_data.extend(result.dataset)
        total_examples += len(result.dataset)

        logging.info(
            f"Generated {len(result.dataset)} examples in batch {batch + 1}. Total: {total_examples}/{num_examples}"
        )

        interim_dataset = Dataset(dataset=all_data)
        with open(f"action_server_data_interim_batch_{batch + 1}.json", "w") as f:
            f.write(interim_dataset.model_dump_json(indent=2))

        if total_examples >= num_examples:
            break

        time.sleep(2)

    if not all_data:
        return None

    final_dataset = Dataset(dataset=all_data)

    with open(f"{split}.json", "w") as f:
        f.write(final_dataset.model_dump_json(indent=2))
        for interim_file in glob.glob("action_server_data_interim_batch_*.json"):
            os.remove(interim_file)

    logging.info(
        f"Successfully generated {total_examples} examples. Saved to {split}.json"
    )
    return final_dataset


def parse_with_retry(
    client: AzureOpenAI,
    model: str,
    messages: List[dict],
    response_format: BaseModel,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Optional[BaseModel]:
    """
    Attempts to parse the API call with retries on specific exceptions.

    Args:
        client: The OpenAI client instance.
        model: The model name to use for the API call.
        messages: The list of messages to send to the API.
        response_format: The Pydantic model to parse the response into.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Factor by which the delay increases after each retry.

    Returns:
        The parsed response or None if all retries fail.
    """
    attempt = 0
    delay = initial_delay

    while attempt < max_retries:
        try:
            logging.info(f"Attempt {attempt + 1}: Making API call to model {model}")
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=response_format, seed=42
            )
            generated_dataset = completion.choices[0].message.parsed
            dataset = response_format.model_validate(generated_dataset)
            logging.info(
                f"API call successful on attempt {attempt + 1} with model {model}"
            )
            return dataset

        except BadRequestError as e:
            logging.error(f"BadRequestError: {e}")
            break

        except RateLimitError as e:
            attempt += 1
            logging.warning(f"RateLimitError: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor

        except APITimeoutError as e:
            attempt += 1
            logging.warning(f"APITimeoutError: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor

        except Exception as e:
            attempt += 1
            logging.warning(f"Unexpected error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor

    logging.error(f"All {max_retries} attempts failed. Unable to parse the dataset.")
    return None


def generate_action_server_data(
    num_examples: int = 100,
    batch_size: int = 10,
    model: str = "gpt-4.1",
    split: str = "all",
) -> Optional[Dataset]:
    """
    Generate action server dataset using the OpenAI API.

    Args:
        num_examples: Total number of examples to generate.
        batch_size: Number of examples to generate in each batch.
        model: The model to use for generation.

    Returns:
        The generated dataset or None if generation fails.
    """

    if split == "all":
        splits = {"train": 0.8, "dev": 0.1, "test": 0.1}
    elif split == "train":
        splits = {"train": 1}
    elif split == "dev":
        splits = {"dev": 1}
    elif split == "test":
        splits = {"test": 1}
    else:
        raise ValueError(f"Invalid split: {split}")

    for split_name, split_value in splits.items():
        num_examples_to_generate = int(split_value * num_examples)
        logging.info(
            f"Generating {num_examples_to_generate} datapoints for split: {split_name}"
        )

        generate_split(
            num_examples=num_examples_to_generate, split=split_name, model=model
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate action server training data")
    parser.add_argument(
        "--num-examples", type=int, default=10, help="Number of examples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for generation"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1", help="Model to use for generation"
    )
    parser.add_argument(
        "--split", type=str, default="all", help="Split to save the data to"
    )

    args = parser.parse_args()

    generate_action_server_data(
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        model=args.model,
        split=args.split,
    )
