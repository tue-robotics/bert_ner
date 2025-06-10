#!/usr/bin/env python3

import logging
import os
import time
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import AzureOpenAI
from openai import BadRequestError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from dotenv import find_dotenv, load_dotenv

from action_server_constants import action_server_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

if find_dotenv():
    load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("azure_key"),
    api_version=os.getenv("AZURE_API_VERSION")
)

thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, 'client'):
        thread_local.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("azure_key"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
    return thread_local.client

class Annotation(BaseModel):
    token: str
    label: str

class DataPoint(BaseModel):
    sentence: str
    annotations: List[Annotation]
    action: str

class Dataset(BaseModel):
    dataset: List[DataPoint]

def parse_with_retry(
    client: AzureOpenAI,
    model: str,
    messages: List[dict],
    response_format: BaseModel,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
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
                model=model, 
                messages=messages,
                response_format=response_format,
                seed=42
            )
            generated_dataset = completion.choices[0].message.parsed
            dataset = response_format.model_validate(generated_dataset)
            logging.info(f"API call successful on attempt {attempt+1} with model {model}")
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

def generate_single_batch(batch_id: int, examples_to_generate: int, model: str) -> Optional[List[DataPoint]]:
    """
    Generate a single batch of data.
    
    Args:
        batch_id: The ID of the batch being generated.
        examples_to_generate: Number of examples to generate in this batch.
        model: The model to use for generation.
        
    Returns:
        List of generated DataPoints or None if generation fails.
    """
    messages = [
        {"role": "system", "content": action_server_prompt},
        {"role": "user", "content": f"Generate {examples_to_generate} diverse examples of robot commands with their BIO annotations."}
    ]
    
    logging.info(f"Worker generating batch {batch_id} with {examples_to_generate} examples")
    
    thread_client = get_client()
    
    result = parse_with_retry(
        client=thread_client,
        model=model,
        messages=messages,
        response_format=Dataset
    )
    
    if result is None:
        logging.error(f"Failed to generate batch {batch_id}")
        return None
    
    logging.info(f"Worker completed batch {batch_id} with {len(result.dataset)} examples")
    return result.dataset

def generate_action_server_data(num_examples: int = 100, batch_size: int = 10, model: str = "gpt-4.1", max_workers: int = 5) -> Optional[Dataset]:
    """
    Generate action server dataset using the OpenAI API with parallel processing.
    
    Args:
        num_examples: Total number of examples to generate.
        batch_size: Number of examples to generate in each batch.
        model: The model to use for generation.
        max_workers: Maximum number of parallel workers.
        
    Returns:
        The generated dataset or None if generation fails.
    """
    all_data = []
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    logging.info(f"Starting parallel generation with {max_workers} workers for {num_batches} batches")
    
    # Create batch tasks
    batch_tasks = []
    for batch in range(num_batches):
        examples_to_generate = min(batch_size, num_examples - batch * batch_size)
        batch_tasks.append((batch + 1, examples_to_generate, model))
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch tasks
        future_to_batch = {
            executor.submit(generate_single_batch, batch_id, examples, model): batch_id
            for batch_id, examples, model in batch_tasks
        }
        
        # Collect results as they complete
        completed_batches = []
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                if result is not None:
                    completed_batches.append((batch_id, result))
                    logging.info(f"Batch {batch_id} completed successfully with {len(result)} examples")
                else:
                    logging.error(f"Batch {batch_id} failed to generate data")
            except Exception as e:
                logging.error(f"Batch {batch_id} generated an exception: {e}")
    
    # Sort batches by ID and combine results
    completed_batches.sort(key=lambda x: x[0])
    total_examples = 0
    
    for batch_id, batch_data in completed_batches:
        all_data.extend(batch_data)
        total_examples += len(batch_data)
    
    if not all_data:
        logging.error("No data was generated successfully")
        return None
        
    final_dataset = Dataset(dataset=all_data)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"action_server_data_{timestamp}.json"
    
    with open(filename, "w") as f:
        f.write(final_dataset.model_dump_json(indent=2))
        
    logging.info(f"Successfully generated {total_examples} examples across {len(completed_batches)} batches. Saved to {filename}")
    return final_dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate action server training data")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use for generation")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    generate_action_server_data(
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        model=args.model,
        max_workers=args.max_workers
    ) 