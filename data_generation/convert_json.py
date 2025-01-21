import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_json_to_conll_like(json_data):
    """
    Converts the JSON formatted LLM output to a CONLL-like format

    Args:
        json_data: Obvious.

    Returns:
        A string representing the data in CONLL-like format.
    """
    conll_data = ""
    converted_count = 0
    skipped_count = 0

    if "dataset" in json_data:
        for item in json_data["dataset"]:
            if (
                "annotations" in item and "action" in item
            ):
                for annotation in item["annotations"]:
                    if "label" in annotation and "token" in annotation:
                        conll_data += f"{annotation['token']}:{annotation['label']} "
                conll_data += f"<=> {item['action']}\n"  # Using "action"
                logger.debug(f"Converted item with action: {item['action']}")
                converted_count += 1
            else:
                logger.warning(
                    f"Skipping item: Missing 'annotations' or 'action' key - {item}"
                )
                skipped_count += 1
    else:
        logger.error("JSON data does not contain a 'dataset' key.")

    return conll_data, converted_count, skipped_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON dataset to CONLL-like format for training."
    )
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output file.")
    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            data = json.load(f)
            logger.info(f"Loaded JSON data from '{args.input_file}'")
    except FileNotFoundError:
        logger.error(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in '{args.input_file}'.")
        exit(1)

    conll_like_output, converted_count, skipped_count = convert_json_to_conll_like(data)

    if conll_like_output.strip():
        with open(args.output_file, "w") as f:
            f.write(conll_like_output)
        logger.info(
            f"Successfully converted JSON to Slot Filling format. Output saved to '{args.output_file}'."
        )
    else:
        logger.warning(
            "Conversion resulted in empty output. Nothing written to output file."
        )

    logger.info("--- Report ---")
    logger.info(f"Total data points processed: {converted_count + skipped_count}")
    logger.info(f"Data points successfully converted: {converted_count}")
    logger.info(f"Data points skipped: {skipped_count}")
