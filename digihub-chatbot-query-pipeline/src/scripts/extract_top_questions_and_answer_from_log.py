#!/usr/bin/env python3
"""
Script to extract log lines containing 'DigiHubChatbot:QueryPipeline - INFO #$#-top_docs'
from a log file and write them to a new file.
"""

def extract_top_docs_logs(input_file: str, output_file: str):
    """
    Extract lines containing the top_docs prefix from input file.

    Args:
        input_file: Path to the source log file
        output_file: Path to the output file
    """
    # search_prefix = "DigiHubChatbot:QueryPipeline - INFO #$#-"
    search_prefix = "DigiHubChatbot:QueryPipeline - INFO - #$#-"
    matched_lines = []

    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if search_prefix in line:
                    matched_lines.append(line)

        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(matched_lines)

        print(f"✓ Extraction complete!")
        print(f"  Found {len(matched_lines)} matching lines")
        print(f"  Written to: {output_file}")

    except FileNotFoundError:
        print(f"✗ Error: Input file not found: {input_file}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Configure input and output file paths
    input_log_file = r"D:\Repos\digihub-chatbot-query-pipeline\QueryPipeline.log"
    output_log_file = r"D:\Repos\digihub-chatbot-query-pipeline\QueryPipeline_extracted.log"

    extract_top_docs_logs(input_log_file, output_log_file)
