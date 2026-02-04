"""
Service Line Keyword Extraction Script

This script reads document headers from JSON files in /src/data/service_line_headers/
and uses an LLM to extract minimal core keywords for service line classification.

Usage:
    python src/scripts/extract_service_line_keywords.py
"""

import os
import json
import sys
from pathlib import Path
import traceback
from typing import Dict, List

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.azure_openai_service import AzureOpenAIService
from src.utils.config import OPENAI_DEPLOYMENT_NAME
from src.enums.subscriptions import Subscription
from src.utils.logger import logger


class ServiceLineKeywordExtractor:
    """Extracts core keywords from document headers for service line classification"""

    def __init__(self):
        self.client = AzureOpenAIService().get_client()
        self.model = OPENAI_DEPLOYMENT_NAME
        self.input_dir = Path("src/data/service_line_headers")
        self.output_file = Path("src/data/service_line_keywords.json")
        self.service_line_mapping = self._build_service_line_mapping()

    def _build_service_line_mapping(self) -> Dict[str, str]:
        """
        Maps filename patterns to official service line names.

        Returns:
            Dict mapping filename (without .json) to service line name
            Example: {"airport_solutions": "Airport Solutions"}
        """
        mapping = {}
        for item in Subscription:
            name = item.name  # e.g., "airport_solutions"
            value = item.value["name"]  # e.g., "Airport Solutions"
            mapping[name] = value

        return mapping

    def _get_keyword_extraction_prompt(self, service_name: str, headers: List[str]) -> str:
        """
        Generates the LLM prompt for keyword extraction.

        Args:
            service_name: Official service line name (e.g., "Airport Solutions")
            headers: List of header strings

        Returns:
            Formatted prompt string
        """
        headers_formatted = "\n".join([f"- {h}" for h in headers])

        return f"""TASK: Extract minimal core keywords from document headers for service line classification.

SERVICE LINE: {service_name}

DOCUMENT HEADERS:
{headers_formatted}

INSTRUCTIONS:
1. Identify the MOST DISTINCTIVE keywords that uniquely identify this service line
2. Extract 5-15 core keywords (prefer fewer, more distinctive terms)
3. Filter out GENERIC terms like:
   - "User Guide", "Manual", "Documentation", "Guide"
   - "Author", "Version", "Table of Contents"
   - "Introduction", "Overview", "Summary"
   - "Chapter", "Section", "Appendix"
   - "Date", "Page", "Copyright", "Revision"

4. Include SPECIFIC terms like:
   - Product names (e.g., "WorldTracer", "Bag Manager", "DIGIHUB AIR DASHBOARD")
   - Technical terms (e.g., "Type B messages", "LNI code", "tracing")
   - Domain-specific concepts (e.g., "baggage tracing", "billing reconciliation", "operational monitoring")
   - System names and features

5. Normalize keywords:
   - Use lowercase for consistency
   - Keep multi-word phrases if they're distinctive (e.g., "air dashboard", "lost baggage")
   - Remove punctuation and special characters
   - Keep acronyms if they're distinctive (e.g., "api", "sdk")

6. Prioritize keywords that appear in MULTIPLE headers (high frequency = high relevance)

7. Output ONLY the keywords as a JSON array, no explanations

EXAMPLES:
Input Headers:
- DIGIHUB AIR DASHBOARD
- User Guide
- Operational Monitoring
- Author:

Output: ["digihub air dashboard", "operational", "monitoring"]

Input Headers:
- WorldTracer Configuration
- Baggage Tracing Setup
- Lost Baggage Handling
- User Manual

Output: ["worldtracer", "baggage", "tracing", "lost", "handling"]

Respond ONLY with a JSON array dont include ```json ''':
["keyword1", "keyword2", ...]"""

    def extract_keywords_for_service_line(self, service_name: str, headers: List[str]) -> List[str]:
        """
        Uses LLM to extract minimal core keywords from headers.

        Args:
            service_name: Official service line name
            headers: List of header strings

        Returns:
            List of extracted keywords
        """
        prompt = self._get_keyword_extraction_prompt(service_name, headers)

        try:
            logger.info(f"Extracting keywords for: {service_name}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a keyword extraction assistant. Always respond with valid JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()
            keywords = json.loads(content)

            if not isinstance(keywords, list):
                logger.error(f"Invalid response format for {service_name}: not a list")
                return []

            logger.info(f"Extracted {len(keywords)} keywords for {service_name}")
            return keywords

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {service_name}: {e}")
            logger.error(f"Raw response: {content}")
            traceback.print_exc()

            return []
        except Exception as e:
            logger.error(f"Error extracting keywords for {service_name}: {e}")
            return []

    def read_headers_from_file(self, file_path: Path) -> List[str]:
        """
        Reads headers from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of header strings
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error(f"Invalid JSON structure in {file_path}: expected list")
                return []

            headers = []
            for item in data:
                if isinstance(item, dict) and "heading" in item:
                    headers.append(item["heading"])
                elif isinstance(item, str):
                    headers.append(item)

            logger.info(f"Read {len(headers)} headers from {file_path.name}")
            return headers

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

    def process_all_service_lines(self) -> Dict[str, List[str]]:
        """
        Main processing loop that reads all JSON files and extracts keywords.

        Returns:
            Dictionary mapping service line names to keyword lists
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return {}

        keywords_dict = {}
        json_files = list(self.input_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return {}

        logger.info(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            # Extract filename without extension (e.g., "airport_solutions")
            filename_stem = json_file.stem

            # Skip README or other non-service-line files
            if filename_stem.lower() in ["readme", "example", "template"]:
                continue

            # Map filename to official service line name
            service_line_name = self.service_line_mapping.get(filename_stem)

            if not service_line_name:
                logger.warning(f"No service line mapping found for: {filename_stem}")
                continue

            # Read headers from file
            headers = self.read_headers_from_file(json_file)

            if not headers:
                logger.warning(f"No headers found in {json_file.name}, skipping")
                continue

            # Extract keywords
            keywords = self.extract_keywords_for_service_line(service_line_name, headers)

            if keywords:
                keywords_dict[service_line_name] = keywords

        return keywords_dict

    def save_keywords(self, keywords_dict: Dict[str, List[str]]):
        """
        Save consolidated keywords JSON to output file.

        Args:
            keywords_dict: Dictionary mapping service line names to keyword lists
        """
        try:
            # Ensure output directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON with pretty formatting
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(keywords_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved keywords to: {self.output_file}")
            logger.info(f"Total service lines processed: {len(keywords_dict)}")

        except Exception as e:
            logger.error(f"Error saving keywords to {self.output_file}: {e}")


def main():
    """Main entry point for the script"""
    logger.info("=" * 60)
    logger.info("Service Line Keyword Extraction Script")
    logger.info("=" * 60)

    extractor = ServiceLineKeywordExtractor()

    logger.info(f"Input directory: {extractor.input_dir}")
    logger.info(f"Output file: {extractor.output_file}")

    # Process all service lines
    keywords_dict = extractor.process_all_service_lines()

    if not keywords_dict:
        logger.error("No keywords extracted. Check input files and logs.")
        return 1

    # Save results
    extractor.save_keywords(keywords_dict)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for service_line, keywords in sorted(keywords_dict.items()):
        print(f"\n{service_line} ({len(keywords)} keywords):")
        print(f"  {', '.join(keywords)}")

    print("\n" + "=" * 60)
    print(f"Keywords saved to: {extractor.output_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
