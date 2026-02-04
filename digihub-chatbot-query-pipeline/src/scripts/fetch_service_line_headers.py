"""
Service Line Header Fetcher Script

This script fetches distinct headings from Azure Cosmos DB for each service line
and saves them as JSON files in /src/data/service_line_headers/

Query: SELECT DISTINCT c.heading FROM c WHERE c.serviceNameid = {id} AND c.validChunk = "yes"

Usage:
    python src/scripts/fetch_service_line_headers.py
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.cosmos_db_service import CosmosDBClientSingleton
from src.utils.config import KNOWLEDGE_BASE_CONTAINER
from src.enums.subscriptions import Subscription
from src.utils.logger import logger


class ServiceLineHeaderFetcher:
    """Fetches distinct headings from Cosmos DB for each service line"""

    def __init__(self):
        self.cosmos_client = CosmosDBClientSingleton()
        self.database = self.cosmos_client.get_database()
        self.container = self.database.get_container_client(KNOWLEDGE_BASE_CONTAINER)
        self.output_dir = Path("src/data/service_line_headers")

    def fetch_headers_for_service_line(self, service_id: int, service_name: str) -> List[Dict[str, str]]:
        """
        Fetches distinct headings from Cosmos DB for a specific service line.

        Args:
            service_id: Service line ID (e.g., 240 for World Tracer)
            service_name: Service line display name (for logging)

        Returns:
            List of dictionaries with "heading" field
            Example: [{"heading": "Header 1"}, {"heading": "Header 2"}]
        """
        query = f"""
        SELECT DISTINCT c.heading
        FROM c
        WHERE c.serviceNameid = {service_id}
        AND c.validChunk = "yes"
        """

        try:
            logger.info(f"Fetching headers for: {service_name} (ID: {service_id})")

            # Execute query
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))

            # Filter out null/empty headings and format results
            headers = []
            for item in items:
                heading = item.get("heading", "").strip()
                if heading:  # Only include non-empty headings
                    headers.append({"heading": heading})

            logger.info(f"Found {len(headers)} distinct headings for {service_name}")
            return headers

        except Exception as e:
            logger.error(f"Error fetching headers for {service_name} (ID: {service_id}): {e}")
            return []

    def save_headers_to_file(self, service_line_enum_name: str, headers: List[Dict[str, str]]) -> bool:
        """
        Saves headers to a JSON file.

        Args:
            service_line_enum_name: Enum name (e.g., "world_tracer")
            headers: List of header dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not headers:
            logger.warning(f"No headers to save for {service_line_enum_name}")
            return False

        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create file path
            file_path = self.output_dir / f"{service_line_enum_name}.json"

            # Write JSON with pretty formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(headers, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved {len(headers)} headers to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving headers to file for {service_line_enum_name}: {e}")
            return False

    def process_all_service_lines(self) -> Dict[str, int]:
        """
        Main processing loop that fetches headers for all service lines.

        Returns:
            Dictionary mapping service line names to header counts
        """
        summary = {}

        logger.info("=" * 60)
        logger.info("Starting header fetch for all service lines")
        logger.info("=" * 60)

        for item in Subscription:
            service_id = item.value["id"]
            service_name = item.value["name"]
            enum_name = item.name  # e.g., "world_tracer"

            # Skip generic/general_info duplicates (both have id=0)
            if enum_name == "generic":
                logger.info(f"Skipping duplicate generic entry (using general_info instead)")
                continue

            logger.info(f"\nProcessing: {service_name} (Enum: {enum_name}, ID: {service_id})")

            # Fetch headers from Cosmos DB
            headers = self.fetch_headers_for_service_line(service_id, service_name)

            # Save to file
            if headers:
                success = self.save_headers_to_file(enum_name, headers)
                if success:
                    summary[service_name] = len(headers)
            else:
                logger.warning(f"No headers found for {service_name}, skipping file creation")
                summary[service_name] = 0

        return summary


def main():
    """Main entry point for the script"""
    logger.info("=" * 60)
    logger.info("Service Line Header Fetcher Script")
    logger.info("=" * 60)

    try:
        fetcher = ServiceLineHeaderFetcher()

        logger.info(f"Database: {fetcher.database.id}")
        logger.info(f"Container: {KNOWLEDGE_BASE_CONTAINER}")
        logger.info(f"Output directory: {fetcher.output_dir}")

        # Process all service lines
        summary = fetcher.process_all_service_lines()

        # Print summary
        print("\n" + "=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)

        total_headers = 0
        for service_name, count in sorted(summary.items()):
            status = "✓" if count > 0 else "✗"
            print(f"{status} {service_name:45} {count:4} headers")
            total_headers += count

        print("=" * 60)
        print(f"Total service lines processed: {len(summary)}")
        print(f"Total headers fetched: {total_headers}")
        print(f"Files saved to: {fetcher.output_dir}")
        print("=" * 60)

        # Next steps
        print("\nNext Steps:")
        print("1. Review the generated JSON files in src/data/service_line_headers/")
        print("2. Run keyword extraction: python src/scripts/extract_service_line_keywords.py")

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
