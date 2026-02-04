import re
from typing import List


class MarkdownTableExtractor:
    """
    Utility for extracting markdown tables from text.

    Supports GitHub-Flavored Markdown (GFM) table syntax with the format:
    | Header 1 | Header 2 | Header 3 |
    |----------|----------|----------|
    | Cell 1   | Cell 2   | Cell 3   |

    Can handle nested tables (tables within table cells) by detecting
    table boundaries and extracting them as separate structures.
    """

    @staticmethod
    def extract_tables_from_markdown(markdown_text: str) -> List[str]:
        """
        Extract all markdown tables from the given text.

        Args:
            markdown_text: Markdown-formatted text that may contain tables

        Returns:
            List of table strings (each table as a separate string)
        """
        if not markdown_text:
            return []

        # Pattern to match markdown tables (GFM format)
        # Matches lines starting with | and containing |
        # Must have at least a header row and separator row
        table_pattern = r'(\|[^\n]+\|(?:\n\|[-:\s|]+\|)(?:\n\|[^\n]+\|)*)'

        tables = re.findall(table_pattern, markdown_text, re.MULTILINE)

        # Filter out false positives (single rows that aren't actually tables)
        valid_tables = []
        for table in tables:
            lines = table.strip().split('\n')
            # A valid table must have at least 3 lines: header, separator, and at least one data row
            if len(lines) >= 3:
                # Check if second line is a separator (contains only |, -, :, and spaces)
                if re.match(r'^\|[\s\-:| ]+\|$', lines[1]):
                    valid_tables.append(table)

        return valid_tables

    @staticmethod
    def is_table_in_text(markdown_text: str) -> bool:
        """
        Check if the markdown text contains any tables.

        Args:
            markdown_text: Markdown-formatted text

        Returns:
            True if at least one table is found, False otherwise
        """
        tables = MarkdownTableExtractor.extract_tables_from_markdown(markdown_text)
        return len(tables) > 0

    @staticmethod
    def count_tables(markdown_text: str) -> int:
        """
        Count the number of tables in the markdown text.

        Args:
            markdown_text: Markdown-formatted text

        Returns:
            Number of tables found
        """
        tables = MarkdownTableExtractor.extract_tables_from_markdown(markdown_text)
        return len(tables)

    @staticmethod
    def remove_tables_from_text(markdown_text: str) -> str:
        """
        Remove all markdown tables from the text.

        Args:
            markdown_text: Markdown-formatted text

        Returns:
            Text with all tables removed
        """
        if not markdown_text:
            return ""

        # Pattern to match markdown tables
        table_pattern = r'(\|[^\n]+\|(?:\n\|[-:\s|]+\|)(?:\n\|[^\n]+\|)*)'

        # Remove all tables
        text_without_tables = re.sub(table_pattern, '', markdown_text, flags=re.MULTILINE)

        # Clean up multiple blank lines
        text_without_tables = re.sub(r'\n{3,}', '\n\n', text_without_tables)

        return text_without_tables.strip()
