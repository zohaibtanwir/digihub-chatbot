"""
Output Parser Module

Provides Python-based response formatting as a faster alternative to LLM-based
OUTPUT_PARSING_TEMPLATE. This saves ~2-3 seconds per request by avoiding an
additional LLM call.

Functions:
    format_response: Formats raw LLM response for display

To revert to LLM-based parsing:
    In response_formatter.py, comment out the Python implementation and
    uncomment the LLM-based implementation in parse_response().
"""

import re
from src.utils.logger import logger


def format_response(message: str) -> str:
    """
    Format the raw LLM response for display.

    This function performs the same transformations as OUTPUT_PARSING_TEMPLATE:
    1. Replace newlines with <br> tags
    2. Consolidate double slashes (// -> /) except in URLs
    3. Format image paths to ![Image](path)! syntax
    4. Add indentation for list items

    Args:
        message (str): The raw response text from the LLM

    Returns:
        str: The formatted response ready for display
    """
    if not message:
        return message

    result = message

    # 1. Replace newlines with <br> tags
    result = result.replace('\n', '<br>')

    # 2. Consolidate double slashes (but not in http:// or https://)
    # Use negative lookbehind to avoid matching protocol slashes
    result = re.sub(r'(?<!:)//', '/', result)

    # 3. Format image paths to ![Image](path)! syntax
    # Handle various malformed image tag patterns

    # Pattern 3a: Fix missing trailing ! on image tags
    # ![Image](path.png) -> ![Image](path.png)!
    result = re.sub(
        r'!\[Image\]\(([^)]+\.(png|jpg|jpeg))\)(?!!)',
        r'![Image](\1)!',
        result,
        flags=re.IGNORECASE
    )

    # Pattern 3b: Convert bare image paths to proper image tags
    # Look for paths ending in image extensions that aren't already in image tags
    # This handles: "image at Documents/image.jpeg" -> "image at ![Image](Documents/image.jpeg)!"
    result = re.sub(
        r'(?<!\[Image\]\()([A-Za-z0-9_\-./\\]+\.(png|jpg|jpeg))(?!\))',
        r'![Image](\1)!',
        result,
        flags=re.IGNORECASE
    )

    # 4. Add indentation for list items after <br>
    # Bullet points: - or •
    result = re.sub(r'<br>(\s*[-•]\s)', r'<br>&nbsp;&nbsp;&nbsp;&nbsp;\1', result)

    # Numbered lists: 1. 2. 3. etc.
    result = re.sub(r'<br>(\s*\d+\.\s)', r'<br>&nbsp;&nbsp;&nbsp;&nbsp;\1', result)

    # 5. Clean up any double <br> tags that might have been created
    result = re.sub(r'(<br>){3,}', '<br><br>', result)

    logger.info(f"[OutputParser] Response formatted using Python implementation")

    return result


def format_response_with_validation(message: str) -> str:
    """
    Format response with additional validation and logging.

    This wrapper function adds validation and comparison logging
    for debugging purposes.

    Args:
        message (str): The raw response text from the LLM

    Returns:
        str: The formatted response ready for display
    """
    if not message:
        logger.warning("[OutputParser] Empty message received")
        return message

    original_length = len(message)
    result = format_response(message)
    formatted_length = len(result)

    logger.debug(
        f"[OutputParser] Formatted response: "
        f"original={original_length} chars, formatted={formatted_length} chars"
    )

    return result
