"""
Centralized partition key generation utilities.

This module provides consistent partition key generation across the entire codebase,
preventing mismatches between different modules (dataprocessor.py, eventhub.py, bulkindex.py).

Partition key format: {folder_name}-{filename_normalized}
- Lowercase
- Spaces removed
- Special characters replaced with underscore
"""

import re


def generate_partition_key(folder_name: str, filename: str) -> str:
    """
    Generate a consistent partition key for Cosmos DB documents.

    This function ensures that partition keys are generated consistently across
    all modules, preventing query/delete failures due to mismatches.

    Args:
        folder_name: Name of the folder/service (e.g., "MyService")
        filename: Name of the file (e.g., "My Document.pdf")

    Returns:
        Normalized partition key string (e.g., "myservice-mydocument.pdf")

    Examples:
        >>> generate_partition_key("MyService", "My Document.pdf")
        'myservice-mydocument.pdf'
        >>> generate_partition_key("Test Folder", "file with spaces.docx")
        'testfolder-filewithspaces.docx'
        >>> generate_partition_key("Path/With/Slashes", "document.pdf")
        'path_with_slashes-document.pdf'
    """
    # Normalize folder name
    # 1. Convert to lowercase
    # 2. Remove spaces
    # 3. Replace forward slashes and backslashes with underscores
    # 4. Replace other special characters with underscores
    normalized_folder = folder_name.lower()
    normalized_folder = normalized_folder.replace(' ', '')
    normalized_folder = normalized_folder.replace('/', '_').replace('\\', '_')
    # Remove any remaining special characters that could cause issues
    normalized_folder = re.sub(r'[^a-z0-9_-]', '_', normalized_folder)

    # Normalize filename
    # 1. Convert to lowercase
    # 2. Remove spaces
    # 3. Replace forward slashes and backslashes with underscores
    # 4. Keep the extension (don't normalize dots)
    normalized_filename = filename.lower()
    normalized_filename = normalized_filename.replace(' ', '')
    normalized_filename = normalized_filename.replace('/', '_').replace('\\', '_')
    # Remove special characters except dots (for file extensions)
    normalized_filename = re.sub(r'[^a-z0-9._-]', '_', normalized_filename)

    # Combine with hyphen separator
    return f"{normalized_folder}-{normalized_filename}"


def generate_chunk_id(folder_name: str, filename: str, chunk_index: int, content: str) -> str:
    """
    Generate a deterministic chunk ID based on content hash.

    This ensures idempotency - the same content will always generate the same ID,
    preventing duplicate chunks when processing the same file multiple times.

    Args:
        folder_name: Name of the folder/service
        filename: Name of the file
        chunk_index: Index of the chunk within the document (0-based)
        content: Content of the chunk (used for hash)

    Returns:
        Deterministic chunk ID string

    Examples:
        >>> generate_chunk_id("service", "doc.pdf", 0, "content")
        'service-doc.pdf-chunk-0-9a0364b9'
    """
    import hashlib

    # Generate content hash (first 8 chars of MD5 hex digest)
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    # Normalize folder and filename
    normalized_folder = folder_name.lower().replace(' ', '')
    normalized_filename = filename.lower().replace(' ', '')

    # Format: {folder}-{filename}-chunk-{index}-{hash}
    return f"{normalized_folder}-{normalized_filename}-chunk-{chunk_index}-{content_hash}"


def validate_partition_key(partition_key: str) -> bool:
    """
    Validate that a partition key follows the expected format.

    Args:
        partition_key: The partition key string to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_partition_key("service-document.pdf")
        True
        >>> validate_partition_key("Service-Document With Spaces.pdf")
        False
        >>> validate_partition_key("")
        False
    """
    if not partition_key:
        return False

    # Must contain at least one hyphen (separator)
    if '-' not in partition_key:
        return False

    # Must be lowercase
    if partition_key != partition_key.lower():
        return False

    # Must not contain spaces
    if ' ' in partition_key:
        return False

    # Must only contain allowed characters: lowercase letters, numbers, dots, hyphens, underscores
    if not re.match(r'^[a-z0-9._-]+$', partition_key):
        return False

    return True


def extract_filename_from_partition_key(partition_key: str) -> str:
    """
    Extract the normalized filename from a partition key.

    Args:
        partition_key: The partition key string (e.g., "service-document.pdf")

    Returns:
        The normalized filename portion (e.g., "document.pdf")

    Examples:
        >>> extract_filename_from_partition_key("myservice-mydocument.pdf")
        'mydocument.pdf'
        >>> extract_filename_from_partition_key("test-folder-with-hyphens-file.docx")
        'file.docx'
    """
    # Split on hyphen and take the last part (assuming filename is at the end)
    # This is a best-effort approach; for more complex cases, additional context is needed
    parts = partition_key.split('-')
    if len(parts) < 2:
        return partition_key

    # Find the first part that contains a dot (likely the filename with extension)
    for i in range(len(parts) - 1, -1, -1):
        if '.' in parts[i]:
            # Rejoin from this point onwards
            return '-'.join(parts[i:])

    # If no dot found, return the last part
    return parts[-1]
