from enum import Enum


class ChunkingParams(Enum):
    """
    Configuration parameters for token-aware document chunking.

    These parameters control the two-stage chunking algorithm:
    - Stage 1: Split on markdown headers (H1, H2, H3)
    - Stage 2: Verify token counts using tiktoken; recursive split if exceeds limit

    Benefits:
    - Accurate token counting prevents embedding dimension mismatches
    - Preserves document structure through header hierarchy
    - Configurable overlap for better context preservation
    """

    # Maximum tokens per chunk (default: 1000)
    # Chunks exceeding this will be recursively split while preserving headers
    chunk_size = 1000

    # Token overlap between consecutive chunks (default: 100)
    # Helps maintain context across chunk boundaries
    chunk_overlap = 100
