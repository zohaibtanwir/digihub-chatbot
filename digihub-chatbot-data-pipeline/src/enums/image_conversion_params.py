from enum import Enum


class ImageConversionParams(Enum):
    """
    Configuration parameters for intelligent image text extraction.

    These parameters control the dual-path routing strategy:
    - Docling OCR (free): For simple images with few words
    - Azure Vision API (paid): For complex images with many words

    PPTX hybrid parameters control when to use full-slide Vision API vs
    individual image processing for PowerPoint presentations.
    """

    # Concurrent processing limit for Vision API calls
    # Controls how many Vision API requests run in parallel (default: 5)
    max_concurrent_extractions = 5

    # Word count threshold for routing decision
    # Images with >= this many words route to Vision API, otherwise Docling OCR (default: 10)
    min_words_for_vision = 10

    # PPTX slide complexity thresholds
    # If a slide has >= this many images, use full-slide Vision API (default: 5)
    pptx_slide_image_threshold = 5

    # If images cover >= this % of slide area, use full-slide Vision API (default: 30.0%)
    pptx_slide_image_area_threshold = 30.0
