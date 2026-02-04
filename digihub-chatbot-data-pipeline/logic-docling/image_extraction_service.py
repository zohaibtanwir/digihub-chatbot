from pathlib import Path
from docling_core.types.doc import TextItem, ImageRefMode
import re
import os
import math
import asyncio
from typing import Optional, Dict, List, Tuple
from docling_core.types.doc.document import DoclingDocument, PictureItem
from langchain_core.messages import HumanMessage
from src.services.azure_services import AzureOpenAIService
from src.utils.logger import logger
from src.enums.image_conversion_params import ImageConversionParams
from src.services.prompt_vault import PromptVaultSingleton, PromptsEnum
import traceback

class ImageTextExtractionService:
    """
    Service for extracting text from images using intelligent routing between Docling OCR and Azure OpenAI Vision API.
    
    Implements cost-optimization by using free Docling OCR for simple images (<10 words) and Vision API for complex images.
    Supports async concurrent processing with configurable limits for Vision API calls.
    
    Key Points:
    - Vision API uses GPT-4o-mini with ~1000-2000 image tokens per image (NOT base64 text tokens)
    - Typical cost reduction: 60-80% by skipping Vision API for simple/empty images
    - Concurrent processing: Default 5 simultaneous Vision API calls (configurable)
    """
    VISION_EXTRACTION_PROMPT = PromptVaultSingleton().get_prompt(PromptsEnum.vision_extraction_prompt.value)

    def __init__(self, max_concurrent_extractions: int = ImageConversionParams.max_concurrent_extractions.value):
        """Initialize the service with Azure OpenAI client from the common service.

        Args:
            max_concurrent_extractions: maximum number of concurrent vision requests
                (used to limit parallel calls to the LLM/vision API).
                Defaults to ImageConversionParams.max_concurrent_extractions.value
        """
        azure_openai_service = AzureOpenAIService()
        self.llm = azure_openai_service.get_client()
        # concurrency limit for async extraction
        self.max_concurrent = max_concurrent_extractions
        # Hash-based mapping for picture lookup (built per document)
        self._hash_to_picture_idx: Dict[str, int] = {}

    def _extract_ocr_from_picture(self, doc: DoclingDocument, picture) -> Tuple[str, int]:
        """
        Extract OCR text from Docling for a picture in one pass.
        Returns both the formatted text and word count for efficient processing.
        
        Returns:
            tuple: (ocr_text: str, word_count: int)
        """
        lines = []
        all_words = []
        
        for item, _lvl in doc.iterate_items(root=picture, traverse_pictures=True):
            if isinstance(item, TextItem):
                text = (item.text or "").strip()
                if text:
                    lines.append(text)
                    all_words.extend(text.split())
        
        ocr_text = "\n".join(lines)
        word_count = len(all_words)
        return ocr_text, word_count

    # =========================================================================
    # Hash-based Picture Mapping Methods
    # =========================================================================
    
    def build_hash_to_picture_mapping(self, doc: DoclingDocument) -> Dict[str, int]:
        """
        Build a mapping from image hash to picture index.
        
        This allows us to identify the original picture from a saved image filename,
        even when saving by page (where the counter resets to 0 for each page).
        
        The filename format is: image_{counter}_{hexhash}.png
        The hash is unique per image, so we can use it as a lookup key.
        
        Returns:
            Dict mapping hexhash -> picture index (0 to len(pictures)-1)
        """
        self._hash_to_picture_idx = {}
        for idx, picture in enumerate(doc.pictures):
            img = picture.get_image(doc=doc)
            if img:
                hexhash = PictureItem._image_to_hexhash(img)
                if hexhash:
                    self._hash_to_picture_idx[hexhash] = idx
        logger.info(f"Built hash mapping for {len(self._hash_to_picture_idx)} pictures")
        return self._hash_to_picture_idx
    
    def get_picture_idx_from_filename(self, filename: str) -> Optional[int]:
        """
        Get the original picture index from a saved image filename.
        
        Args:
            filename: Image filename like 'image_000000_2e4a7be5029b3dd...png'
            
        Returns:
            Picture index (0 to len(pictures)-1) or None if not found
        """
        # Extract hash from filename: image_000000_{hash}.png
        basename = os.path.basename(filename)
        parts = basename.replace('.png', '').split('_')
        if len(parts) >= 3:
            hexhash = parts[2]
            return self._hash_to_picture_idx.get(hexhash)
        return None
    
    def get_picture_base64_uri_from_doc(self, doc: DoclingDocument, picture_idx: int) -> Optional[str]:
        """
        Get the full base64 data URI directly from the DoclingDocument picture.
        
        This avoids reading from file - the image data is already embedded in the document.
        
        Args:
            doc: DoclingDocument
            picture_idx: Index into doc.pictures
            
        Returns:
            Full data URI string (e.g., 'data:image/png;base64,...') or None
        """
        if picture_idx < 0 or picture_idx >= len(doc.pictures):
            return None
        
        picture = doc.pictures[picture_idx]
        if picture.image and picture.image.uri:
            uri = str(picture.image.uri)
            if uri.startswith('data:image/'):
                return uri
        
        return None

    async def _extract_text_with_vision_async(self, image_uri: str) -> str:
        """Async vision extraction using Azure OpenAI with base64 data URI.
        
        Args:
            image_uri: Base64 data URI (data:image/png;base64,...)
            
        Returns:
            Extracted text or empty string on failure.
        """
        try:
            user_message = HumanMessage(
                content=[
                    {"type": "text", "text": self.VISION_EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]
            )
            response = await self.llm.ainvoke([user_message])
            return response.content if hasattr(response, "content") else ""
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return ""

    def _get_image_path_from_markdown(self, image_markdown: str, base_path: str = None) -> Optional[str]:
        """
        Extract the image file path from markdown image syntax.
        Example: ![Image](path/to/image.png) -> path/to/image.png
        
        Args:
            image_markdown: The markdown image string
            base_path: Optional base path to resolve relative paths
            
        Returns:
            Full path to the image file, or None if extraction fails
        """
        # Extract path from ![alt](path) format
        match = re.search(r'!\[.*?\]\((.*?)\)', image_markdown)
        if not match:
            return None
            
        image_path = match.group(1).strip()
        
        # If path is already absolute, return it
        if os.path.isabs(image_path):
            # Verify the file exists
            return image_path if os.path.exists(image_path) else None
            
        # If base_path provided, join them
        if base_path:
            full_path = os.path.join(base_path, image_path)
            return full_path if os.path.exists(full_path) else None
            
        # If no base_path and path is relative, return None
        logger.warning(f"Relative image path without base_path: {image_path}")
        return None

    async def inject_image_text_into_markdown(
        self, 
        markdown_text: str, 
        doc: DoclingDocument, 
        base_path: str = None,
        min_words_for_vision: int = ImageConversionParams.min_words_for_vision.value,
        is_pptx_hybrid: bool = False
    ) -> Optional[str]:
        """
        Extract text from images in markdown and inject beneath each image.
        
        Uses hash-based lookup to map saved image filenames back to original pictures,
        then extracts base64 image data directly from the DoclingDocument.
        
        Args:
            markdown_text: Markdown text containing ![Image](path) references
            doc: DoclingDocument
            base_path: Base directory for resolving image paths (used for path validation)
            min_words_for_vision: Word count threshold for Vision API vs Docling OCR
            is_pptx_hybrid: If True and any image needs Vision API, return None to signal diversion
        
        Returns:
            Processed markdown string, or None if is_pptx_hybrid=True and Vision API is needed
        """
        try:
            pattern = r'!\[.*?\]\(.*?\)'
            matches = list(re.finditer(pattern, markdown_text))
        
            if not matches:
                logger.info("No images found in markdown")
                return markdown_text
            
            logger.info(f"Found {len(matches)} image references in markdown")
            
            # Build hash mapping if not already done
            if not self._hash_to_picture_idx:
                self.build_hash_to_picture_mapping(doc)
            
            # First pass: identify pictures from markdown and categorize them
            match_to_picture_idx: Dict[int, int] = {}
            images_for_vision: List[int] = []
            images_for_docling: Dict[int, str] = {}
            
            for i, match in enumerate(matches):
                image_md = match.group(0)
                image_path = self._get_image_path_from_markdown(image_md, base_path)
                if not image_path:
                    continue
                
                # Use hash-based lookup to get the original picture index
                filename = os.path.basename(image_path)
                picture_idx = self.get_picture_idx_from_filename(filename)
                
                if picture_idx is None:
                    logger.warning(f"Could not find picture for filename: {filename}")
                    continue
                
                match_to_picture_idx[i] = picture_idx
                picture = doc.pictures[picture_idx]
                
                # Extract OCR text and word count in one pass
                ocr_text, word_count = self._extract_ocr_from_picture(doc, picture)
                
                # Decide routing based on word count threshold
                if word_count >= min_words_for_vision:
                    # In PPTX hybrid mode, signal diversion to full-slide Vision API
                    if is_pptx_hybrid:
                        logger.info(f"PPTX hybrid: image {picture_idx} has {word_count} words, diverting to Vision API")
                        return None
                    if picture_idx not in images_for_vision:
                        images_for_vision.append(picture_idx)
                elif word_count > 0:
                    if picture_idx not in images_for_docling:
                        images_for_docling[picture_idx] = ocr_text
            
            logger.info(
                f"Image strategy: {len(images_for_vision)} Vision API, "
                f"{len(images_for_docling)} Docling OCR, "
                f"{len(match_to_picture_idx) - len(images_for_vision) - len(images_for_docling)} skipped"
            )
            
            # Build extraction tasks for Vision API - use base64 URI from document
            extraction_tasks = []
            for picture_idx in images_for_vision:
                base64_uri = self.get_picture_base64_uri_from_doc(doc, picture_idx)
                if base64_uri:
                    extraction_tasks.append((picture_idx, base64_uri))
                else:
                    logger.warning(f"Could not get base64 URI for picture {picture_idx}")
            
            # Run Vision API extractions concurrently
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def _extract_task(picture_idx: int, image_uri: str):
                async with semaphore:
                    text = await self._extract_text_with_vision_async(image_uri)
                    return picture_idx, text
            
            vision_results = {}
            if extraction_tasks:
                logger.info(f"Starting Vision API for {len(extraction_tasks)} images (using embedded base64)")
                coros = [_extract_task(idx, b64) for idx, b64 in extraction_tasks]
                completed = await asyncio.gather(*coros)
                vision_results = {idx: text for idx, text in completed}
            
            # Build new markdown by inserting extracted text
            out = markdown_text
            for i in reversed(range(len(matches))):
                match = matches[i]
                picture_idx = match_to_picture_idx.get(i)
                
                if picture_idx is None:
                    continue
                
                source_text = None
                if picture_idx in vision_results and vision_results[picture_idx]:
                    source_text = vision_results[picture_idx]
                elif picture_idx in images_for_docling:
                    source_text = images_for_docling[picture_idx]
                
                if source_text:
                    text_block = "\n\n" + source_text + "\n\n"
                    out = out[:match.end()] + text_block + out[match.end():]
            
            return out
        except Exception as e:
            logger.error(f"Error in inject_image_text_into_markdown: {e} - {traceback.format_exc()}")
            return markdown_text

    def _get_pictures_for_page(self, doc: DoclingDocument, page_no: int) -> List:
        """
        Get all pictures that belong to a specific page.
        
        Args:
            doc: DoclingDocument with pictures
            page_no: Page number to filter by
            
        Returns:
            List of pictures on that page
        """
        page_pictures = []
        for picture in doc.pictures:
            if hasattr(picture, 'prov') and picture.prov:
                if picture.prov[0].page_no == page_no:
                    page_pictures.append(picture)
        return page_pictures
    
    def _calculate_image_area_percentage(self, doc: DoclingDocument, page_no: int) -> float:
        """
        Calculate what percentage of the page is covered by images.
        Uses Docling's bbox data from picture provenance.
        
        Args:
            doc: DoclingDocument with pictures and pages
            page_no: Page number to analyze
            
        Returns:
            Percentage of page area covered by images (0.0 to 100.0)
        """
        # Get page dimensions
        if not hasattr(doc, 'pages') or page_no not in doc.pages:
            return 0.0
        
        page = doc.pages[page_no]
        
        # Try to get page size - handle different Docling versions
        page_width = 0.0
        page_height = 0.0
        
        if hasattr(page, 'size'):
            if hasattr(page.size, 'width') and hasattr(page.size, 'height'):
                page_width = page.size.width
                page_height = page.size.height
        elif hasattr(page, 'width') and hasattr(page, 'height'):
            page_width = page.width
            page_height = page.height
        
        page_area = page_width * page_height
        
        if page_area == 0:
            logger.debug(f"Page {page_no}: Could not determine page dimensions")
            return 0.0
        
        # Sum image areas on this page
        total_image_area = 0.0
        for picture in doc.pictures:
            if hasattr(picture, 'prov') and picture.prov:
                if picture.prov[0].page_no == page_no:
                    bbox = picture.prov[0].bbox
                    if bbox:
                        # bbox typically has: l (left), t (top), r (right), b (bottom)
                        # or x, y, width, height depending on Docling version
                        if hasattr(bbox, 'r') and hasattr(bbox, 'l'):
                            image_width = abs(bbox.r - bbox.l)
                            image_height = abs(bbox.b - bbox.t)
                        elif hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                            image_width = bbox.width
                            image_height = bbox.height
                        else:
                            continue
                        total_image_area += image_width * image_height
        
        percentage = (total_image_area / page_area) * 100
        return min(percentage, 100.0)  # Cap at 100%

    def _analyze_page_metrics(self, doc: DoclingDocument, page_no: int) -> dict:
        """
        Analyze a page and return metrics for routing decision.
        Centralized page analysis - easy to extend with new conditions.
        
        Args:
            doc: DoclingDocument
            page_no: Page number to analyze
            
        Returns:
            Dict with metrics: image_count, image_area_percentage, etc.
        """
        # Count images on this page
        image_count = 0
        for picture in doc.pictures:
            if hasattr(picture, 'prov') and picture.prov:
                if picture.prov[0].page_no == page_no:
                    image_count += 1
        
        # Calculate image area percentage
        image_area_pct = self._calculate_image_area_percentage(doc, page_no)
        
        return {
            'page_no': page_no,
            'image_count': image_count,
            'image_area_percentage': image_area_pct
        }

    def _should_use_vision_for_page(
        self, 
        metrics: dict,
        image_count_threshold: int = ImageConversionParams.pptx_slide_image_threshold.value,
        image_area_threshold: float = ImageConversionParams.pptx_slide_image_area_threshold.value
    ) -> Tuple[bool, str]:
        """
        Decide whether to use Vision API for a page based on metrics.
        Centralized decision logic - easy to modify conditions.
        
        Args:
            metrics: Dict from _analyze_page_metrics()
            image_count_threshold: Max images before using Vision
            image_area_threshold: Max image area % before using Vision
            
        Returns:
            Tuple of (use_vision: bool, reason: str)
        """
        image_count = metrics.get('image_count', 0)
        image_area_pct = math.ceil(metrics.get('image_area_percentage', 0.0))
        
        # Condition 1: Too many images (>= threshold)
        if image_count >= image_count_threshold:
            return True, f"{image_count} images (threshold: {image_count_threshold})"
        
        # Condition 2: Images cover too much area (>= threshold)
        if image_area_pct >= image_area_threshold:
            return True, f"{image_area_pct:.1f}% image area (threshold: {image_area_threshold}%)"
        
        return False, f"{image_count} images, {image_area_pct:.1f}% area"

    async def convert_pptx_to_markdown_hybrid(
        self,
        doc: DoclingDocument,
        file_name: str,
        base_dir: Optional[str] = None,
        image_count_threshold: int = ImageConversionParams.pptx_slide_image_threshold.value,
        image_area_threshold: float = ImageConversionParams.pptx_slide_image_area_threshold.value
    ) -> str:
        """
        PPTX Hybrid Processing: Process slides conditionally based on page metrics.
        
        Uses streaming approach - processes and writes each page immediately to reduce memory.
        
        For each slide, analyzes metrics and routes:
        - Vision API: If image count > threshold OR image area > threshold
        - Standard: Otherwise (text + image-by-image processing)
        
        IMPORTANT: Both paths include textual content from the slide!
        - Vision path: Full slide → Vision API gets text + layout
        - Standard path: Docling text extraction + individual image OCR/Vision
        
        Args:
            doc: DoclingDocument from Docling converter
            file_name: Source filename (e.g., "presentation.pptx")
            base_dir: Base directory for artifacts
            image_count_threshold: If slide has more images than this, use Vision
            image_area_threshold: If images cover more than this % of slide, use Vision
            
        Returns:
            Combined markdown with slide-by-slide processing (text + images)
        """
        try:
            # Setup directories
            if base_dir is None:
                base_dir = os.getcwd()
            
            markdown_artifacts_dir = os.path.join(base_dir, 'markdown_artifacts')
            os.makedirs(markdown_artifacts_dir, exist_ok=True)
            
            file_name_only = os.path.basename(file_name)
            file_name_without_ext = os.path.splitext(file_name_only)[0]
            doc_artifacts_dir = Path(os.path.join(markdown_artifacts_dir, f"{file_name_without_ext}_artifacts"))
            os.makedirs(doc_artifacts_dir, exist_ok=True)
            
            # Save markdown with all artifacts (page images + individual images)
            markdown_file_path = os.path.join(markdown_artifacts_dir, f"{file_name_without_ext}.md")
            doc.save_as_markdown(
                markdown_file_path,
                image_mode=ImageRefMode.REFERENCED,
                artifacts_dir=doc_artifacts_dir
            )
            
            logger.info(f"PPTX artifacts saved to: {doc_artifacts_dir}")

            # Get all pages from doc.pages (keys are ints in DoclingDocument)
            sorted_pages = sorted(doc.pages.keys(), key=int)
            
            if not sorted_pages:
                logger.warning("No pages found in document")
                with open(markdown_file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            total_pages = len(sorted_pages)
            logger.info(f"Processing {total_pages} slides with streaming approach")
            
            # ============================================================
            # Phase 1: Analyze all pages and categorize
            # ============================================================
            vision_pages = []  # Pages that need Vision API
            standard_pages = []  # Pages for standard processing
            page_metrics = {}  # Store metrics for logging
            
            for page_no in sorted_pages:
                metrics = self._analyze_page_metrics(doc, page_no)
                page_metrics[page_no] = metrics
                
                use_vision, reason = self._should_use_vision_for_page(
                    metrics, 
                    image_count_threshold, 
                    image_area_threshold
                )
                
                if use_vision:
                    vision_pages.append(page_no)
                    logger.info(f"Slide {page_no}: Vision API - {reason}")
                else:
                    standard_pages.append(page_no)
                    logger.info(f"Slide {page_no}: Standard - {reason}")
            
            # Build hash-based mapping for picture lookup (ONCE for entire document)
            self.build_hash_to_picture_mapping(doc)
            
            # ============================================================
            # Phase 2: Process Vision pages (batch) - use embedded base64 from page.image.uri
            # ============================================================
            vision_results = {}
            if vision_pages:
                # Build page_no -> base64 URI mapping directly from page.image.uri
                page_base64_uris = {}
                for page_no in vision_pages:
                    if page_no in doc.pages:
                        page = doc.pages[page_no]
                        if hasattr(page, 'image') and page.image is not None:
                            # Get URI directly from page.image.uri (already base64 encoded)
                            if hasattr(page.image, 'uri') and page.image.uri:
                                page_base64_uris[page_no] = str(page.image.uri)
                
                semaphore = asyncio.Semaphore(self.max_concurrent)
                
                async def _process_vision_slide(page_no: int, image_uri: str):
                    async with semaphore:
                        description = await self._extract_text_with_vision_async(image_uri)
                        return page_no, description
                
                tasks = [
                    _process_vision_slide(p, page_base64_uris[p]) 
                    for p in vision_pages if p in page_base64_uris
                ]
                results = await asyncio.gather(*tasks)
                vision_results = {page_no: desc for page_no, desc in results}
                logger.info(f"Vision API processed {len(vision_results)} slides")
            
            # ============================================================
            # Phase 3: Stream output - process standard pages and write immediately
            # ============================================================
            output_file_path = os.path.join(markdown_artifacts_dir, f"{file_name_without_ext}_processed.md")
            
            vision_count = 0
            standard_count = 0
            
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                # Write header
                out_file.write(f"# {file_name_without_ext}\n\n")
                
                for page_no in sorted_pages:
                    metrics = page_metrics[page_no]
                    image_count = metrics['image_count']
                    image_area_pct = metrics['image_area_percentage']
                
                    # Write slide header
                    logger.info(f"Processing slide {page_no}")
                    out_file.write(f"## Slide {page_no}\n\n")
                    
                    if page_no in vision_results:
                        # Vision API result (already processed)
                        # Add slide image reference
                        relative_image_path = f"{file_name_without_ext}_artifacts/page-{page_no}.png"
                        out_file.write(f"![Slide {page_no}]({relative_image_path})\n\n")
                        if vision_results[page_no]:
                            out_file.write(f"{vision_results[page_no]}\n\n")
                        logger.info(f"*[Vision API - {image_count} images, {image_area_pct:.1f}% area]*\n\n")
                        vision_count += 1
                        
                    else:
                        # Standard processing path - try to process with Docling OCR
                        # If any image needs Vision API, inject_image_text_into_markdown returns None
                        page_md_filename = f"{file_name_without_ext}_page_{page_no}.md"
                        page_md_path = os.path.join(markdown_artifacts_dir, page_md_filename)
                        
                        doc.save_as_markdown(
                            page_md_path,
                            image_mode=ImageRefMode.REFERENCED,
                            artifacts_dir=doc_artifacts_dir,
                            page_no=page_no
                        )
                        
                        slide_markdown = ""
                        if os.path.exists(page_md_path):
                            with open(page_md_path, 'r', encoding='utf-8') as page_file:
                                slide_markdown = page_file.read()
                        
                        if slide_markdown:
                            # Try processing - returns None if any image needs Vision API
                            processed_markdown = await self.inject_image_text_into_markdown(
                                markdown_text=slide_markdown,
                                doc=doc,
                                base_path=str(doc_artifacts_dir),
                                is_pptx_hybrid=True
                            )
                            
                            if processed_markdown is None:
                                # Divert to full-slide Vision API
                                if page_no in doc.pages:
                                    page = doc.pages[page_no]
                                    if hasattr(page, 'image') and page.image is not None:
                                        page_uri = str(page.image.uri) if hasattr(page.image, 'uri') and page.image.uri else None
                                        if page_uri:
                                            slide_description = await self._extract_text_with_vision_async(page_uri)
                                            if slide_description:
                                                out_file.write(f"{slide_description}\n\n")
                                
                                logger.info(f"*[Vision API (diverted) - {image_count} images, {image_area_pct:.1f}% area]*\n\n")
                                vision_count += 1
                            else:
                                # Standard processing succeeded
                                out_file.write(f"{processed_markdown}\n\n")
                                logger.info(f"*[Standard - {image_count} images, {image_area_pct:.1f}% area]*\n\n")
                                standard_count += 1
                        else:
                            logger.info(f"*[Standard (no content) - {image_count} images]*\n\n")
                            standard_count += 1
                    
                    out_file.write("---\n\n")
                    
                    # Flush periodically to release memory
                    if (page_no) % 10 == 0:
                        out_file.flush()
                        logger.debug(f"Processed {page_no}/{total_pages} slides")            
                        
                logger.info(f"PPTX hybrid processing complete: {vision_count} via Vision, {standard_count} via Standard")
            
            # Read the final output
            with open(output_file_path, 'r', encoding='utf-8') as f:
                final_markdown = f.read()
            
            return final_markdown
        except Exception as e:
            logger.error(f"Error in convert_pptx_to_markdown_hybrid: {e} - {traceback.format_exc()}")
            raise

    async def convert_document_to_markdown_with_images(
        self,
        doc: DoclingDocument,
        file_name: str,
        base_dir: Optional[str] = None
    ) -> str:
        """
        Convert a Docling document to markdown with image text extraction.
        
        This method:
        1. Creates artifact directories for storing images
        2. Saves the document as markdown with referenced images
        3. Injects extracted text from images using Vision API
        
        Args:
            doc: The DoclingDocument from Docling document converter
            file_name: Name of the source file (e.g., "document.pdf")
            base_dir: Base directory for artifacts (defaults to /app in Docker)
        
        Returns:
            Markdown string with injected image text
        """
        try:
            # Define base paths - use current working directory (set to /app in Docker)
            if base_dir is None:
                base_dir = os.getcwd()
            
            markdown_artifacts_dir = os.path.join(base_dir, 'markdown_artifacts')
            
            # Create markdown_artifacts directory if it doesn't exist
            os.makedirs(markdown_artifacts_dir, exist_ok=True)
            
            # Create subdirectory using filename without extension + "_artifacts" (Docling convention)
            file_name_only = os.path.basename(file_name)
            file_name_without_ext = os.path.splitext(file_name_only)[0]
            doc_artifacts_dir = Path(os.path.join(markdown_artifacts_dir, f"{file_name_without_ext}_artifacts"))
            os.makedirs(doc_artifacts_dir, exist_ok=True)
            
            # Save markdown with artifacts (images will be saved to doc_artifacts_dir)
            markdown_file_path = os.path.join(markdown_artifacts_dir, f"{file_name_without_ext}.md")
            doc.save_as_markdown(
                markdown_file_path,
                image_mode=ImageRefMode.REFERENCED,
                artifacts_dir=doc_artifacts_dir
            )
            
            logger.info(f"Markdown and artifacts saved to: {doc_artifacts_dir}")
            
            # Read the saved markdown file
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                markdown_string = f.read()
        
            # Inject image text using Vision API with saved artifacts
            markdown_with_ocr = await self.inject_image_text_into_markdown(
                markdown_text=markdown_string,
                doc=doc,
                base_path=doc_artifacts_dir
            )
            
            return markdown_with_ocr
        except Exception as e:
            logger.error(f"Error in convert_document_to_markdown_with_images: {e} - {traceback.format_exc()}")
            raise