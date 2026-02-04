from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, PowerpointFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from src.utils.config import DOCLING_ARTIFACTS_PATH, OCR_STATUS


class DocumentConverterService:
    """
    Singleton service for managing Docling document conversion with format-specific pipeline options.

    This service provides two pipeline configurations:
    - Standard Mode: generate_picture_images=True for PDFs/DOCX (extracts individual images/charts)
    - PPTX Mode: generate_page_images=True for PPTX-as-PDF (generates full slide images)

    Benefits:
    - Singleton pattern prevents repeated model loading across processes
    - Reduces memory footprint in multiprocessing scenarios
    - Centralizes Docling configuration
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentConverterService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _get_pipeline_options_for_pptx_as_pdf(self) -> PdfPipelineOptions:
        """
        Get pipeline options specifically for PPTX files downloaded as PDF.

        PPTX files are downloaded as PDF from SharePoint (content?format=pdf).
        This enables full slide/page image generation which wouldn't work with native PPTX.

        Configuration:
        - generate_page_images=True: Saves full slide images (page-0.png, page-1.png, etc.)
        - generate_picture_images=False: Skips individual icon extraction

        Returns:
            PdfPipelineOptions configured for slide-level extraction
        """
        pipeline_options = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS_PATH)
        pipeline_options.do_ocr = OCR_STATUS
        pipeline_options.images_scale = 2.0
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # PPTX-as-PDF: Generate full page/slide images, skip individual icons
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = False

        # Accelerator options for CPU processing
        accelerator_options = AcceleratorOptions(num_threads=3, device=AcceleratorDevice.CPU)
        pipeline_options.accelerator_options = accelerator_options

        return pipeline_options

    def _get_default_pipeline_options(self) -> PdfPipelineOptions:
        """
        Get default pipeline options for PDF and DOCX files.

        Configuration:
        - generate_page_images=False: No full page images (not needed for PDFs)
        - generate_picture_images=True: Extract individual images/charts

        Returns:
            PdfPipelineOptions configured for standard document processing
        """
        pipeline_options = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS_PATH)
        pipeline_options.do_ocr = OCR_STATUS
        pipeline_options.images_scale = 2.0
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # Standard: Extract individual images, no full page images
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        # Accelerator options for CPU processing
        accelerator_options = AcceleratorOptions(num_threads=3, device=AcceleratorDevice.CPU)
        pipeline_options.accelerator_options = accelerator_options

        return pipeline_options

    def _initialize(self):
        """Initialize the service (called once during singleton creation)."""
        # Initialization happens dynamically per conversion to support different pipeline modes
        pass

    def get_document_converter(self, is_pptx_as_pdf: bool = False) -> DocumentConverter:
        """
        Get document converter with appropriate pipeline options.

        Args:
            is_pptx_as_pdf: If True, uses PPTX pipeline (generate_page_images=True)
                           If False, uses default pipeline (generate_picture_images=True)

        Returns:
            Configured DocumentConverter instance
        """
        if is_pptx_as_pdf:
            # PPTX (downloaded as PDF): Enable full page/slide image generation
            pipeline_options = self._get_pipeline_options_for_pptx_as_pdf()
        else:
            # Standard PDF/DOCX: Use default options
            pipeline_options = self._get_default_pipeline_options()

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                InputFormat.PPTX: PowerpointFormatOption()
            },
        )
