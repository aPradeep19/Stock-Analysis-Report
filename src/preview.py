"""
Renders the newsletter's first page to a PNG.

GitHub READMEs can't embed a PDF but can show an image, so the workflow publishes this
thumbnail next to the PDF and the README links one to the other.

Usage: python -m src.preview output/premarket_<date>.pdf latest.png
"""

import sys

import pypdfium2 as pdfium


def render_preview(pdf_path: str, png_path: str, dpi: int = 110) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        # PDF units are points (1/72 inch), so scale = dpi / 72.
        image = pdf[0].render(scale=dpi / 72).to_pil()
    finally:
        pdf.close()

    image.save(png_path, optimize=True)
    return png_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.preview <report.pdf> <preview.png>")
    print(f"[*] Preview image: {render_preview(sys.argv[1], sys.argv[2])}")
