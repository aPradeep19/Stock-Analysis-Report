"""
PDF Preview Image

Turns the newsletter PDF into one tall PNG with every page stacked top to bottom. GitHub can't
embed a PDF in a README, but it can show an image, so the workflow publishes this next to the
PDF and the README links one to the other.

Usage: python -m src.preview output/premarket_<date>.pdf latest.png
"""

import sys

import pypdfium2 as pdfium
from PIL import Image

GAP = 16             # px of grey between pages so they read as separate sheets
GAP_COLOR = "#d0d0d0"


def render_preview(pdf_path: str, png_path: str, dpi: int = 110) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pages = [pdf[i].render(scale=dpi / 72).to_pil() for i in range(len(pdf))]
    finally:
        pdf.close()

    width = max(p.width for p in pages)
    height = sum(p.height for p in pages) + GAP * (len(pages) - 1)
    sheet = Image.new("RGB", (width, height), GAP_COLOR)
    y = 0
    for page in pages:
        sheet.paste(page, (0, y))
        y += page.height + GAP
    sheet.save(png_path, optimize=True)
    return png_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.preview <report.pdf> <preview.png>")
    print(f"[*] Preview image: {render_preview(sys.argv[1], sys.argv[2])}")
