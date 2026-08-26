from __future__ import annotations

import unittest
from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from deep_research.infrastructure.pdf_extraction import (
    PdfExtractionError,
    _extract_pdf_inline,
    extract_pdf,
)


def make_pdf(
    text: str | None = None,
    *,
    pages: int = 1,
    title: str = "Evidence review",
    encrypted: bool = False,
) -> bytes:
    writer = PdfWriter()
    writer.add_metadata({"/Title": title})
    for index in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        content = DecodedStreamObject()
        page_text = f"{text} Page {index + 1}."
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)
    if encrypted:
        writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class PdfExtractionTests(unittest.TestCase):
    def test_extracts_clean_text_and_metadata_in_bounded_worker(self) -> None:
        raw = make_pdf(
            "A controlled study found a measurable reduction in emissions",
            pages=2,
        )

        result = extract_pdf(raw, timeout_seconds=5, max_pages=10)

        self.assertEqual("Evidence review", result.title)
        self.assertEqual(2, result.page_count)
        self.assertIn("measurable reduction in emissions", result.text)

    def test_rejects_image_only_pdf_instead_of_emitting_noise(self) -> None:
        with self.assertRaisesRegex(PdfExtractionError, "no extractable text"):
            _extract_pdf_inline(make_pdf(), max_pages=10, max_chars=10_000)

    def test_rejects_encrypted_pdf(self) -> None:
        with self.assertRaisesRegex(PdfExtractionError, "encrypted"):
            _extract_pdf_inline(
                make_pdf("Evidence remains inaccessible", encrypted=True),
                max_pages=10,
                max_chars=10_000,
            )

    def test_rejects_pdf_over_page_limit_before_extraction(self) -> None:
        with self.assertRaisesRegex(PdfExtractionError, "page limit"):
            _extract_pdf_inline(
                make_pdf("Evidence text", pages=3),
                max_pages=2,
                max_chars=10_000,
            )

    def test_rejects_low_signal_extracted_text(self) -> None:
        with self.assertRaisesRegex(PdfExtractionError, "low-signal"):
            _extract_pdf_inline(
                make_pdf("! " * 120),
                max_pages=10,
                max_chars=10_000,
            )

    def test_caps_output_including_page_separators(self) -> None:
        result = _extract_pdf_inline(
            make_pdf("A sufficiently detailed evidence statement", pages=4),
            max_pages=10,
            max_chars=100,
        )

        self.assertLessEqual(len(result.text), 100)


if __name__ == "__main__":
    unittest.main()
