"""Extract structural units before any LLM sees document contents."""

import csv
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

from backend.core.errors import ApplicationError


@dataclass
class SourceUnit:
    text: str
    provenance: dict


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".docx", ".pdf"}


def table_units(rows, table: int, page: int | None = None, header: bool = False):
    """Never combine data rows. Only explicit/recognized headers become labels."""
    rows = [[str(cell or "").strip() for cell in row] for row in rows]
    if not rows:
        return []
    labels = rows[0] if header else [f"Column {n + 1}" for n in range(len(rows[0]))]
    output = []
    for row_number, values in enumerate(rows[1:] if header else rows, 2 if header else 1):
        if not any(values):
            continue
        columns = [
            {
                "name": labels[n] if n < len(labels) and labels[n] else f"Column {n + 1}",
                "value": value,
            }
            for n, value in enumerate(values)
        ]
        output.append(
            SourceUnit(
                "\n".join(f"{column['name']}: {column['value']}" for column in columns),
                {
                    "kind": "row",
                    "table": table,
                    "page": page,
                    "row": row_number,
                    "columns": columns,
                },
            )
        )
    return output


def _looks_like_header(row):
    # Conservative: unknown rows remain data rather than being silently discarded.
    names = {
        "id",
        "requirement",
        "specification",
        "parameter",
        "value",
        "unit",
        "units",
        "description",
        "criterion",
        "limit",
        "standard",
        "name",
        "status",
        "test",
        "minimum",
        "maximum",
        "material",
        "property",
        "result",
    }
    values = [str(value or "").strip().lower() for value in row]
    return bool(values) and all(value in names for value in values)


def text_units(text: str, page: int | None = None):
    output = []
    lines = text.replace("\r\n", "\n").splitlines()
    paragraph = []
    table = 0

    def flush():
        if paragraph:
            output.append(SourceUnit("\n".join(paragraph), {"kind": "prose", "page": page}))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        # Markdown tables and tab-separated blocks preserve physical rows.
        if (
            "|" in line
            and index + 1 < len(lines)
            and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", lines[index + 1])
        ):
            flush()
            rows = [[v.strip() for v in line.strip().strip("|").split("|")]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append([v.strip() for v in lines[index].strip().strip("|").split("|")])
                index += 1
            table += 1
            output.extend(table_units(rows, table, page, header=True))
            continue
        if "\t" in line:
            flush()
            rows = []
            while index < len(lines) and "\t" in lines[index]:
                rows.append(lines[index].split("\t"))
                index += 1
            table += 1
            output.extend(table_units(rows, table, page, header=_looks_like_header(rows[0])))
            continue
        if not line.strip():
            flush()
        elif re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line):
            flush()
            output.append(SourceUnit(line.strip(), {"kind": "prose", "page": page}))
        else:
            paragraph.append(line.strip())
        index += 1
    flush()
    return output


def parse_document(filename: str, content: bytes, max_units: int = 1000):
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ApplicationError(
            "unsupported_document", "Use PDF, DOCX, TXT, Markdown, CSV or TSV.", 415
        )
    try:
        if extension in {".txt", ".md", ".csv", ".tsv"}:
            text = content.decode("utf-8-sig")
            if "\x00" in text:
                raise ValueError("Binary content")
            if extension in {".csv", ".tsv"}:
                rows = list(
                    csv.reader(
                        StringIO(text), delimiter="\t" if extension == ".tsv" else ",", strict=True
                    )
                )
                units = table_units(rows, 1, header=True)
            else:
                units = text_units(text)
        elif extension == ".docx":
            from docx import Document
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            with ZipFile(BytesIO(content)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 50 * 1024 * 1024:
                    raise ApplicationError(
                        "document_too_large", "Expanded DOCX exceeds 50 MB.", 413
                    )
            document = Document(BytesIO(content))
            units, table = [], 0
            for block in document.iter_inner_content():
                if isinstance(block, Paragraph) and block.text.strip():
                    units.extend(text_units(block.text))
                elif isinstance(block, Table):
                    if any(cell.tables for row in block.rows for cell in row.cells):
                        raise ApplicationError(
                            "unsupported_table_layout",
                            "Flatten nested DOCX tables before upload so each row can be indexed separately.",
                            422,
                        )
                    if block._tbl.xpath(".//w:vMerge"):
                        raise ApplicationError(
                            "unsupported_table_layout",
                            "Unmerge vertically merged DOCX cells before upload to preserve row boundaries.",
                            422,
                        )
                    table += 1
                    rows = [[cell.text for cell in row.cells] for row in block.rows]
                    explicit = bool(block.rows) and any(
                        node.get(qn("w:val"), "1") not in {"0", "false", "off"}
                        for node in block.rows[0]._tr.xpath("./w:trPr/w:tblHeader")
                    )
                    units.extend(
                        table_units(
                            rows,
                            table,
                            header=explicit or bool(rows and _looks_like_header(rows[0])),
                        )
                    )
        else:
            import pdfplumber

            units, table_number = [], 0
            with pdfplumber.open(BytesIO(content)) as document:
                if len(document.pages) > 500:
                    raise ApplicationError("document_too_large", "PDF exceeds 500 pages.", 413)
                for page_number, page in enumerate(document.pages, 1):
                    if not page.chars and page.images:
                        raise ApplicationError(
                            "document_requires_ocr",
                            f"PDF page {page_number} has no extractable text. Run OCR before upload.",
                            422,
                        )
                    tables = page.find_tables()
                    body = page
                    for table in tables:
                        table_number += 1
                        rows = table.extract()
                        units.extend(
                            table_units(
                                rows,
                                table_number,
                                page_number,
                                header=bool(rows and _looks_like_header(rows[0])),
                            )
                        )
                        body = body.outside_bbox(table.bbox)
                    units.extend(text_units(body.extract_text() or "", page_number))
                    if len(units) > max_units:
                        raise ApplicationError(
                            "too_many_chunks", "Document exceeds the configured chunk limit.", 413
                        )
    except ApplicationError:
        raise
    except Exception as exc:  # noqa: BLE001 - parser errors are not exposed to clients
        raise ApplicationError(
            "document_parse_failed",
            "Cannot parse this document. Check its format and encoding; encrypted PDFs are unsupported.",
            422,
        ) from exc
    if not units:
        raise ApplicationError(
            "document_empty", "No extractable content. Scanned PDFs require OCR before upload.", 422
        )
    if len(units) > max_units:
        raise ApplicationError(
            "too_many_chunks", "Document exceeds the configured chunk limit.", 413
        )
    return units
