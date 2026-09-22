"""Uploads are never trusted: content must match the extension, paths stay inside allowed roots,
limits are enforced, nothing is written, and errors/logs never expose paths, stack traces or content."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile

import pytest

from config import DocumentProcessingSettings
from models.documents import DocumentStatus
from parsers import DocumentProcessor
from parsers.base import BaseDocumentParser
from parsers.detection import safe_filename
from models.documents import FileType
from tests.fixtures.documents.builders import contract_text_pdf, image_bytes


@pytest.mark.parametrize("filename", ["contract.txt", "contract.doc", "setup.exe", "sheet.xlsx", "noextension", ".pdf"])
def test_unsupported_extensions_are_rejected(processor, filename):
    document = processor.parse_bytes(b"%PDF-1.7 whatever", filename)
    assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE
    assert [e.code for e in document.errors] == ["unsupported_file_type"]


@pytest.mark.parametrize("data, filename", [
    (b"MZ\x90\x00\x03\x00\x00\x00 executable header", "contract.pdf"),
    (b"#!/bin/sh\nrm -rf /\n", "contract.docx"),
    (b"<html><script>alert(1)</script></html>", "contract.pdf"),
])
def test_content_that_does_not_match_the_extension_is_rejected(processor, data, filename):
    document = processor.parse_bytes(data, filename)
    assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE
    assert [e.code for e in document.errors] == ["content_type_mismatch"]


def test_pdf_disguised_as_docx_and_zip_that_is_not_word_are_rejected(processor):
    assert processor.parse_bytes(contract_text_pdf(), "contract.docx").errors[0].code == "content_type_mismatch"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types><Override ContentType="application/vnd.openxmlformats-'
                                                'officedocument.spreadsheetml.sheet.main+xml"/></Types>')
        archive.writestr("xl/workbook.xml", "<workbook/>")
    document = processor.parse_bytes(buffer.getvalue(), "renamed-spreadsheet.docx")
    assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE and document.errors[0].code == "content_type_mismatch"


def test_macro_enabled_word_documents_are_rejected(processor, sample_bytes):
    with zipfile.ZipFile(io.BytesIO(sample_bytes["sample_contract_en.docx"])) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    entries["[Content_Types].xml"] = entries["[Content_Types].xml"].replace(
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        b"application/vnd.ms-word.document.macroEnabled.main+xml")
    entries["word/vbaProject.bin"] = b"\x00fake macro payload"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for name, data in entries.items():
            target.writestr(name, data)
    document = processor.parse_bytes(buffer.getvalue(), "contract.docx")
    assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE
    assert [e.code for e in document.errors] == ["macro_enabled_document"]


def test_archive_bomb_limits_are_enforced(sample_bytes):
    tight = DocumentProcessor(DocumentProcessingSettings(max_docx_uncompressed_bytes=10_000))
    document = tight.parse_bytes(sample_bytes["sample_contract_en.docx"], "contract.docx")
    assert document.status is DocumentStatus.FILE_TOO_LARGE and document.errors[0].code == "archive_too_large"
    few_entries = DocumentProcessor(DocumentProcessingSettings(max_docx_entries=3))
    assert few_entries.parse_bytes(sample_bytes["sample_contract_en.docx"], "c.docx").errors[0].code == "archive_too_large"


def test_file_size_limit_for_paths_and_bytes(tmp_path):
    processor = DocumentProcessor(DocumentProcessingSettings(max_file_size_bytes=1024))
    big = tmp_path / "big.pdf"
    big.write_bytes(contract_text_pdf())
    assert processor.parse_path(big).status is DocumentStatus.FILE_TOO_LARGE
    assert processor.parse_bytes(big.read_bytes(), "big.pdf").errors[0].code == "file_too_large"


def test_missing_file_and_directories_are_explicit_without_leaking_paths(processor, tmp_path):
    missing = processor.parse_path(tmp_path / "private-folder" / "contract.pdf")
    assert missing.status is DocumentStatus.FILE_NOT_FOUND and missing.filename == "contract.pdf"
    assert str(tmp_path) not in missing.model_dump_json()
    directory = tmp_path / "folder.pdf"
    directory.mkdir()
    result = processor.parse_path(directory)
    assert result.status is DocumentStatus.FILE_NOT_FOUND and result.errors[0].code == "not_a_regular_file"


def test_paths_outside_the_allowed_upload_root_are_refused(tmp_path):
    uploads, outside = tmp_path / "uploads", tmp_path / "outside"
    uploads.mkdir()
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(contract_text_pdf())
    (uploads / "ok.pdf").write_bytes(contract_text_pdf())
    (uploads / "link.pdf").symlink_to(outside / "secret.pdf")
    processor = DocumentProcessor(DocumentProcessingSettings(allowed_roots=(uploads,)))

    assert processor.parse_path(uploads / "ok.pdf").status is DocumentStatus.SUCCESS
    traversal = processor.parse_path(uploads / ".." / "outside" / "secret.pdf")
    assert traversal.status is DocumentStatus.PATH_NOT_ALLOWED and traversal.errors[0].code == "path_not_allowed"
    symlink = processor.parse_path(uploads / "link.pdf")
    assert symlink.status is DocumentStatus.PATH_NOT_ALLOWED
    assert traversal.full_text == "" and str(tmp_path) not in traversal.model_dump_json()


def test_empty_file_is_explicit(processor):
    document = processor.parse_bytes(b"", "contract.pdf")
    assert document.status is DocumentStatus.EMPTY_DOCUMENT and document.errors[0].code == "empty_file"


def test_declared_mime_type_must_match_content(processor):
    assert processor.parse_bytes(contract_text_pdf(), "a.pdf", declared_mime_type="application/pdf").status is (
        DocumentStatus.SUCCESS)
    mismatch = processor.parse_bytes(contract_text_pdf(), "a.pdf", declared_mime_type="image/png")
    assert mismatch.status is DocumentStatus.UNSUPPORTED_FILE_TYPE and mismatch.errors[0].code == "content_type_mismatch"


@pytest.mark.parametrize("fmt, filename", [("PNG", "scan.png"), ("JPEG", "scan.jpg"), ("JPEG", "photo.jpeg"),
                                           ("PNG", "SCAN.PNG")])
def test_standalone_jpeg_and_png_uploads_are_unsupported_in_the_mvp(processor, tmp_path, fmt, filename):
    data = image_bytes(fmt)
    path = tmp_path / filename
    path.write_bytes(data)
    for document in (processor.parse_bytes(data, filename), processor.parse_path(path)):
        assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE
        assert [e.code for e in document.errors] == ["unsupported_file_type"]
        assert document.file_type is FileType.UNKNOWN and document.full_text == "" and not document.ocr.required


@pytest.mark.parametrize("filename", ["disguised.pdf", "disguised.docx"])
def test_images_renamed_to_supported_extensions_are_rejected(processor, filename):
    document = processor.parse_bytes(image_bytes("PNG"), filename)
    assert document.status is DocumentStatus.UNSUPPORTED_FILE_TYPE
    assert [e.code for e in document.errors] == ["content_type_mismatch"]


def test_only_pdf_and_docx_are_accepted(processor):
    assert set(processor.supported_file_types) == {FileType.PDF, FileType.DOCX}
    assert {file_type.value for file_type in FileType} == {"pdf", "docx", "unknown"}


@pytest.mark.parametrize("given, expected", [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("C:\\Users\\someone\\Desktop\\contract.docx", "contract.docx"),
    ("/var/uploads/cv.png", "cv.png"),
    ("con\x00tract\x1b.pdf", "contract.pdf"),
    ("..", "unnamed"),
    ("", "unnamed"),
])
def test_filenames_are_reduced_to_a_safe_base_name(given, expected):
    assert safe_filename(given) == expected


def test_parse_bytes_never_keeps_directory_components(processor):
    document = processor.parse_bytes(contract_text_pdf(), "../../etc/uploads/contract.pdf")
    assert document.filename == "contract.pdf" and "etc/uploads" not in document.model_dump_json()


def test_parsing_writes_no_files(processor, sample_bytes, tmp_path, monkeypatch):
    workdir, temp = tmp_path / "cwd", tmp_path / "tmp"
    workdir.mkdir()
    temp.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.setenv("TMPDIR", str(temp))
    for name, data in sample_bytes.items():
        processor.parse_bytes(data, name)
    assert os.listdir(workdir) == [] and os.listdir(temp) == []


def test_logs_contain_no_document_content_or_personal_data(processor, sample_bytes, caplog):
    caplog.set_level(logging.DEBUG, logger="sanad")
    documents = [processor.parse_bytes(sample_bytes[name], name) for name in ("sample_cv_en.docx", "sample_contract_ar.docx")]
    logged = caplog.text
    assert documents[0].document_id in logged
    for forbidden in ("Jordan Sample", "jordan.sample@example.com", "+966 55 000 1234", "ريم الاختبار",
                      "sample_cv_en", "sample_contract_ar", "9,500"):
        assert forbidden not in logged


class _ExplodingParser(BaseDocumentParser):
    name = "exploding"
    file_types = (FileType.PDF,)

    def parse(self, source):
        raise RuntimeError("internal failure while reading /srv/private/uploads/contract.pdf with key sk-123")


def test_unexpected_parser_crash_becomes_sanitized_extraction_failed():
    processor = DocumentProcessor(DocumentProcessingSettings(), parser_classes=[_ExplodingParser])
    document = processor.parse_bytes(contract_text_pdf(), "contract.pdf")
    assert document.status is DocumentStatus.EXTRACTION_FAILED
    [error] = document.errors
    assert (error.code, error.exception_type) == ("extraction_failed", "RuntimeError")
    dumped = document.model_dump_json()
    assert "/srv/private" not in dumped and "sk-123" not in dumped and "Traceback" not in dumped
