"""
PDF loader for AMP schedules. Loads metadata and extracts PDFs into
an in-memory, consistent structure for downstream GTFS conversion.

The environment used in this workspace does not ship Python PDF libraries,
so the loader uses the system `pdftotext -layout` command when available.

Structure returned for each schedule:
{
    'codamp': str,
    'ut': str,
    'designa': str,
    'local_path': str,
    'pages': [
         {
             'text': str
         }
    ]
}
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import re
from typing import List, Dict, Any
import tempfile
import subprocess
import json


def load_metadata(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_with_pdftotext(path: str):
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext not available")
    # Try multiple extraction modes: -layout, default, -raw, and with explicit UTF-8
    # encoding. Some PDFs produce better text with different options.
    modes = [
        ["-layout"],
        [],
        ["-raw"],
        ["-enc", "UTF-8"],
        ["-enc", "ISO-8859-1"],
        ["-enc", "CP1252"],
    ]
    output = ""
    for mode in modes:
        try:
            cmd = ["pdftotext"] + mode + [path, "-"]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            output = ""
        # if extract produced at least one time token, accept it
        if re.search(r"\d{1,2}:\d{2}", output):
            break
    pages = output.split("\f") if output else []
    # drop trailing empty page if the converter emits one
    while pages and not pages[-1].strip():
        pages.pop()
    if pages:
        return [{"text": page.strip("\n")} for page in pages]

    # if pdftotext produced nothing usable, try OCR fallback
    ocr_pages = _ocr_with_tesseract(path)
    if ocr_pages:
        return ocr_pages

    return []


def _ocr_with_tesseract(path: str) -> List[Dict[str, Any]]:
    """Fallback OCR using pdftoppm + tesseract when pdftotext yields nothing.

    Returns a list of page dicts with an additional key `ocr: True`.
    """
    if shutil.which("pdftoppm") is None or shutil.which("tesseract") is None:
        return []
    tmpdir = tempfile.mkdtemp(prefix="pdf_ocr_")
    try:
        # convert pdf to png pages
        subprocess.check_call(["pdftoppm", "-png", "-r", "300", path, os.path.join(tmpdir, "page")])
        pages: List[Dict[str, Any]] = []
        images = sorted([f for f in os.listdir(tmpdir) if f.lower().endswith(".png")])
        for img in images:
            imgpath = os.path.join(tmpdir, img)
            outbase = imgpath
            # tesseract will write outbase.txt
            try:
                subprocess.check_call(["tesseract", imgpath, outbase, "-l", "por"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                txtpath = outbase + ".txt"
                if os.path.isfile(txtpath):
                    with open(txtpath, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                else:
                    txt = ""
            except Exception:
                txt = ""
            pages.append({"text": txt.strip(), "ocr": True})
        # drop trailing empty pages
        while pages and not pages[-1]["text"].strip():
            pages.pop()
        return pages
    except Exception:
        return []
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def extract_pdf(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return _extract_with_pdftotext(path)


def load_all_schedules_from_metadata(json_path: str, base_dir: str = ".", limit: int = None) -> List[Dict[str, Any]]:
    meta = load_metadata(json_path)
    schedules = meta.get("schedules", [])
    out = []
    count = 0
    for s in schedules:
        if s.get("download_status") != "success":
            continue
        local_path = s.get("local_path")
        if not local_path:
            continue
        md_dir = os.path.dirname(json_path)
        bn = os.path.basename(local_path)
        candidates = []
        if os.path.isabs(local_path):
            candidates.append(local_path)
        # try metadata dir + full local_path (may duplicate 'amp_schedules')
        candidates.append(os.path.join(md_dir, local_path))
        # try parent of metadata dir + local_path (covers cases where local_path is relative to repo subdir)
        parent_md = os.path.dirname(md_dir)
        if parent_md:
            candidates.append(os.path.join(parent_md, local_path))
        # try metadata dir + basename (common case where local_path already contains top folder)
        candidates.append(os.path.join(md_dir, bn))
        # try provided base_dir variants
        candidates.append(os.path.join(base_dir, local_path))
        candidates.append(os.path.join(base_dir, bn))
        # finally raw local_path
        candidates.append(local_path)

        full_path = None
        for cand in candidates:
            cand_norm = os.path.normpath(cand)
            if os.path.isfile(cand_norm):
                full_path = cand_norm
                break
        if full_path is None:
            # skip missing files
            continue
        try:
            pages = extract_pdf(full_path)
        except Exception as e:
            pages = [{"text": "", "words": []}]
        out.append({
            "codamp": s.get("codamp"),
            "idcarr": s.get("idcarr"),
            "idcar": s.get("idcar"),
            "geo_line_id": s.get("geo_line_id"),
            "geo_idop": s.get("geo_idop"),
            "ut": s.get("ut"),
            "designa": s.get("designa"),
            "local_path": full_path,
            "pages": pages,
        })
        count += 1
        if limit and count >= limit:
            break
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("metadata", help="Path to schedules_metadata.json")
    p.add_argument("--limit", type=int, default=5, help="How many schedules to load")
    args = p.parse_args()
    docs = load_all_schedules_from_metadata(args.metadata, base_dir=".", limit=args.limit)
    print(f"Loaded {len(docs)} schedules. Samples:")
    for d in docs[:3]:
        print(d["codamp"], d["ut"], len(d["pages"]))
