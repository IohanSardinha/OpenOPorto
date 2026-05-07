"""Parse AMP schedule PDFs into structured timetable data.

The PDFs are not GTFS yet, but this module converts them into an in-memory
representation that is much closer to GTFS:

- each schedule keeps its metadata
- each timetable block keeps a direction/service label
- each block stores the stop sequence and per-stop rows
- each block also stores transposed trips, which are easy to map to GTFS trips

The parser handles three layout families:

- UT1: unique directional matrix layout
- UT2, UT3, UT4: shared principal-stops matrix layout
- UT5: unique directional matrix layout with footnotes

The system environment for this workspace provides `pdftotext`, so text is
extracted from the PDFs before parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pdf_loader import load_all_schedules_from_metadata, load_metadata

TIME_RE = re.compile(r"\b\d{2}:\d{2}\b")
UT234_HEADER_RE = re.compile(r"^\s*PARAGENS PRINCIPAIS\b(?:\s+(?P<label>.*))?$", re.IGNORECASE)
DIR_RE = re.compile(r"\b(IDA|VOLTA|OUTBOUND|INBOUND)\b", re.IGNORECASE)
FOOTER_RE = re.compile(
    r"(HORÁRIO EM VIGOR|OS HORÁRIOS PODEM VARIAR|PODEM EXISTIR OUTRAS PARAGENS|"
    r"ESTE HORÁRIO ANULA E SUBSTITUI|EDIÇÃO DE)",
    re.IGNORECASE,
)
ARROW_RE = re.compile(r"^\s*[▼▲]")
SERVICE_DAY_RE = re.compile(
    r"(DIAS\s+ÚTEIS|BUSINESS\s+DAYS|\\bDU\\b)|" +
    r"(SÁBADOS|SATURDAY|\\bSAB\\b)|" +
    r"(DOMINGOS|FERIADOS|SUNDAY|HOLIDAYS|FESTIVOS|DOM\\s+e\\s+FER|DOM e FER)",
    re.IGNORECASE
)
SERVICE_SPLIT_RE = re.compile(
    r"(DIAS\s+ÚTEIS|BUSINESS\s+DAYS|\bDU\b|"
    r"SÁBADOS|SATURDAYS|\bSAB\b|"
    r"DOMINGOS(?:\s+E\s+FERIADOS)?|SUNDAYS|HOLIDAYS|"
    r"FINS\s+DE\s+SEMANA\s+E\s+FERIADOS|NON-BUSINESS\s+DAYS|DOM\s+e\s+FER)",
    re.IGNORECASE,
)


def _detect_service_day(lines: Sequence[str]) -> str:
    """Detect service day from header lines.
    
    Returns:
    - "DU": Dias Úteis (weekdays)
    - "SAB": Sábado (Saturday) only
    - "DOM": Domingo/Feriados (Sunday/Holidays) only  
    - "SAB+DOM": Both Saturday and Sunday/Holidays in separate columns
    - "unknown": Unable to determine
    """
    joined = " ".join(lines).upper()
    
    # Check for explicit SAB and DOM columns on the same line
    has_sab = "SAB" in joined
    has_dom_col = "DOM" in joined and ("e FER" in joined or "FERIADO" in joined or "HOLIDAY" in joined)
    
    if has_sab and has_dom_col:
        return "SAB+DOM"
    elif has_sab:
        return "SAB"
    elif has_dom_col:
        return "DOM"
    elif "DIAS" in joined or "BUSINESS" in joined:
        return "DU"
    
    return "unknown"


def _extract_service_day_from_line(line: str) -> Optional[str]:
    """Extract service-day marker from a standalone header-like line."""
    upper = line.upper()

    has_sab = "SAB" in upper or "SÁBADO" in upper or "SATURDAY" in upper
    has_dom = "DOM" in upper or "SUNDAY" in upper or "HOLIDAY" in upper or "FERIADO" in upper
    has_du = "DU" in upper or "DIAS ÚTEIS" in upper or "BUSINESS DAYS" in upper

    if has_sab and has_dom:
        return "SAB+DOM"
    if has_du:
        return "DU"
    if has_sab:
        return "SAB"
    if has_dom:
        return "DOM"
    return None


def _clean(text: str) -> str:
    return text.replace("\xa0", " ").rstrip()


def _split_stop_and_times(line: str) -> Optional[Tuple[str, List[str]]]:
    matches = list(TIME_RE.finditer(line))
    if not matches:
        return None
    times = TIME_RE.findall(line)
    if ARROW_RE.match(line) or re.search(r"\s{2,}", line):
        parts = [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]
        time_part_index = next((index for index, part in enumerate(parts) if TIME_RE.search(part)), None)
        if time_part_index is not None:
            stop_name = parts[time_part_index - 1] if time_part_index > 0 else ""
        else:
            stop_name = line[: matches[0].start()].strip()
    else:
        first = matches[0]
        stop_name = line[: first.start()].strip()
    return stop_name, times


def _looks_like_note(line: str) -> bool:
    upper = line.upper()
    return "|" in line or upper.startswith("PARAGENS") or upper.startswith("HORÁRIO") or upper.startswith("ESTE HORÁRIO")


def _transpose_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    max_len = max(len(row["times"]) for row in rows)
    trips: List[Dict[str, Any]] = []
    for trip_index in range(max_len):
        trip_times = [row["times"][trip_index] if trip_index < len(row["times"]) else None for row in rows]
        if any(time_value is not None for time_value in trip_times):
            trips.append(
                {
                    "trip_index": trip_index + 1,
                    "departure_time": trip_times[0],
                    "stop_times": trip_times,
                }
            )
    return trips


def _parse_rows_from_lines(lines: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    pending_stop_name = ""
    last_row: Optional[Dict[str, Any]] = None

    for raw_line in lines:
        line = _clean(raw_line)
        stripped = line.strip()
        if not stripped:
            continue
        if FOOTER_RE.search(stripped):
            continue

        parsed = _split_stop_and_times(stripped)
        if parsed is None:
            if last_row and stripped.startswith("("):
                last_row["stop_name"] = f"{last_row['stop_name']} {stripped}".strip()
                continue
            if ARROW_RE.match(stripped):
                context_lines.append(stripped)
                continue
            if _looks_like_note(stripped):
                context_lines.append(stripped)
                continue
            if len(stripped) <= 80 and not any(ch.isdigit() for ch in stripped):
                pending_stop_name = f"{pending_stop_name} {stripped}".strip() if pending_stop_name else stripped
            else:
                context_lines.append(stripped)
            continue

        stop_name, times = parsed
        if pending_stop_name:
            stop_name = f"{pending_stop_name} {stop_name}".strip()
            pending_stop_name = ""
        if not stop_name and last_row is not None:
            stop_name = last_row["stop_name"]
        row = {
            "stop_name": stop_name,
            "times": times,
            "raw_line": stripped,
        }
        rows.append(row)
        last_row = row

    return rows, context_lines


def _build_block(base: Dict[str, Any], rows: List[Dict[str, Any]], context_lines: List[str]) -> Dict[str, Any]:
    stop_sequence = [row["stop_name"] for row in rows]
    service_day = _detect_service_day(base.get("header_lines", []))
    return {
        **base,
        "service_day": service_day,
        "stop_sequence": stop_sequence,
        "rows": rows,
        "trips": _transpose_rows(rows),
    }


def _parse_ut234_schedule_text(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    preamble: List[str] = []
    body_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current, body_lines
        if current is None:
            body_lines = []
            return
        rows, context_lines = _parse_rows_from_lines(body_lines)
        if rows:
            blocks.append(_build_block(current, rows, context_lines))
        current = None
        body_lines = []

    for raw_line in lines:
        line = _clean(raw_line)
        stripped = line.strip()
        if not stripped:
            continue
        if FOOTER_RE.search(stripped):
            continue

        header_match = UT234_HEADER_RE.match(stripped)
        if header_match:
            flush_current()
            current = {
                "layout": "ut234",
                "service_label": (header_match.group("label") or "").strip(),
                "header_lines": preamble[-4:],
                "direction_label": None,
            }
            preamble = []
            body_lines = []
            continue

        if current is None:
            preamble.append(stripped)
            continue

        body_lines.append(stripped)

    flush_current()

    return {
        "layout": "ut234",
        "blocks": blocks,
    }


def _parse_directional_schedule_text(text: str, layout_name: str) -> Dict[str, Any]:
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    body_lines: List[str] = []
    preamble: List[str] = []

    def flush_current() -> None:
        nonlocal current, body_lines
        if current is None:
            body_lines = []
            return
        rows, context_lines = _parse_rows_from_lines(body_lines)
        if rows:
            blocks.append(_build_block(current, rows, context_lines))
        current = None
        body_lines = []

    for raw_line in lines:
        line = _clean(raw_line)
        stripped = line.strip()
        if not stripped:
            continue
        if FOOTER_RE.search(stripped):
            continue

        direction_match = DIR_RE.search(stripped)
        if direction_match and not TIME_RE.search(stripped):
            marker = direction_match.group(1).upper()
            if marker in {"IDA", "VOLTA"}:
                flush_current()
                current = {
                    "layout": layout_name,
                    "direction_label": marker,
                    "header_lines": preamble[-4:] + [stripped],
                }
                preamble = []
                body_lines = []
                continue
            if current is not None:
                current.setdefault("header_lines", []).append(stripped)
                continue

        if current is None:
            preamble.append(stripped)
            continue

        body_lines.append(stripped)

    flush_current()

    return {
        "layout": layout_name,
        "blocks": blocks,
    }


def _parse_generic_schedule_text(text: str, layout_name: str) -> Dict[str, Any]:
    """Fallback parser for matrix-like layouts without explicit direction markers.

    This captures schedules that are text-based but do not contain the standard
    UT-specific markers used by the primary parsers.
    """
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    body_lines: List[str] = []
    preamble: List[str] = []

    def flush_current() -> None:
        nonlocal current, body_lines
        if current is None:
            body_lines = []
            return
        rows, context_lines = _parse_rows_from_lines(body_lines)
        if rows:
            block = _build_block(current, rows, context_lines)
            forced_day = current.get("forced_service_day")
            if forced_day:
                block["service_day"] = forced_day
            block.pop("forced_service_day", None)
            blocks.append(block)
        current = None
        body_lines = []

    for raw_line in lines:
        line = _clean(raw_line)
        stripped = line.strip()
        if not stripped:
            continue
        if FOOTER_RE.search(stripped):
            continue

        # Ignore footnote legend rows as service separators (e.g. "1 | Escolar ...").
        is_legend_row = bool(re.match(r"^\d+\s*\|", stripped))
        service_hint = _extract_service_day_from_line(stripped)
        if service_hint and SERVICE_SPLIT_RE.search(stripped) and not TIME_RE.search(stripped) and not is_legend_row:
            flush_current()
            current = {
                "layout": layout_name,
                "direction_label": None,
                "service_label": None,
                "header_lines": preamble[-4:] + [stripped],
                "forced_service_day": service_hint,
            }
            preamble = []
            body_lines = []
            continue

        if current is None:
            if TIME_RE.search(stripped):
                current = {
                    "layout": layout_name,
                    "direction_label": None,
                    "service_label": None,
                    "header_lines": preamble[-4:],
                }
            else:
                preamble.append(stripped)
                continue

        body_lines.append(stripped)

    flush_current()

    return {
        "layout": layout_name,
        "blocks": blocks,
    }


def parse_schedule_document(schedule_doc: Dict[str, Any]) -> Dict[str, Any]:
    pages = schedule_doc.get("pages", [])
    full_text = "\n".join(page.get("text", "") for page in pages if page.get("text"))
    ut = schedule_doc.get("ut", "")
    if ut in {"UT2", "UT3", "UT4"}:
        parsed = _parse_ut234_schedule_text(full_text)
        if not parsed.get("blocks"):
            parsed = _parse_generic_schedule_text(full_text, "ut234")
    elif ut == "UT1":
        parsed = _parse_directional_schedule_text(full_text, "ut1")
        if not parsed.get("blocks"):
            parsed = _parse_generic_schedule_text(full_text, "ut1")
    elif ut == "UT5":
        parsed = _parse_directional_schedule_text(full_text, "ut5")
        if not parsed.get("blocks"):
            parsed = _parse_generic_schedule_text(full_text, "ut5")
    else:
        parsed = _parse_directional_schedule_text(full_text, "unknown")
        if not parsed.get("blocks"):
            parsed = _parse_generic_schedule_text(full_text, "unknown")

    # If any page was produced by OCR, tag the layout with +ocr so downstream
    # consumers know this schedule was OCR-derived.
    if any(page.get("ocr") for page in pages):
        layout = parsed.get("layout") or "unknown"
        if not layout.endswith("+ocr"):
            parsed["layout"] = f"{layout}+ocr"

    return {
        **schedule_doc,
        "parsed": parsed,
    }



def load_and_parse_schedules(
    metadata_path: str, base_dir: str = ".", limit: Optional[int] = None, skip_empty: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Load and parse schedules.
    
    Args:
        metadata_path: Path to schedules_metadata.json
        base_dir: Base directory for resolving local_path
        limit: Max schedules to load
        skip_empty: If True, skip PDFs with no extracted text (image-based)
    
    Returns:
        (parsed_schedules, skipped_info) where skipped_info is a list of dicts with codamp, reason
    """
    schedules = load_all_schedules_from_metadata(metadata_path, base_dir=base_dir, limit=limit)
    parsed = []
    skipped = []
    
    for schedule_doc in schedules:
        pages = schedule_doc.get("pages", [])
        if not pages or not any(page.get("text", "").strip() for page in pages):
            skipped.append({"codamp": schedule_doc.get("codamp"), "reason": "no_text_extracted"})
            if skip_empty:
                continue
        
        parsed_doc = parse_schedule_document(schedule_doc)
        if parsed_doc["parsed"]["blocks"]:
            parsed.append(parsed_doc)
        else:
            skipped.append({"codamp": schedule_doc.get("codamp"), "reason": "no_blocks_parsed"})
    
    return parsed, skipped


def summarize_parsed_schedules(parsed_schedules: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for schedule in parsed_schedules:
        parsed = schedule.get("parsed", {})
        blocks = parsed.get("blocks", [])
        summary.append(
            {
                "codamp": schedule.get("codamp"),
                "ut": schedule.get("ut"),
                "layout": parsed.get("layout"),
                "blocks": len(blocks),
                "trips": sum(len(block.get("trips", [])) for block in blocks),
                "rows": sum(len(block.get("rows", [])) for block in blocks),
            }
        )
    return summary


def to_schedule_format(parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert to a lean schedule-only format, suitable for GTFS generation.
    Removes full page text, context lines, and raw_line fields.
    """
    parsed = parsed_doc.get("parsed", {})
    blocks = parsed.get("blocks", [])

    lean_blocks = []
    # helper to infer service_day from service_label when missing
    SERVICE_LABEL_MAP = {
        "du": "DU",
        "dias uteis": "DU",
        "dias úteis": "DU",
        "sab": "SAB",
        "sáb": "SAB",
        "sábado": "SAB",
        "sabado": "SAB",
        "dom": "DOM",
        "domingo": "DOM",
        "domingos": "DOM",
        "domingos e feriados": "SAB+DOM",
        "sabado e domingo": "SAB+DOM",
        "fim de semana": "SAB+DOM",
    }

    def infer_service_day(service_day: Optional[str], service_label: Optional[str]) -> str:
        if service_day and isinstance(service_day, str) and service_day.strip() and service_day.strip().lower() != "unknown":
            return service_day
        if not service_label:
            return "unknown"
        key = service_label.strip().lower()
        for k, v in SERVICE_LABEL_MAP.items():
            if k in key:
                return v
        return "unknown"

    for block in blocks:
        # build lean rows (keep times as-is for now)
        lean_rows = [
            {"stop_name": row["stop_name"], "times": row["times"]}
            for row in block.get("rows", [])
        ]

        # Prepare trips: ensure stop_times have no nulls and departure_time is set
        trips = []
        for trip in block.get("trips", []):
            stop_times = list(trip.get("stop_times", []))
            # replace None with explicit SKIP marker
            for i, t in enumerate(stop_times):
                if t is None:
                    stop_times[i] = "SKIP"

            # derive departure_time if missing by finding first real time
            departure_time = trip.get("departure_time")
            if departure_time is None:
                first_time = next((t for t in stop_times if t and t != "SKIP"), None)
                departure_time = first_time

            trips.append({
                "trip_index": trip.get("trip_index"),
                "departure_time": departure_time,
                "stop_times": stop_times,
            })

        lean_block = {
            "service_day": infer_service_day(block.get("service_day"), block.get("service_label")),
            "direction_label": block.get("direction_label") if block.get("direction_label") is not None else "Circular",
            "service_label": block.get("service_label"),
            "stop_sequence": block.get("stop_sequence", []),
            "rows": lean_rows,
            "trips": trips,
        }
        lean_blocks.append(lean_block)

    return {
        "codamp": parsed_doc.get("codamp"),
        "idcarr": parsed_doc.get("idcarr") or parsed_doc.get("idcar"),
        "geo_line_id": parsed_doc.get("geo_line_id"),
        "geo_idop": parsed_doc.get("geo_idop"),
        "ut": parsed_doc.get("ut"),
        "designa": parsed_doc.get("designa"),
        "layout": parsed.get("layout"),
        "route_geojson": parsed_doc.get("route_geojson"),
        "blocks": lean_blocks,
    }


def _resolve_local_path(local_path: str, metadata_path: str, base_dir: str = ".") -> Optional[str]:
    if not local_path:
        return None
    md_dir = os.path.dirname(metadata_path)
    bn = os.path.basename(local_path)
    candidates: List[str] = []
    if os.path.isabs(local_path):
        candidates.append(local_path)
    candidates.append(os.path.join(md_dir, local_path))
    parent_md = os.path.dirname(md_dir)
    if parent_md:
        candidates.append(os.path.join(parent_md, local_path))
    candidates.append(os.path.join(md_dir, bn))
    candidates.append(os.path.join(base_dir, local_path))
    candidates.append(os.path.join(base_dir, bn))
    candidates.append(local_path)

    for cand in candidates:
        norm = os.path.normpath(cand)
        if os.path.isfile(norm):
            return norm
    return None


def _build_metadata_index(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    meta = load_metadata(metadata_path)
    index: Dict[str, Dict[str, Any]] = {}
    for schedule in meta.get("schedules", []):
        codamp = schedule.get("codamp")
        if codamp:
            index[str(codamp)] = schedule
    return index


def _fetch_line_geojson(line_id: Any, timeout: int = 20) -> Optional[Dict[str, Any]]:
    if line_id in (None, ""):
        return None
    line_id = str(line_id)
    url = f"https://paragens.amp.pt/unirmap/getlinhas?idcarr={line_id}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="ignore")
        payload = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    features = payload.get("features")
    if not isinstance(features, list):
        return None

    # Keep GTFS-useful subset: geometry + minimal per-feature metadata.
    compact_features: List[Dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or "coordinates" not in geometry:
            continue
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        kept_props = {
            key: props[key]
            for key in (
                "id",
                "idlinha",
                "linha",
                "sentido",
                "designacao",
                "nome",
                "codigo",
            )
            if key in props
        }
        compact_features.append({"type": "Feature", "geometry": geometry, "properties": kept_props})

    if not compact_features:
        return None

    return {
        "source": "https://paragens.amp.pt/unirmap/getlinhas",
        "geo_line_id": line_id,
        "type": "FeatureCollection",
        "features": compact_features,
        "bbox": payload.get("bbox"),
    }


def enrich_existing_schedule_json(
    schedule_json_path: str,
    metadata_path: str,
    base_dir: str = ".",
    output_path: Optional[str] = None,
) -> Dict[str, int]:
    with open(schedule_json_path, "r", encoding="utf-8") as fh:
        current = json.load(fh)

    metadata_index = _build_metadata_index(metadata_path)
    needs_reparse = {
        str(item.get("codamp", ""))
        for item in current
        if not item.get("blocks") and metadata_index.get(str(item.get("codamp", "")))
    }
    parsed_lookup: Dict[str, Dict[str, Any]] = {}
    if needs_reparse:
        parsed_docs, _ = load_and_parse_schedules(metadata_path, base_dir=base_dir, skip_empty=False)
        parsed_lookup = {
            str(doc.get("codamp")): doc
            for doc in parsed_docs
            if str(doc.get("codamp")) in needs_reparse
        }

    reparsed = 0
    geo_added = 0
    geo_failed = 0
    id_backfilled = 0

    for idx, item in enumerate(current):
        codamp = str(item.get("codamp", ""))
        meta_item = metadata_index.get(codamp)

        if meta_item and not item.get("idcarr"):
            item["idcarr"] = meta_item.get("idcarr") or meta_item.get("idcar")
            if item.get("idcarr"):
                id_backfilled += 1
        if meta_item and not item.get("geo_line_id"):
            item["geo_line_id"] = meta_item.get("geo_line_id")
            item["geo_idop"] = meta_item.get("geo_idop")

        # Reparse only if schedule is missing parsed blocks.
        if not item.get("blocks") and meta_item and meta_item.get("download_status") == "success":
            doc = parsed_lookup.get(codamp)
            if doc and doc.get("parsed", {}).get("blocks"):
                enriched_doc = {
                    **doc,
                    "route_geojson": item.get("route_geojson"),
                }
                current[idx] = to_schedule_format(enriched_doc)
                reparsed += 1
                item = current[idx]

        # Add geojson if missing.
        if not item.get("route_geojson"):
            line_id = item.get("geo_line_id") or item.get("idcarr")
            route_geojson = _fetch_line_geojson(line_id)
            if route_geojson:
                item["route_geojson"] = route_geojson
                geo_added += 1
            else:
                geo_failed += 1

    out_path = output_path or schedule_json_path
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, ensure_ascii=False, indent=2)

    return {
        "total": len(current),
        "id_backfilled": id_backfilled,
        "reparsed": reparsed,
        "geo_added": geo_added,
        "geo_failed": geo_failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse AMP schedule PDFs into structured timetable data.")
    parser.add_argument("metadata", help="Path to schedules_metadata.json")
    parser.add_argument("--base-dir", default=".", help="Base directory used to resolve local_path entries")
    parser.add_argument("--limit", type=int, default=None, help="Limit how many schedules to load")
    parser.add_argument("--output", help="Optional JSON file to write the parsed schedules to")
    parser.add_argument(
        "--format",
        choices=["complete", "schedule"],
        default="schedule",
        help="Output format: 'complete' includes full page text and metadata; 'schedule' is lean format for GTFS",
    )
    parser.add_argument("--summary-only", action="store_true", help="Print a short summary instead of full JSON")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep PDFs with no extracted text (image-based); they will have empty blocks",
    )
    parser.add_argument(
        "--enrich-existing",
        help=(
            "Path to an existing schedule JSON file to enrich: reparses entries with empty blocks "
            "and fills missing route_geojson from line API by idcarr."
        ),
    )
    args = parser.parse_args()

    if args.enrich_existing:
        stats = enrich_existing_schedule_json(
            schedule_json_path=args.enrich_existing,
            metadata_path=args.metadata,
            base_dir=args.base_dir,
            output_path=args.output,
        )
        print(json.dumps({"enriched": stats}, ensure_ascii=False, indent=2))
        return

    parsed_schedules, skipped = load_and_parse_schedules(
        args.metadata, base_dir=args.base_dir, limit=args.limit, skip_empty=not args.keep_empty
    )

    if args.summary_only:
        summary = summarize_parsed_schedules(parsed_schedules)
        summary_obj = {"parsed": summary, "skipped": skipped}
        print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
    else:
        if args.format == "schedule":
            output = [to_schedule_format(schedule) for schedule in parsed_schedules]
        else:
            output = parsed_schedules
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            if args.summary_only:
                summary = summarize_parsed_schedules(parsed_schedules)
                json.dump({"parsed": summary, "skipped": skipped}, fh, ensure_ascii=False, indent=2)
            elif args.format == "schedule":
                json.dump([to_schedule_format(s) for s in parsed_schedules], fh, ensure_ascii=False, indent=2)
            else:
                json.dump(parsed_schedules, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
