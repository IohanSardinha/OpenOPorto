# AMP Schedule PDF Parser — Usage Guide

## Quick Start

Parse PDFs into lean schedule format (no full page text):
```bash
cd PhysicalNetwork/OPorto
python3 unir_pdf_to_gtfs.py amp_schedules/schedules_metadata.json --format schedule
```

Save to file:
```bash
python3 unir_pdf_to_gtfs.py amp_schedules/schedules_metadata.json --format schedule --output schedules.json
```

## Output Formats

### `--format schedule` (default, recommended for GTFS)
Lean format with just essential schedule data (empty blocks are filtered out by default):
- `codamp`: route code
- `ut`: operator unit (UT1–UT5)
- `designa`: route description
- `layout`: parser layout (ut1, ut234, ut5)
- `blocks`: array of timetable sections (only non-empty blocks included)

Each block contains:
- `service_day`: "DU" (weekdays), "SAB" (Saturday), "DOM" (Sunday/holidays), or "SAB+DOM" (combined)
- `direction_label`: "IDA"/"VOLTA" (directional routes) or null
- `service_label`: service type annotation (if any)
- `stop_sequence`: list of stops in order
- `rows`: per-stop arrival times (raw data)
- `trips`: transposed trips with departure and stop times (GTFS-ready)

### `--format complete`
Full parsed document including all metadata and page text (for reference):
```bash
python3 unir_pdf_to_gtfs.py amp_schedules/schedules_metadata.json --format complete
```

### `--summary-only`
Quick overview with skipped schedules:
```bash
python3 unir_pdf_to_gtfs.py amp_schedules/schedules_metadata.json --summary-only
```

Returns:
```json
{
  "parsed": [...count summary...],
  "skipped": [
    {"codamp": "7322", "reason": "no_text_extracted"},
    {"codamp": "....", "reason": "no_blocks_parsed"}
  ]
}
```

## Filtering

By default, schedules with empty blocks are skipped:
- **Image-based PDFs** (no_text_extracted): PDFs are scanned/image-only and require OCR
- **Parse failed** (no_blocks_parsed): Text is present but the parser couldn't extract timetable blocks

To keep these empty schedules in the output (for debugging or later OCR processing):
```bash
python3 unir_pdf_to_gtfs.py amp_schedules/schedules_metadata.json --keep-empty
```

See `PARSING_REPORT.md` for a list of all skipped schedules and reasons.

## Service Day Handling

The parser detects and separates schedules by service type:

- **DU blocks**: Dias Úteis (weekdays) — clean, single-column schedules
- **SAB blocks**: Sábado (Saturday) only — separate schedule
- **DOM blocks**: Domingos e Feriados (Sunday and holidays) — separate schedule
- **SAB+DOM blocks**: Both Saturday and Sunday/holidays in one timetable with separate columns

For SAB+DOM blocks, the `times` array contains both column groups concatenated. You can separate them manually if needed by analyzing the time pattern or splitting at natural gaps.

## Example: Filter by Service Day

```python
import json
docs = json.load(open("schedules.json"))
for doc in docs:
    codamp = doc["codamp"]
    du_blocks = [b for b in doc["blocks"] if b["service_day"] == "DU"]
    weekend_blocks = [b for b in doc["blocks"] if "DOM" in b["service_day"] or "SAB" in b["service_day"]]
    print(f"Route {codamp}: {len(du_blocks)} weekday, {len(weekend_blocks)} weekend blocks")
```

## Example: Extract Stop Times for GTFS

```python
import json
docs = json.load(open("schedules.json"))
doc = docs[0]  # First route
block = doc["blocks"][0]  # First schedule block

# Get stop sequence
stops = block["stop_sequence"]
print("Stops:", stops)

# Get trips
for trip in block["trips"][:3]:
    print(f"Trip {trip['trip_index']}: departs {trip['departure_time']}")
    print(f"  Stop times: {trip['stop_times']}")
```

## Metadata in Blocks

- `header_lines`: First few lines from the PDF (for reference/debugging)
- `direction_label`: "IDA" or "VOLTA" (for directional routes), None for matrix-format tables

All blocks are ordered as they appear in the PDF, with service day markers clearly labeled.

---

**Parser Status**: All 5 UT formats (UT1, UT2, UT3, UT4, UT5) are supported with UT-specific handling.
