#!/usr/bin/env python3
"""
AMP Transit Schedule PDF Crawler
Downloads all PDF schedules from paragens.amp.pt
"""

import requests
import os
import time
import json
import csv
import re
from pathlib import Path
from datetime import datetime

# Configuration
API_URL = "https://paragens.amp.pt/acarto2/getcarreiras"
GEO_MAP_URL = "https://paragens.amp.pt/unirmap/getmun"
PDF_BASE_URL = "https://paragens.amp.pt/web/horarios_pdf/schedules"
OUTPUT_DIR = "amp_schedules"
UT_RANGE = range(1, 6)  # 1 to 5 inclusive
IDOPS = ["aro", "esp", "gon", "mai", "mat", "oaz", "prd", "prt", "pov", "smf", "str", "sjm", "trf", "vcm", "vlg", "vcd", "vng"]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://paragens.amp.pt/",
}

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def create_session():
    """Create a primed session.

    The AMP endpoints may return 500 for stateless/curl-like calls. Visiting
    the site first sets a session cookie and then API calls succeed.
    """
    session = requests.Session()
    try:
        session.get("https://paragens.amp.pt/", timeout=10, headers=REQUEST_HEADERS)
        session.get("https://paragens.amp.pt/unirmap/", timeout=10, headers=REQUEST_HEADERS)
    except requests.exceptions.RequestException:
        pass
    return session

def fetch_schedules(session, ut_num):
    """Fetch the schedule data from the API for a given UT ID"""
    ut_id = f"UT{ut_num}"
    url = f"{API_URL}?idop={ut_id}"
    print(f"\n📡 Fetching schedules for {ut_id}: {url}")
    
    try:
        response = session.get(url, timeout=10, headers=REQUEST_HEADERS)
        
        # Check for specific error responses
        if response.status_code == 403:
            print(f"❌ Access denied (403). The server may be blocking requests.")
            print(f"   Response: {response.text[:100]}")
            return None
        
        response.raise_for_status()
        
        # Try to parse JSON
        try:
            data = response.json()
            return data
        except ValueError as e:
            print(f"❌ Error parsing JSON response")
            print(f"   Response (first 200 chars): {response.text[:200]}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching schedules for {ut_id}: {e}")
        return None


def _norm_digits(value):
    if value is None:
        return None
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    return digits or text


def fetch_geo_map(session, idop):
    """Fetch mapping of codamp -> geo route gids for one idop."""
    url = f"{GEO_MAP_URL}?idop={idop}"
    try:
        response = session.get(url, timeout=12, headers=REQUEST_HEADERS)
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return {}

    if not isinstance(payload, list):
        return {}

    out = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        codamp = _norm_digits(item.get("codamp"))
        if not codamp:
            continue
        gid_ida = item.get("gid_ida")
        gid_volta = item.get("gid_volta")
        geo_line_id = gid_ida if gid_ida not in (None, "") else gid_volta
        out[str(codamp)] = {
            "geo_line_id": str(geo_line_id) if geo_line_id not in (None, "") else "",
            "geo_gid_ida": str(gid_ida) if gid_ida not in (None, "") else "",
            "geo_gid_volta": str(gid_volta) if gid_volta not in (None, "") else "",
            "geo_idop": idop,
        }
    return out


def build_geo_mappings(session):
    """Fetch all idops and build a global codamp->geo mapping."""
    combined = {}
    for idop in IDOPS:
        mapping = fetch_geo_map(session, idop)
        if not mapping:
            print(f"⚠️  Could not map geo gids for idop={idop}")
            continue
        combined.update(mapping)
        print(f"  ✅ idop={idop}: {len(mapping)} mapped lines")
    return combined

def parse_schedules(data):
    """Parse the JSON response and extract schedule information"""
    schedule_info = []
    
    if not data:
        return schedule_info
    
    # Data is a list of schedule items
    for item in data:
        if isinstance(item, dict):
            schedule_info.append({
                'idcarr': item.get('idcarr', ''),
                'codamp': str(item.get('codamp', '')),
                'ut': item.get('ut', ''),
                'tipo': item.get('tipo', ''),
                'designa': item.get('designa', ''),
                'mun': item.get('mun', ''),
            })
    
    return schedule_info

def download_pdf(session, ut_id, line_id, description):
    """Download a single PDF file"""
    pdf_url = f"{PDF_BASE_URL}/{ut_id}/{line_id}.pdf"
    
    # Create subdirectory for this UT
    ut_dir = Path(OUTPUT_DIR) / ut_id
    ut_dir.mkdir(exist_ok=True)
    
    # Sanitize filename
    safe_description = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"{line_id}_{safe_description[:50]}.pdf"
    filepath = ut_dir / filename
    
    # Skip if already downloaded
    if filepath.exists():
        print(f"  ⏭️  Skipping {line_id} (already exists)")
        return str(filepath), "skipped"
    
    try:
        print(f"  ⬇️  Downloading {line_id}: {description[:60]}...")
        response = session.get(pdf_url, timeout=15, headers=REQUEST_HEADERS)
        response.raise_for_status()
        
        # Save PDF
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✅ Saved: {filepath}")
        return str(filepath), "success"
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error downloading {pdf_url}: {e}")
        return None, "failed"

def save_metadata_json(all_schedules, filepath):
    """Save complete metadata to JSON file"""
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_schedules': len(all_schedules),
        'schedules': all_schedules
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved metadata to {filepath}")

def save_metadata_csv(all_schedules, filepath):
    """Save complete metadata to CSV file"""
    if not all_schedules:
        return
    
    # Get all unique fields
    fieldnames = ['idcarr', 'codamp', 'ut', 'tipo', 'designa', 'mun', 'geo_line_id', 'geo_gid_ida', 'geo_gid_volta', 'geo_idop',
                  'pdf_url', 'local_path', 'download_status']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_schedules)
    
    print(f"💾 Saved metadata to {filepath}")

def main():
    """Main crawler function"""
    print("=" * 70)
    print("🚌 AMP Transit Schedule PDF Crawler")
    print("=" * 70)
    
    total_downloaded = 0
    total_failed = 0
    total_skipped = 0
    all_schedules = []  # Store all metadata

    session = create_session()
    print("🔎 Building geo mapping from getmun endpoints...")
    geo_map_by_codamp = build_geo_mappings(session)
    print(f"   Total mapped codamp entries: {len(geo_map_by_codamp)}")
    
    # Loop through each UT ID
    for ut_num in UT_RANGE:
        print(f"\n{'='*70}")
        print(f"Processing UT{ut_num}")
        print(f"{'='*70}")
        
        # Fetch schedules from API
        data = fetch_schedules(session, ut_num)
        if not data:
            continue
        
        # Parse the JSON response
        schedules = parse_schedules(data)
        print(f"📋 Found {len(schedules)} schedules for UT{ut_num}")
        
        if not schedules:
            print(f"⚠️  No schedules found for UT{ut_num}")
            continue
        
        # Download each PDF
        for idx, schedule in enumerate(schedules, 1):
            print(f"\n[{idx}/{len(schedules)}]", end=" ")
            
            # Download the PDF
            filepath, status = download_pdf(
                session,
                schedule['ut'],
                schedule['codamp'],
                schedule['designa']
            )
            
            # Create complete metadata record
            pdf_url = f"{PDF_BASE_URL}/{schedule['ut']}/{schedule['codamp']}.pdf"
            metadata_record = {
                'idcarr': schedule['idcarr'],
                'codamp': schedule['codamp'],
                'ut': schedule['ut'],
                'tipo': schedule['tipo'],
                'designa': schedule['designa'],
                'mun': schedule['mun'],
                'geo_line_id': '',
                'geo_gid_ida': '',
                'geo_gid_volta': '',
                'geo_idop': '',
                'pdf_url': pdf_url,
                'local_path': filepath if filepath else '',
                'download_status': status
            }

            geo_info = geo_map_by_codamp.get(str(schedule['codamp']))
            if geo_info:
                metadata_record['geo_line_id'] = geo_info.get('geo_line_id', '')
                metadata_record['geo_gid_ida'] = geo_info.get('geo_gid_ida', '')
                metadata_record['geo_gid_volta'] = geo_info.get('geo_gid_volta', '')
                metadata_record['geo_idop'] = geo_info.get('geo_idop', '')
            all_schedules.append(metadata_record)
            
            # Update counters
            if status == "success":
                total_downloaded += 1
            elif status == "skipped":
                total_skipped += 1
            else:
                total_failed += 1
            
            # Be nice to the server - small delay between downloads
            time.sleep(0.5)
    
    # Save metadata files
    print("\n" + "=" * 70)
    print("💾 Saving metadata files...")
    print("=" * 70)
    
    json_path = Path(OUTPUT_DIR) / "schedules_metadata.json"
    csv_path = Path(OUTPUT_DIR) / "schedules_metadata.csv"
    
    save_metadata_json(all_schedules, json_path)
    save_metadata_csv(all_schedules, csv_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {total_downloaded} PDFs")
    print(f"⏭️  Skipped (already exists): {total_skipped} PDFs")
    print(f"❌ Failed downloads: {total_failed} PDFs")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"📄 Metadata JSON: {json_path}")
    print(f"📄 Metadata CSV: {csv_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()