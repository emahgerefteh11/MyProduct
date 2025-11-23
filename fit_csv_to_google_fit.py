#!/usr/bin/env python3
# CSV → Google Fit (Weight) Uploader (Windows-friendly, no fancy syntax)

import argparse, csv, datetime as dt, os, sys
from dataclasses import dataclass
from typing import List, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/fitness.body.write",
    "https://www.googleapis.com/auth/fitness.body.read",
]
DATA_TYPE_NAME = "com.google.weight"
DEFAULT_SOURCE_NAME = "CSV Importer"
DEFAULT_STREAM_NAME = "csv-weight"

@dataclass
class WeightPoint:
    ts_start_ns: int
    ts_end_ns: int
    kg: float

def to_kg(value, unit):
    unit = (unit or "").strip().lower()
    if unit in ("kg", "kgs", "kilogram", "kilograms"):
        return float(value)
    if unit in ("lb", "lbs", "pound", "pounds"):
        return float(value) * 0.45359237
    raise ValueError("Unsupported unit. Use 'lbs' or 'kg'.")

def _try_parse_dt(dt_str):
    for fmt in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M"):
        try:
            return dt.datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    return None

def parse_csv(path, tzname=None):
    tz = ZoneInfo(tzname) if (tzname and ZoneInfo) else None
    results: List[WeightPoint] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Date", "Time", "Data Type", "Value", "Unit"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV missing columns: %s" % ", ".join(sorted(missing)))

        for row in reader:
            if str(row.get("Data Type", "")).strip().lower() != "weight":
                continue
            dt_str = "%s %s" % (row["Date"], row["Time"])
            dt_obj = _try_parse_dt(dt_str)
            if not dt_obj:
                raise ValueError("Bad date/time: %r (use YYYY-MM-DD and HH:MM 24h)" % dt_str)

            if tz is not None:
                dt_local = dt_obj.replace(tzinfo=tz)
            else:
                # treat as local time
                dt_local = dt_obj.astimezone()

            ts_ns = int(dt_local.timestamp() * 1e9)
            kg = round(to_kg(row["Value"], row["Unit"]), 3)
            results.append(WeightPoint(ts_start_ns=ts_ns, ts_end_ns=ts_ns, kg=kg))

    if not results:
        raise ValueError("No 'Weight' rows found in CSV.")
    return results

def get_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("client_secret.json"):
                print("ERROR: client_secret.json missing.", file=sys.stderr)
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("fitness", "v1", credentials=creds, cache_discovery=False)

def ensure_data_source(service, source_name, stream_name):
    body = {
        "dataStreamName": stream_name,
        "type": "raw",
        "application": {"name": source_name, "version": "1"},
        "dataType": {"name": DATA_TYPE_NAME, "field": [{"name": "weight", "format": "floatPoint"}]},
    }
    try:
        created = service.users().dataSources().create(userId="me", body=body).execute()
        return created["dataStreamId"]
    except Exception:
        # Reuse existing
        sources = service.users().dataSources().list(userId="me").execute().get("dataSource", [])
        for s in sources:
            if s.get("dataType", {}).get("name") == DATA_TYPE_NAME:
                return s["dataStreamId"]
        raise

def write_points(service, data_source_id, points):
    start_ns = min(p.ts_start_ns for p in points)
    end_ns = max(p.ts_end_ns for p in points) + 1
    dataset_id = f"{start_ns}-{end_ns}"
    body = {
        "dataSourceId": data_source_id,
        "minStartTimeNs": start_ns,
        "maxEndTimeNs": end_ns,
        "point": [{
            "dataTypeName": DATA_TYPE_NAME,
            "startTimeNanos": p.ts_start_ns,
            "endTimeNanos": p.ts_end_ns,
            "value": [{"fpVal": p.kg}],
        } for p in points]
    }
    return service.users().dataSources().datasets().patch(
        userId="me", dataSourceId=data_source_id, datasetId=dataset_id, body=body
    ).execute()

def main():
    ap = argparse.ArgumentParser(description="Upload CSV weights into Google Fit")
    ap.add_argument("--csv", required=True, help="Path to CSV (Date,Time,Data Type,Value,Unit)")
    ap.add_argument("--tz", default=None, help="IANA tz, e.g. America/Los_Angeles")
    ap.add_argument("--dry-run", action="store_true", help="Preview only, no upload")
    args = ap.parse_args()

    points = parse_csv(args.csv, args.tz)
    print("Parsed %d points." % len(points))
    if args.dry_run:
        for p in points[:5]:
            ts = dt.datetime.fromtimestamp(p.ts_start_ns/1e9)
            print("%s -> %.3f kg" % (ts.isoformat(), p.kg))
        if len(points) > 5:
            print("... and %d more" % (len(points)-5))
        print("Dry run complete (no upload).")
        return

    service = get_service()
    ds_id = ensure_data_source(service, DEFAULT_SOURCE_NAME, DEFAULT_STREAM_NAME)
    print("Using dataSourceId:", ds_id)
    resp = write_points(service, ds_id, points)
    print("Upload complete.")
    if isinstance(resp, dict):
        print("Server response keys:", list(resp.keys()))

if __name__ == "__main__":
    main()
