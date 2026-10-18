"""Source validation and provenance for billing reports (standard library only)."""

import hashlib
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation


def numeric_value(value, path, line, column, optional=False):
    text = str(value or "").strip()
    if not text and optional:
        return Decimal(0)
    try:
        number = Decimal(text)
        if not number.is_finite() or not math.isfinite(float(number)):
            raise InvalidOperation
        return number
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError(f"{path.name}: row {line}: {column or 'numeric field'} must be a finite number.") from None


def iso_date(value):
    if not value:
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%m/%d/%Y")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class SourceAudit:
    def __init__(self, path, schema, assumed_currency=None):
        self.path = path
        self.schema = schema
        self.assumed_currency = (assumed_currency or "").strip().upper() or None
        if self.assumed_currency and not re.fullmatch(r"[A-Z]{3}", self.assumed_currency):
            raise ValueError("--currency must be a three-letter currency code, such as USD.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.rows = 0
        self.total = Decimal(0)
        self.currencies = set()
        self.missing_currency_rows = 0
        self.assumed_currency_rows = 0
        self.invalid_date_fields = 0
        self.invalid_tag_rows = 0
        self.dates = {key: None for key in ("usage_start", "usage_end", "billing_start", "billing_end")}

    def observe(self, row, cost, line):
        self.rows += 1
        self.total += cost
        currency = (row.get(self.schema.get("currency")) or "").strip().upper()
        if not currency:
            if self.assumed_currency:
                currency = self.assumed_currency
                self.assumed_currency_rows += 1
            else:
                self.missing_currency_rows += 1
        if currency:
            if not re.fullmatch(r"[A-Z]{3}", currency):
                raise ValueError(f"{self.path.name}: row {line}: invalid billing currency code.")
            self.currencies.add(currency)
        for key in self.dates:
            raw = row.get(self.schema.get(key)) or ""
            value = iso_date(raw)
            if raw and value is None:
                self.invalid_date_fields += 1
            if value:
                current = self.dates[key]
                self.dates[key] = (min if key.endswith("start") else max)(current, value) if current else value

    def finish(self):
        if not self.rows:
            raise ValueError(f"{self.path.name}: no billing rows found.")
        return {
            "file": self.path.name,
            "sha256": self.sha256,
            "schema": self.schema["schema"],
            "costColumn": self.schema["cost"],
            "costBasis": self.schema.get("cost_basis", "billed"),
            "rows": self.rows,
            "sourceCostTotal": str(self.total),
            "currencies": sorted(self.currencies),
            "missingCurrencyRows": self.missing_currency_rows,
            "assumedCurrencyRows": self.assumed_currency_rows,
            "invalidDateFields": self.invalid_date_fields,
            "invalidTagRows": self.invalid_tag_rows,
            "usagePeriod": {"start": self.dates["usage_start"], "end": self.dates["usage_end"]},
            "billingPeriod": {"start": self.dates["billing_start"], "end": self.dates["billing_end"]},
        }


def build_analysis(sources, total):
    currencies = {currency for source in sources for currency in source["currencies"]}
    missing = sum(source["missingCurrencyRows"] for source in sources)
    bases = {source["costBasis"] for source in sources}
    if len(currencies) > 1:
        raise ValueError("Cannot aggregate different billing currencies: " + ", ".join(sorted(currencies)) + ". Analyze them separately; FX conversion is not supported.")
    if currencies and missing:
        raise ValueError("Some billing rows have no currency. Specify --currency only if you know their currency; explicit source currencies are never overwritten.")
    if len(bases) > 1:
        raise ValueError("Cannot combine billed and effective costs in one report. Analyze each cost basis separately.")
    if not math.isfinite(total):
        raise ValueError("The aggregate cost is not finite. Check the input amounts.")
    source_total = sum((Decimal(source["sourceCostTotal"]) for source in sources), Decimal(0))
    warnings = []
    if missing:
        warnings.append("Currency is missing. Amounts are shown without a currency symbol; cross-file monetary comparability has not been verified.")
    if any(source["assumedCurrencyRows"] for source in sources):
        warnings.append("Missing currencies were filled using the explicit --currency assumption.")
    if any(source["invalidDateFields"] for source in sources):
        warnings.append("Some dates could not be parsed. The displayed period may be incomplete.")
    if any(source["invalidTagRows"] for source in sources):
        warnings.append("Some tag values could not be parsed. Review unallocated spend and naming fallbacks.")
    if any(not source["usagePeriod"]["start"] for source in sources):
        warnings.append("Usage start dates are unavailable for one or more sources.")
    periods = {}
    for key in ("usagePeriod", "billingPeriod"):
        starts = [source[key]["start"] for source in sources if source[key]["start"]]
        ends = [source[key]["end"] for source in sources if source[key]["end"]]
        periods[key] = {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
            "complete": len(starts) == len(sources) and len(ends) == len(sources),
        }
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "currency": next(iter(currencies), None),
        "costBasis": next(iter(bases), "billed"),
        "sourceCount": len(sources),
        "rowCount": sum(source["rows"] for source in sources),
        **periods,
        "sources": sources,
        "reconciliation": {
            "sourceTotal": float(source_total),
            "reportTotal": total,
            "difference": round(total - float(source_total), 2),
        },
        "warnings": warnings,
    }
