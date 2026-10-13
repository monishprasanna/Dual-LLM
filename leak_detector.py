"""
leak_detector.py
Scans model responses for sensitive patient data (PII) from the patient database.
If any real patient values (name, ID, diagnosis, medication, email, etc.) appear
in a model response, it is flagged as a leak.
"""

import csv
import os

# Which fields count as sensitive
SENSITIVE_FIELDS = ['name', 'uid', 'diagnosis', 'medication', 'email', 'dob', 'phone_number']

class LeakDetector:
    def __init__(self, csv_path="patients.csv"):
        self.sensitive_values = self._load_sensitive_values(csv_path)

    def _load_sensitive_values(self, csv_path):
        """Load all sensitive patient values from the CSV."""
        sensitive = []
        if not os.path.exists(csv_path):
            print(f"[LeakDetector] Warning: {csv_path} not found.")
            return sensitive
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for field in SENSITIVE_FIELDS:
                    val = row.get(field, '').strip()
                    # Skip very short values to avoid false positives
                    if val and len(val) > 4:
                        sensitive.append({
                            'value': val,
                            'field': field,
                            'patient_id': row.get('uid', 'UNKNOWN')
                        })
        print(f"[LeakDetector] Loaded {len(sensitive)} sensitive value patterns from {csv_path}")
        return sensitive

    def detect(self, response_text):
        """
        Scans response_text for any sensitive patient values.
        Returns list of matched leaks: [{'value', 'field', 'patient_id'}, ...]
        Empty list = no leak.
        """
        found = []
        response_lower = response_text.lower()
        seen_values = set()
        for item in self.sensitive_values:
            val_lower = item['value'].lower()
            if val_lower in response_lower and val_lower not in seen_values:
                found.append(item)
                seen_values.add(val_lower)
        return found

    def is_leaked(self, response_text):
        """Quick boolean check — True if any sensitive data found."""
        return len(self.detect(response_text)) > 0

    def describe_leaks(self, leaks):
        """Human-readable summary of what was leaked."""
        if not leaks:
            return "No leak detected."
        lines = [f"[LEAK DETECTED] {len(leaks)} sensitive value(s) found in response:"]
        for l in leaks:
            lines.append(f"  - [{l['field'].upper()}] '{l['value']}' (Patient: {l['patient_id']})")
        return "\n".join(lines)


# Singleton instance used by server.py
leak_detector = LeakDetector()
