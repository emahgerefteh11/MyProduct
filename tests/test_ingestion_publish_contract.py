"""Local executable contract for publishing connected billing reports.

SQLite makes the write guard concrete without choosing a production database,
adding a backend endpoint, or connecting to a live AWS account. Tenant and
connection IDs in this harness stand in for values resolved by the server.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


class PublicationStore:
    def __init__(self, path):
        self.path = path
        self._run("""
            CREATE TABLE connection (
                tenant TEXT NOT NULL, source TEXT NOT NULL,
                generation INTEGER NOT NULL, connected INTEGER NOT NULL,
                PRIMARY KEY (tenant, source)
            );
            CREATE TABLE accepted (
                tenant TEXT NOT NULL, source TEXT NOT NULL, period TEXT NOT NULL,
                report TEXT NOT NULL, accepted_at TEXT NOT NULL,
                PRIMARY KEY (tenant, source, period)
            );
        """)

    def _run(self, schema):
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript(schema)

    def _query(self, sql, params):
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute(sql, params).fetchone()

    def connect(self, tenant, source):
        return self._change(tenant, source, True)

    def disconnect(self, tenant, source):
        return self._change(tenant, source, False)

    def _change(self, tenant, source, connected):
        return self._query("""
            INSERT INTO connection VALUES (?, ?, 1, ?)
            ON CONFLICT (tenant, source) DO UPDATE SET
                generation = connection.generation + 1,
                connected = excluded.connected
            RETURNING generation
        """, (tenant, source, connected))[0]

    def capture(self, tenant, source):
        row = self._query("""
            SELECT generation FROM connection
            WHERE tenant = ? AND source = ? AND connected = 1
        """, (tenant, source))
        return row[0] if row else None

    def publish(self, tenant, source, period, generation, report, accepted_at):
        # This single write is the commit boundary. Checking the generation
        # before a separate write would leave a disconnect/publication race.
        return self._query("""
            INSERT INTO accepted (tenant, source, period, report, accepted_at)
            SELECT tenant, source, ?, ?, ? FROM connection
            WHERE tenant = ? AND source = ? AND connected = 1 AND generation = ?
            ON CONFLICT (tenant, source, period) DO UPDATE SET
                report = excluded.report, accepted_at = excluded.accepted_at
            RETURNING 1
        """, (period, json.dumps(report), accepted_at, tenant, source, generation)) is not None

    def latest(self, tenant, source, period):
        row = self._query("""
            SELECT report, accepted_at FROM accepted
            WHERE tenant = ? AND source = ? AND period = ?
        """, (tenant, source, period))
        return (json.loads(row[0]), row[1]) if row else None


class ConnectedPublishContractTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = PublicationStore(Path(temp.name) / "contract.sqlite")
        self.tenant, self.source, self.period = "tenant-a", "aws-cost", "2026-01"
        self.scope = self.tenant, self.source, self.period
        self.first = {"total": 100, "currency": "USD"}
        self.correction = {"total": 105, "currency": "USD"}
        self.earlier, self.later = "2026-02-02T01:00:00Z", "2026-02-02T02:00:00Z"

    def test_disconnect_fences_a_validated_job_even_after_reconnect(self):
        self.store.connect(self.tenant, self.source)
        generation = self.store.capture(self.tenant, self.source)
        self.assertTrue(self.store.publish(*self.scope, generation, self.first, self.earlier))

        # The corrected job has already finished validation but not committed.
        corrected_job = self.store.capture(self.tenant, self.source)
        self.store.disconnect(self.tenant, self.source)
        self.assertIsNone(self.store.capture(self.tenant, self.source))
        self.assertFalse(self.store.publish(*self.scope, corrected_job, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))

        new_generation = self.store.connect(self.tenant, self.source)
        self.assertNotEqual(new_generation, corrected_job)
        self.assertFalse(self.store.publish(*self.scope, corrected_job, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))
        self.assertTrue(self.store.publish(*self.scope, new_generation, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.correction, self.later))

    def test_same_source_and_period_are_separate_for_two_tenants(self):
        gen_a = self.store.connect(self.tenant, self.source)
        gen_b = self.store.connect("tenant-b", self.source)
        self.assertTrue(self.store.publish(*self.scope, gen_a, self.first, self.earlier))
        self.assertIsNone(self.store.latest("tenant-b", self.source, self.period))
        self.assertTrue(self.store.publish("tenant-b", self.source, self.period, gen_b, self.correction, self.later))
        self.store.disconnect(self.tenant, self.source)
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))
        self.assertEqual(self.store.latest("tenant-b", self.source, self.period), (self.correction, self.later))
        self.assertIsNone(self.store.latest(self.tenant, self.source, "2026-02"))
