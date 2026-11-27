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
            CREATE TABLE IF NOT EXISTS connection (
                tenant TEXT NOT NULL, source TEXT NOT NULL,
                generation INTEGER NOT NULL, connected INTEGER NOT NULL,
                mapping_version INTEGER NOT NULL,
                PRIMARY KEY (tenant, source)
            );
            CREATE TABLE IF NOT EXISTS accepted (
                tenant TEXT NOT NULL, source TEXT NOT NULL, period TEXT NOT NULL,
                assembly TEXT NOT NULL, digest TEXT NOT NULL, mapping_version INTEGER NOT NULL,
                report TEXT NOT NULL, accepted_at TEXT NOT NULL,
                PRIMARY KEY (tenant, source, period)
            );
            CREATE TABLE IF NOT EXISTS job_result (
                tenant TEXT NOT NULL, job_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, published_at TEXT,
                PRIMARY KEY (tenant, job_id)
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
            INSERT INTO connection VALUES (?, ?, 1, ?, 1)
            ON CONFLICT (tenant, source) DO UPDATE SET
                generation = connection.generation + 1,
                connected = excluded.connected
            RETURNING generation
        """, (tenant, source, connected))[0]

    def capture(self, tenant, source):
        row = self._query("""
            SELECT generation, mapping_version FROM connection
            WHERE tenant = ? AND source = ? AND connected = 1
        """, (tenant, source))
        return row

    def change_mapping(self, tenant, source):
        row = self._query("""
            UPDATE connection SET mapping_version = mapping_version + 1
            WHERE tenant = ? AND source = ? AND connected = 1
            RETURNING mapping_version
        """, (tenant, source))
        return row[0] if row else None

    def publish(self, *candidate):
        with closing(sqlite3.connect(self.path)) as db, db:
            return self._publish(db, *candidate)

    def publish_job(self, job_id, tenant, source, period, job, assembly, digest, report, accepted_at):
        # The stable job ID identifies the immutable input, not the retry time.
        fingerprint = json.dumps([source, period, job, assembly, digest])
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("""
                SELECT fingerprint, status, published_at FROM job_result
                WHERE tenant = ? AND job_id = ?
            """, (tenant, job_id)).fetchone()
            if previous:
                if previous[0] != fingerprint:
                    raise ValueError("The job ID is already bound to a different candidate.")
                return previous[1:]

            published = self._publish(db, tenant, source, period, job, assembly, digest, report, accepted_at)
            result = ("completed", accepted_at) if published else ("not_published", None)
            db.execute("INSERT INTO job_result VALUES (?, ?, ?, ?, ?)",
                       (tenant, job_id, fingerprint, *result))
            return result

    @staticmethod
    def _publish(db, tenant, source, period, job, assembly, digest, report, accepted_at):
        # Jobs carry a validated kept-versions UTC assembly ID (sortable) and
        # fingerprint. Current connection state and monotonic report revision
        # must be checked in the same atomic write across workers.
        generation, mapping_version = job
        return db.execute("""
            INSERT INTO accepted (tenant, source, period, assembly, digest, mapping_version, report, accepted_at)
            SELECT tenant, source, ?, ?, ?, mapping_version, ?, ? FROM connection
            WHERE tenant = ? AND source = ? AND connected = 1
              AND generation = ? AND mapping_version = ?
            ON CONFLICT (tenant, source, period) DO UPDATE SET
                assembly = excluded.assembly, digest = excluded.digest,
                mapping_version = excluded.mapping_version,
                report = excluded.report, accepted_at = excluded.accepted_at
            WHERE excluded.assembly > accepted.assembly
               OR (excluded.assembly = accepted.assembly AND excluded.digest = accepted.digest
                   AND excluded.mapping_version > accepted.mapping_version)
            RETURNING 1
        """, (period, assembly, digest, json.dumps(report), accepted_at,
              tenant, source, generation, mapping_version)).fetchone() is not None

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
        self.v1 = ("20260202T005500Z", "input-a")
        self.v2 = ("20260202T015500Z", "input-b")

    def test_disconnect_fences_a_validated_job_even_after_reconnect(self):
        self.store.connect(self.tenant, self.source)
        first_job = self.store.capture(self.tenant, self.source)
        self.assertTrue(self.store.publish(*self.scope, first_job, *self.v1, self.first, self.earlier))

        # The corrected job has already finished validation but not committed.
        corrected_job = self.store.capture(self.tenant, self.source)
        self.store.disconnect(self.tenant, self.source)
        self.assertIsNone(self.store.capture(self.tenant, self.source))
        self.assertFalse(self.store.publish(*self.scope, corrected_job, *self.v2, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))

        self.store.connect(self.tenant, self.source)
        new_job = self.store.capture(self.tenant, self.source)
        self.assertNotEqual(new_job, corrected_job)
        self.assertFalse(self.store.publish(*self.scope, corrected_job, *self.v2, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))
        self.assertTrue(self.store.publish(*self.scope, new_job, *self.v2, self.correction, self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.correction, self.later))

    def test_same_source_and_period_are_separate_for_two_tenants(self):
        self.store.connect(self.tenant, self.source)
        self.store.connect("tenant-b", self.source)
        job_a = self.store.capture(self.tenant, self.source)
        job_b = self.store.capture("tenant-b", self.source)
        self.assertTrue(self.store.publish(*self.scope, job_a, *self.v1, self.first, self.earlier))
        self.assertIsNone(self.store.latest("tenant-b", self.source, self.period))
        self.assertTrue(self.store.publish("tenant-b", self.source, self.period, job_b, *self.v2, self.correction, self.later))
        self.store.disconnect(self.tenant, self.source)
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))
        self.assertEqual(self.store.latest("tenant-b", self.source, self.period), (self.correction, self.later))
        self.assertIsNone(self.store.latest(self.tenant, self.source, "2026-02"))


    def test_late_older_assembly_and_exact_retry_do_not_move_accepted_time(self):
        self.store.connect(self.tenant, self.source)
        slow = self.store.capture(self.tenant, self.source)
        fast = self.store.capture(self.tenant, self.source)
        self.assertEqual(slow, fast)
        self.assertTrue(self.store.publish(*self.scope, fast, *self.v2, self.correction, self.later))
        late = "2026-02-02T03:00:00Z"
        self.assertFalse(self.store.publish(*self.scope, slow, *self.v1, self.first, late))
        self.assertFalse(self.store.publish(*self.scope, fast, *self.v2, self.correction, late))
        self.assertEqual(self.store.latest(*self.scope), (self.correction, self.later))

    def test_only_current_mapping_may_reprocess_exact_current_assembly(self):
        self.store.connect(self.tenant, self.source)
        old_mapping = self.store.capture(self.tenant, self.source)
        self.assertTrue(self.store.publish(*self.scope, old_mapping, *self.v2, self.correction, self.later))
        self.store.change_mapping(self.tenant, self.source)
        new_mapping = self.store.capture(self.tenant, self.source)
        self.assertEqual(new_mapping[0], old_mapping[0])
        self.assertGreater(new_mapping[1], old_mapping[1])
        after_change = "2026-02-02T03:00:00Z"
        remapped = {**self.correction, "unallocated": 35}
        v3 = ("20260202T025500Z", "input-c")
        self.assertFalse(self.store.publish(*self.scope, old_mapping, *v3, remapped, after_change))
        self.assertFalse(self.store.publish(*self.scope, new_mapping, *self.v1, self.first, after_change))
        self.assertFalse(self.store.publish(
            *self.scope, new_mapping, self.v2[0], "different-input", remapped, after_change
        ))
        self.assertEqual(self.store.latest(*self.scope), (self.correction, self.later))
        self.assertTrue(self.store.publish(*self.scope, new_mapping, *self.v2, remapped, after_change))
        self.assertEqual(self.store.latest(*self.scope), (remapped, after_change))

    def test_lost_success_response_replays_historical_job_without_regressing_latest(self):
        self.store.connect(self.tenant, self.source)
        job = self.store.capture(self.tenant, self.source)
        self.assertTrue(self.store.publish(*self.scope, job, *self.v1, self.first, self.earlier))

        def write_then_lose_response():
            self.store.publish_job("correction", *self.scope, job, *self.v2, self.correction, self.later)
            raise TimeoutError("worker lost the committed response")

        with self.assertRaises(TimeoutError):
            write_then_lose_response()
        restarted = PublicationStore(self.store.path)
        late = "2026-02-02T03:00:00Z"
        self.assertEqual(restarted.publish_job("correction", *self.scope, job, *self.v2, self.correction, late),
                         ("completed", self.later))
        self.assertEqual(restarted.latest(*self.scope), (self.correction, self.later))

        newer = {"total": 110, "currency": "USD"}
        v3 = ("20260202T025500Z", "input-c")
        self.assertEqual(restarted.publish_job("newer", *self.scope, job, *v3, newer, late), ("completed", late))
        self.assertEqual(restarted.publish_job("correction", *self.scope, job, *self.v2, self.correction, late),
                         ("completed", self.later))
        self.assertEqual(restarted.publish_job("stale", *self.scope, job, *self.v2, self.correction, late),
                         ("not_published", None))
        with self.assertRaisesRegex(ValueError, "already bound"):
            restarted.publish_job("correction", *self.scope, job, *v3, newer, late)
        self.assertEqual(restarted.latest(*self.scope), (newer, late))

    def test_job_result_write_failure_rolls_back_report_publication(self):
        self.store.connect(self.tenant, self.source)
        job = self.store.capture(self.tenant, self.source)
        self.assertTrue(self.store.publish(*self.scope, job, *self.v1, self.first, self.earlier))
        self.store._run("""
            CREATE TRIGGER fail_result BEFORE INSERT ON job_result
            BEGIN SELECT RAISE(ABORT, 'test write failure'); END;
        """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test write failure"):
            self.store.publish_job("correction", *self.scope, job, *self.v2, self.correction, self.later)
        self.assertEqual(self.store.latest(*self.scope), (self.first, self.earlier))
        self.store._run("DROP TRIGGER fail_result;")
        self.assertEqual(self.store.publish_job("correction", *self.scope, job, *self.v2, self.correction, self.later),
                         ("completed", self.later))
        self.assertEqual(self.store.latest(*self.scope), (self.correction, self.later))
