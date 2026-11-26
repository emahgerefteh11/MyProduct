"""Loopback-only read contract; opaque fixture sessions stand in for real auth.

No login, cloud access or deployable service. The real backend must derive tenant
identity from its verified session and bind every job and report lookup to it.
"""

from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
import json
from pathlib import Path
import secrets
import tempfile
from threading import Thread
import unittest
from urllib.parse import urlsplit

from test_ingestion_publish_contract import PublicationStore


class ReadContractHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        scheme, _, token = self.headers.get("Authorization", "").partition(" ")
        session = self.server.sessions.get(token) if scheme == "Bearer" else None
        if not session or session["expires"] <= self.server.clock or session["revoked"]:
            return self.reply(401, {"error": "Sign-in required."})

        # Selectors are untrusted; they never supply the tenant. The fixture
        # returns the same result for another tenant's ID and an unknown ID.
        tenant = session["tenant"]
        parts = urlsplit(self.path).path.split("/")
        item = None
        if len(parts) == 4 and parts[1:3] == ["api", "jobs"]:
            item = self.server.jobs.get((tenant, parts[3]))
        elif len(parts) == 5 and parts[1:3] == ["api", "reports"]:
            snapshot = self.server.store.latest(tenant, parts[3], parts[4])
            if snapshot is not None:
                report, accepted_at = snapshot
                item = {"report": report, "lastSuccessfulSync": accepted_at}
        if item is None:
            return self.reply(404, {"error": "Not found."})
        return self.reply(200, item)


class TestIngestionReadContract(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ReadContractHandler)
        self.addCleanup(self.server.server_close)
        self.server.store = PublicationStore(Path(tmp.name) / "read.sqlite")
        self.server.sessions, self.server.jobs = {}, {}
        self.server.clock = 100
        self.a, self.b = self.seed("tenant-a", 105), self.seed("tenant-b", 50)
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)

    def seed(self, tenant, cost):
        token = secrets.token_urlsafe(32)
        self.server.sessions[token] = {"tenant": tenant, "expires": 110, "revoked": False}
        store = self.server.store
        store.connect(tenant, "aws-cost")
        job = store.capture(tenant, "aws-cost")
        self.assertTrue(store.publish(
            tenant, "aws-cost", "2026-01", job, "20260202T020000Z",
            "digest-" + tenant, {"total": cost, "currency": "USD"}, "2026-02-02T02:05:00Z"
        ))
        self.server.jobs[(tenant, "job-" + tenant)] = {"status": "completed"}
        return token

    def get(self, path, token=None, **headers):
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        with closing(HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)) as conn:
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            return response.status, json.loads(response.read())

    def test_report_uses_verified_session_for_same_source_and_period(self):
        path = "/api/reports/aws-cost/2026-01"
        status, own = self.get(path, self.a)
        self.assertEqual(status, 200)
        self.assertEqual(own["report"]["total"], 105)
        self.assertEqual(own["lastSuccessfulSync"], "2026-02-02T02:05:00Z")
        status, second = self.get(path, self.b, **{"X-Tenant": "tenant-a"})
        self.assertEqual(status, 200)
        self.assertEqual(second["report"]["total"], 50)
        self.assertEqual(self.get(path + "?tenant=tenant-a", self.b), (status, second))

    def test_job_lookup_cannot_distinguish_other_tenant_from_unknown(self):
        self.assertEqual(self.get("/api/jobs/job-tenant-a", self.a), (200, {"status": "completed"}))
        foreign = self.get("/api/jobs/job-tenant-a", self.b)
        self.assertEqual(foreign[0], 404)
        self.assertEqual(foreign, self.get("/api/jobs/absent", self.b))

    def test_expired_or_revoked_session_cannot_read_report_or_job(self):
        self.server.sessions[self.a]["revoked"] = True
        self.assertLess(self.server.clock, self.server.sessions[self.a]["expires"])
        report = "/api/reports/aws-cost/2026-01"
        for path_a, path_b in ((report, report), ("/api/jobs/job-tenant-a", "/api/jobs/job-tenant-b")):
            missing = self.get(path_a)
            self.assertEqual(missing[0], 401)
            self.assertEqual(self.get(path_a, self.a), missing)
            self.assertEqual(self.get(path_b, self.b)[0], 200)
        self.server.clock = self.server.sessions[self.b]["expires"]
        self.assertFalse(self.server.sessions[self.b]["revoked"])
        for path_b in (report, "/api/jobs/job-tenant-b"):
            self.assertEqual(self.get(path_b, self.b), self.get(path_b))
