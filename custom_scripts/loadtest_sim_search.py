#!/usr/bin/env python3
"""
HTTP load test for the personalized similarity-search app.

The script simulates authenticated users performing the same flow as the UI:
1. login
2. text search
3. save positive/negative feedback on returned images
4. apply feedback to learn/update the matrix
5. rerun the query
6. optionally fetch the progressive full rerun

It uses only the Python standard library so it can run wherever the app does.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.cookiejar
import json
import random
import re
import ssl
import statistics
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_QUERIES = [
    "dog",
    "cat",
    "car",
    "person",
    "football",
    "beach",
    "mountain",
    "city",
    "fashion",
    "food",
    "flower",
    "bird",
    "horse",
    "baby",
    "family",
    "winter",
    "summer",
    "concert",
    "airplane",
    "building",
]

SESSION_ID_RE = re.compile(r"const sessionId = '([^']+)';")


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        raise ValueError("base_url is required")
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url


def build_url(base_url: str, relative_path: str) -> str:
    return urllib.parse.urljoin(base_url, relative_path)


def load_queries(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_QUERIES)
    query_path = Path(path)
    queries = [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise ValueError(f"No queries found in {query_path}")
    return queries


@dataclass
class HttpResult:
    status: int
    text: str
    url: str
    headers: Dict[str, str]

    def json(self) -> Dict:
        return json.loads(self.text)


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self.successes: Dict[str, int] = defaultdict(int)
        self.failures: Dict[str, int] = defaultdict(int)
        self.error_samples: Dict[str, List[str]] = defaultdict(list)
        self.user_rounds_ok = 0
        self.user_rounds_failed = 0

    def record(self, name: str, elapsed_ms: float, ok: bool, error: Optional[str] = None) -> None:
        with self._lock:
            self.latencies[name].append(elapsed_ms)
            if ok:
                self.successes[name] += 1
            else:
                self.failures[name] += 1
                if error and len(self.error_samples[name]) < 5:
                    self.error_samples[name].append(error)

    def record_round(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.user_rounds_ok += 1
            else:
                self.user_rounds_failed += 1

    def summary(self) -> Dict:
        data = {
            "rounds_ok": self.user_rounds_ok,
            "rounds_failed": self.user_rounds_failed,
            "steps": {},
        }
        for name in sorted(set(self.latencies) | set(self.successes) | set(self.failures)):
            values = self.latencies.get(name, [])
            data["steps"][name] = {
                "count": len(values),
                "ok": self.successes.get(name, 0),
                "failed": self.failures.get(name, 0),
                "avg_ms": round(sum(values) / len(values), 2) if values else 0.0,
                "p50_ms": round(percentile(values, 0.50), 2) if values else 0.0,
                "p95_ms": round(percentile(values, 0.95), 2) if values else 0.0,
                "max_ms": round(max(values), 2) if values else 0.0,
                "errors": list(self.error_samples.get(name, [])),
            }
        return data


class SimSearchHttpClient:
    def __init__(self, base_url: str, timeout: float, insecure: bool = False, verbose: bool = False) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.verbose = verbose
        self.cookie_jar = http.cookiejar.CookieJar()
        handlers: List[urllib.request.BaseHandler] = [urllib.request.HTTPCookieProcessor(self.cookie_jar)]
        if self.base_url.startswith("https://"):
            context = ssl.create_default_context()
            if insecure:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self.opener = urllib.request.build_opener(*handlers)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _cookie(self, name: str) -> Optional[str]:
        for cookie in self.cookie_jar:
            if cookie.name == name:
                return cookie.value
        return None

    def _request(
        self,
        method: str,
        relative_path: str,
        data: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        referer: Optional[str] = None,
    ) -> HttpResult:
        url = build_url(self.base_url, relative_path)
        request_headers = {
            "User-Agent": "sim-search-loadtest/1.0",
            "Accept": "application/json,text/html,application/xhtml+xml,*/*",
        }
        if headers:
            request_headers.update(headers)

        payload = None
        if data is not None:
            payload_data = dict(data)
            csrf_token = self._cookie("csrftoken")
            if csrf_token:
                payload_data.setdefault("csrfmiddlewaretoken", csrf_token)
                request_headers.setdefault("X-CSRFToken", csrf_token)
            request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            request_headers.setdefault("Referer", referer or self.base_url)
            payload = urllib.parse.urlencode(payload_data).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            headers=request_headers,
            method=method.upper(),
        )

        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return HttpResult(
                    status=response.getcode(),
                    text=body,
                    url=response.geturl(),
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return HttpResult(
                status=exc.code,
                text=body,
                url=exc.geturl(),
                headers=dict(exc.headers.items()),
            )

    def get(self, relative_path: str) -> HttpResult:
        return self._request("GET", relative_path)

    def post(self, relative_path: str, data: Dict[str, str], referer: Optional[str] = None) -> HttpResult:
        return self._request("POST", relative_path, data=data, referer=referer)

    def ensure_login(self, username: str, password: str, email: str) -> str:
        register_page = self.get("register/")
        if register_page.status != 200:
            raise RuntimeError(f"Failed to load register page: HTTP {register_page.status}")

        register_response = self.post(
            "register/",
            data={
                "username": username,
                "password": password,
                "password_confirm": password,
                "email": email,
            },
            referer=build_url(self.base_url, "register/"),
        )

        if SESSION_ID_RE.search(register_response.text):
            return "registered"
        if "Username already exists" in register_response.text:
            return self.login(username, password)
        if "Username and password are required" in register_response.text:
            raise RuntimeError("Registration failed: missing username/password")
        if "Passwords do not match" in register_response.text:
            raise RuntimeError("Registration failed: password mismatch")

        # Some deployments may not show messages after redirect. Confirm auth by loading the home page.
        home_response = self.get("")
        if SESSION_ID_RE.search(home_response.text):
            return "registered"
        return self.login(username, password)

    def login(self, username: str, password: str) -> str:
        login_page = self.get("login/")
        if login_page.status != 200:
            raise RuntimeError(f"Failed to load login page: HTTP {login_page.status}")

        login_response = self.post(
            "login/",
            data={
                "username": username,
                "password": password,
            },
            referer=build_url(self.base_url, "login/"),
        )
        if SESSION_ID_RE.search(login_response.text):
            return "logged_in"
        if "Invalid username or password" in login_response.text:
            raise RuntimeError(f"Login failed for {username}: invalid credentials")

        home_response = self.get("")
        if SESSION_ID_RE.search(home_response.text):
            return "logged_in"
        raise RuntimeError(f"Login failed for {username}: could not confirm authenticated session")

    def fetch_home(self) -> HttpResult:
        return self.get("")

    def text_search(self, query: str, metric: str, num_results: int) -> Dict:
        params = urllib.parse.urlencode({
            "text": query,
            "metric": metric,
            "num_results": str(num_results),
        })
        response = self.get(f"api/text-search/?{params}")
        if response.status != 200:
            raise RuntimeError(f"text_search failed: HTTP {response.status} body={response.text[:300]}")
        return response.json()

    def save_feedback(
        self,
        session_id: str,
        image_index: int,
        image_name: str,
        feedback_type: str,
        query_type: str,
        query_text: str,
        query_image_index: Optional[int] = None,
    ) -> Dict:
        data = {
            "image_index": str(image_index),
            "image_name": image_name,
            "feedback_type": feedback_type,
            "session_id": session_id,
            "query_type": query_type,
            "query_text": query_text,
        }
        if query_image_index is not None:
            data["query_image_index"] = str(query_image_index)
        response = self.post("api/feedback/save/", data=data, referer=self.base_url)
        if response.status != 200:
            raise RuntimeError(f"save_feedback failed: HTTP {response.status} body={response.text[:300]}")
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"save_feedback failed: {payload}")
        return payload

    def apply_feedback(self, session_id: str) -> Dict:
        response = self.post(
            "api/feedback/apply/",
            data={"session_id": session_id},
            referer=self.base_url,
        )
        if response.status != 200:
            raise RuntimeError(f"apply_feedback failed: HTTP {response.status} body={response.text[:300]}")
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"apply_feedback failed: {payload}")
        return payload

    def rerun_query(
        self,
        session_id: str,
        metric: str,
        num_results: int,
        total_flow_time_ms: Optional[float] = None,
        feedback_processing_matrix_save_ms: Optional[float] = None,
        model_learning_ms: Optional[float] = None,
        feedback_search_id: Optional[str] = None,
        background_only: bool = False,
        progressive_full: bool = False,
    ) -> Dict:
        data = {
            "session_id": session_id,
            "num_results": str(num_results),
            "metric": metric,
        }
        if total_flow_time_ms is not None:
            data["total_flow_time_ms"] = f"{total_flow_time_ms:.2f}"
        if feedback_processing_matrix_save_ms is not None:
            data["feedback_processing_matrix_save_ms"] = str(feedback_processing_matrix_save_ms)
        if model_learning_ms is not None:
            data["model_learning_ms"] = str(model_learning_ms)
        if feedback_search_id:
            data["feedback_search_id"] = feedback_search_id
        if background_only:
            data["background_only"] = "1"
        if progressive_full:
            data["progressive_full"] = "1"

        response = self.post("api/query/rerun/", data=data, referer=self.base_url)
        if response.status != 200:
            raise RuntimeError(f"rerun_query failed: HTTP {response.status} body={response.text[:300]}")
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"rerun_query failed: {payload}")
        return payload


def time_step(metrics: MetricsCollector, name: str, fn):
    start = now_ms()
    try:
        result = fn()
        metrics.record(name, now_ms() - start, ok=True)
        return result
    except Exception as exc:
        metrics.record(name, now_ms() - start, ok=False, error=str(exc))
        raise


def choose_feedback_targets(results: List[Dict], positive_count: int, negative_count: int) -> List[Tuple[Dict, str]]:
    if not results:
        return []
    positives = results[:positive_count]
    negatives = list(reversed(results[-negative_count:])) if negative_count else []
    seen = set()
    chosen: List[Tuple[Dict, str]] = []
    for item in positives:
        key = item.get("index")
        if key not in seen:
            chosen.append((item, "positive"))
            seen.add(key)
    for item in negatives:
        key = item.get("index")
        if key not in seen:
            chosen.append((item, "negative"))
            seen.add(key)
    return chosen


def prepare_users(
    args: argparse.Namespace,
    usernames: Iterable[str],
    metrics: MetricsCollector,
) -> None:
    if not args.prepare_users:
        return

    print(f"Preparing {len(list(usernames))} test users...", flush=True)
    for username in usernames:
        client = SimSearchHttpClient(args.base_url, timeout=args.timeout, insecure=args.insecure, verbose=False)
        email = f"{username}@loadtest.local"
        try:
            time_step(metrics, "prepare_user", lambda: client.ensure_login(username, args.password, email))
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare user {username}: {exc}") from exc


def run_virtual_user(
    user_number: int,
    args: argparse.Namespace,
    queries: List[str],
    metrics: MetricsCollector,
    start_barrier: threading.Barrier,
) -> Dict:
    rng = random.Random(args.seed + user_number)
    username = f"{args.username_prefix}{user_number:03d}"
    email = f"{username}@loadtest.local"
    client = SimSearchHttpClient(args.base_url, timeout=args.timeout, insecure=args.insecure, verbose=args.verbose)

    result_summary = {
        "username": username,
        "rounds_ok": 0,
        "rounds_failed": 0,
        "errors": [],
    }

    try:
        time_step(metrics, "login", lambda: client.login(username, args.password))
    except Exception as exc:
        result_summary["errors"].append(f"login failed: {exc}")
        return result_summary

    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        result_summary["errors"].append("start barrier broken")
        return result_summary

    for round_index in range(args.rounds):
        round_ok = False
        try:
            session_id = str(uuid.uuid4())
            query = rng.choice(queries)

            search_payload = time_step(
                metrics,
                "text_search",
                lambda: client.text_search(query=query, metric=args.metric, num_results=args.num_results),
            )
            search_results = search_payload.get("results", [])
            if len(search_results) < max(args.positive_feedback, args.negative_feedback, 2):
                raise RuntimeError(f"too few search results returned: {len(search_results)}")

            feedback_targets = choose_feedback_targets(
                search_results,
                positive_count=args.positive_feedback,
                negative_count=args.negative_feedback,
            )
            if not feedback_targets:
                raise RuntimeError("no feedback targets chosen")

            for item, feedback_type in feedback_targets:
                time_step(
                    metrics,
                    "save_feedback",
                    lambda item=item, feedback_type=feedback_type: client.save_feedback(
                        session_id=session_id,
                        image_index=int(item["index"]),
                        image_name=str(item["image_name"]),
                        feedback_type=feedback_type,
                        query_type="text",
                        query_text=query,
                    ),
                )

            feedback_started_at = now_ms()
            apply_payload = time_step(
                metrics,
                "apply_feedback",
                lambda: client.apply_feedback(session_id=session_id),
            )
            rerun_payload = time_step(
                metrics,
                "rerun_query",
                lambda: client.rerun_query(
                    session_id=session_id,
                    metric=args.metric,
                    num_results=args.num_results,
                    total_flow_time_ms=now_ms() - feedback_started_at,
                    feedback_processing_matrix_save_ms=float(apply_payload.get("feedback_processing_matrix_save_ms", 0.0)),
                    model_learning_ms=float(apply_payload.get("model_learning_ms", 0.0)),
                    feedback_search_id=apply_payload.get("feedback_search_id"),
                ),
            )

            if args.fetch_progressive_full and rerun_payload.get("results", {}).get("progressive_pending"):
                time_step(
                    metrics,
                    "rerun_query_full",
                    lambda: client.rerun_query(
                        session_id=session_id,
                        metric=args.metric,
                        num_results=args.num_results,
                        feedback_search_id=apply_payload.get("feedback_search_id"),
                        background_only=True,
                        progressive_full=True,
                    ),
                )

            round_ok = True
            result_summary["rounds_ok"] += 1
            metrics.record_round(ok=True)
        except Exception as exc:
            result_summary["rounds_failed"] += 1
            result_summary["errors"].append(f"round {round_index + 1}: {exc}")
            metrics.record_round(ok=False)
            if args.verbose:
                traceback.print_exc()
        finally:
            if args.think_time_max > 0:
                time.sleep(rng.uniform(args.think_time_min, args.think_time_max))

    return result_summary


def print_summary(summary: Dict, per_user: List[Dict], elapsed_s: float) -> None:
    print("\nLoad test summary")
    print("=================")
    print(f"Wall time: {elapsed_s:.2f}s")
    print(f"Rounds ok: {summary['rounds_ok']}")
    print(f"Rounds failed: {summary['rounds_failed']}")
    print("")
    print("Endpoint timings")
    print("----------------")
    for step_name, step_data in summary["steps"].items():
        print(
            f"{step_name:18} "
            f"count={step_data['count']:4d} "
            f"ok={step_data['ok']:4d} "
            f"fail={step_data['failed']:3d} "
            f"avg={step_data['avg_ms']:8.2f}ms "
            f"p50={step_data['p50_ms']:8.2f}ms "
            f"p95={step_data['p95_ms']:8.2f}ms "
            f"max={step_data['max_ms']:8.2f}ms"
        )
        for error in step_data.get("errors", []):
            print(f"  error sample: {error}")

    failing_users = [item for item in per_user if item["errors"]]
    if failing_users:
        print("")
        print("Per-user errors")
        print("---------------")
        for item in failing_users[:10]:
            print(f"{item['username']}:")
            for error in item["errors"][:5]:
                print(f"  - {error}")


def write_json_output(path: str, summary: Dict, per_user: List[Dict], elapsed_s: float, args: argparse.Namespace) -> None:
    output = {
        "base_url": args.base_url,
        "users": args.users,
        "rounds": args.rounds,
        "num_results": args.num_results,
        "metric": args.metric,
        "elapsed_seconds": round(elapsed_s, 3),
        "summary": summary,
        "users_detail": per_user,
    }
    Path(path).write_text(json.dumps(output, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load test the personalized similarity-search app.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8942/",
        help="Base public URL of the app, including any prefix, e.g. http://host/demos/personalized-similarity-search/",
    )
    parser.add_argument("--users", type=int, default=20, help="Number of concurrent virtual users.")
    parser.add_argument("--rounds", type=int, default=3, help="Search/feedback/rerun loops per user.")
    parser.add_argument("--num-results", type=int, default=20, help="Number of results requested per query.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="Search metric to exercise.")
    parser.add_argument("--positive-feedback", type=int, default=2, help="Positive labels to submit per round.")
    parser.add_argument("--negative-feedback", type=int, default=2, help="Negative labels to submit per round.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument("--username-prefix", default="loadtest_user_", help="Prefix for generated usernames.")
    parser.add_argument("--password", default="loadtest123", help="Password for all load-test users.")
    parser.add_argument("--prepare-users", action="store_true", default=True, help="Ensure test users exist before the timed run.")
    parser.add_argument("--no-prepare-users", action="store_false", dest="prepare_users", help="Skip user preparation and only log in.")
    parser.add_argument("--queries-file", help="Optional text file with one search query per line.")
    parser.add_argument("--think-time-min", type=float, default=0.1, help="Minimum pause between user rounds in seconds.")
    parser.add_argument("--think-time-max", type=float, default=0.5, help="Maximum pause between user rounds in seconds.")
    parser.add_argument("--fetch-progressive-full", action="store_true", default=True, help="Also fetch the background progressive full rerun when the server returns progressive_pending=true.")
    parser.add_argument("--no-fetch-progressive-full", action="store_false", dest="fetch_progressive_full", help="Skip the follow-up progressive full rerun fetch.")
    parser.add_argument("--output-json", help="Optional path to write machine-readable summary JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic query selection.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for HTTPS targets.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose per-user exceptions.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.base_url = normalize_base_url(args.base_url)
    if args.users <= 0:
        raise SystemExit("--users must be > 0")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be > 0")
    if args.think_time_min < 0 or args.think_time_max < args.think_time_min:
        raise SystemExit("Invalid think-time range")

    queries = load_queries(args.queries_file)
    metrics = MetricsCollector()
    usernames = [f"{args.username_prefix}{index:03d}" for index in range(1, args.users + 1)]

    prepare_users(args, usernames, metrics)

    print(
        f"Starting load test against {args.base_url} with {args.users} users, "
        f"{args.rounds} rounds each, metric={args.metric}, num_results={args.num_results}",
        flush=True,
    )

    start_barrier = threading.Barrier(args.users)
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as executor:
        futures = [
            executor.submit(run_virtual_user, idx, args, queries, metrics, start_barrier)
            for idx in range(1, args.users + 1)
        ]
        per_user = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed_s = time.perf_counter() - started_at

    summary = metrics.summary()
    print_summary(summary, per_user, elapsed_s)
    if args.output_json:
        write_json_output(args.output_json, summary, per_user, elapsed_s, args)
        print(f"\nWrote JSON summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
