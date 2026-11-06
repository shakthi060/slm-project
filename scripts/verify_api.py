"""Start the FastAPI service, call it over real HTTP, and save evidence.

The server is started as a separate process using the local model configured in
configs/api_config.yaml. This script records actual status codes, bodies, and
latencies under experiments/deployment/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.config import load_settings  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "deployment"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_server(client: httpx.Client, base_url: str, timeout_s: int = 300) -> dict:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                return response.json()
            last_error = f"health returned {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def record(
    client: httpx.Client,
    base_url: str,
    method: str,
    path: str,
    **kwargs,
) -> dict:
    started = time.perf_counter()
    try:
        response = client.request(method, f"{base_url}{path}", **kwargs)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            body = response.json()
        except ValueError:
            body = response.text[:4000]
        return {
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "body": body,
        }
    except Exception as exc:
        return {
            "method": method,
            "path": path,
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_markdown(path: Path, evidence: dict) -> None:
    lines = [
        "# API Verification Evidence",
        "",
        f"Generated: {evidence['generated_at']}",
        f"Base URL: `{evidence['base_url']}`",
        "",
        "## Startup health",
        "",
        "```json",
        json.dumps(evidence["startup_health"], indent=2),
        "```",
        "",
        "## HTTP calls",
        "",
        "| Method | Path | Status | Elapsed ms |",
        "|---|---|---:|---:|",
    ]
    for call in evidence["requests"]:
        lines.append(
            f"| {call['method']} | {call['path']} | {call['status_code']} | {call['elapsed_ms']} |"
        )
    lines += ["", "## Successful chat results", ""]
    chats = [
        call
        for call in evidence["requests"]
        if call["method"] == "POST"
        and call["path"] == "/chat"
        and call["status_code"] == 200
    ]
    if chats:
        for index, chat in enumerate(chats, start=1):
            body = chat["body"]
            lines += [
                f"### Chat {index}",
                "",
                f"Fallback used: `{body.get('fallback_used')}` "
                f"({body.get('fallback_reason')})",
                "",
                "```json",
                json.dumps(body, indent=2),
                "```",
                "",
            ]
    else:
        lines.append("No successful real-model chat response was captured.")
    lines += [
        "",
        "## Final metrics",
        "",
        "```json",
        json.dumps(evidence["final_metrics"], indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = load_settings()
    host = str(settings.get("app", {}).get("host", "127.0.0.1"))
    port = int(settings.get("app", {}).get("port", 8000))
    base_url = f"http://{host}:{port}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server_log = OUTPUT_DIR / "server.log"

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    with server_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                startup_health = wait_for_server(client, base_url)
                requests = [
                    record(client, base_url, "GET", "/"),
                    record(client, base_url, "GET", "/health"),
                    record(client, base_url, "GET", "/docs"),
                    record(client, base_url, "GET", "/ui"),
                    record(client, base_url, "GET", "/metrics"),
                    record(
                        client,
                        base_url,
                        "POST",
                        "/chat",
                        json={"message": "How can I return a damaged product?"},
                    ),
                    record(
                        client,
                        base_url,
                        "POST",
                        "/chat",
                        json={
                            "message": (
                                "What is the return window for a standard product?"
                            )
                        },
                    ),
                    record(client, base_url, "POST", "/chat", json={"message": ""}),
                    record(
                        client,
                        base_url,
                        "POST",
                        "/chat",
                        json={"message": "   \n\t  "},
                    ),
                    record(
                        client,
                        base_url,
                        "POST",
                        "/chat",
                        json={"message": "x" * 1001},
                    ),
                    record(
                        client,
                        base_url,
                        "POST",
                        "/chat",
                        content="{not valid json",
                        headers={"Content-Type": "application/json"},
                    ),
                    record(client, base_url, "GET", "/metrics"),
                ]
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)

    final_metrics = requests[-1]["body"] if requests else {}
    evidence = {
        "generated_at": utc_now(),
        "base_url": base_url,
        "server_command": " ".join(command),
        "startup_health": startup_health,
        "requests": requests,
        "final_metrics": final_metrics,
        "server_log": str(server_log),
    }
    json_path = OUTPUT_DIR / "api_verification.json"
    md_path = OUTPUT_DIR / "api_verification.md"
    json_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_markdown(md_path, evidence)
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
