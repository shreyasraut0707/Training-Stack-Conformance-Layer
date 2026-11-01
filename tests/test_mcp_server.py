"""End-to-end MCP test: start the server over stdio as an agent would, then call its tools.
Runs where the optional `mcp` package is installed (the conform-agent environment)."""
import asyncio
import json
import os
import sys

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

from conform.bank import PROJECT_ROOT  # noqa: E402
from conform.proxy_model import LLAMA_TYPE, build_proxy  # noqa: E402


def _payload(result):
    data = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]
    return data if data is not None else json.loads(result.content[0].text)


async def _session_calls(calls, cwd):
    params = StdioServerParameters(command=sys.executable, args=["-m", "conform.mcp_server"], cwd=str(cwd),
                                   env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            results = [_payload(await session.call_tool(name, args, read_timeout_seconds=900))
                       for name, args in calls]
            return tools, results


@pytest.fixture(scope="module", autouse=True)
def tokenizer_available():
    try:
        build_proxy(LLAMA_TYPE)
    except OSError:
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")


def test_agent_can_list_probes_and_run_diagnose(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(json.dumps({"model": LLAMA_TYPE, "trainer": "trl", "intent": {"supervise_eos": True}}),
                      encoding="utf-8")
    tools, (probes, diagnosis) = asyncio.run(_session_calls(
        [("list_probes", {}), ("diagnose", {"symptom": "no-stop", "recipe_path": str(recipe)})], tmp_path))
    assert tools == ["diagnose", "diff_environments", "list_probes", "run_all", "run_probe"]
    assert {p["id"] for p in probes["probes"]} >= {"P1", "P2", "P3", "P6", "P7"}
    assert diagnosis["probes_in_order"] == ["P7", "P2", "P3"]
    assert [r["probe"] for r in diagnosis["results"]] == ["P7-eos-stopping", "P2-loss-bearing-tokens",
                                                          "P3-train-inference-tokens"]
    assert all(r["status"] == "pass" for r in diagnosis["results"]), diagnosis
    assert diagnosis["command"].startswith("conform diagnose --symptom no-stop")
    assert os.path.exists(diagnosis["report"])
    assert "unknown" in diagnosis["note"]
