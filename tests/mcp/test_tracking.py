import json
import sys
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from click.testing import CliRunner
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import mlflow
from mlflow.mcp import server
from mlflow.mcp.tracking import experiment_commands, run_commands
from mlflow.runs import list_run
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE


@pytest_asyncio.fixture
async def client() -> AsyncIterator[Client]:
    transport = StdioTransport(
        command=sys.executable,
        args=[server.__file__],
        env={
            "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri(),
            "MLFLOW_MCP_TOOLS": "experiments,runs",
        },
    )
    async with Client(transport) as client:
        yield client


def test_search_runs_pagination():
    exp_id = mlflow.create_experiment("mcp_search_runs_pagination")
    for i in range(3):
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metric("score", float(i))

    result = CliRunner().invoke(
        run_commands.commands["search"],
        ["--experiment-id", exp_id, "--max-results", "2"],
    )
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert len(payload["runs"]) == 2
    assert payload["next_page_token"] is not None

    result = CliRunner().invoke(
        run_commands.commands["search"],
        [
            "--experiment-id",
            exp_id,
            "--max-results",
            "2",
            "--page-token",
            payload["next_page_token"],
        ],
    )
    assert result.exit_code == 0

    second_page = json.loads(result.output)
    assert len(second_page["runs"]) == 1
    assert second_page["next_page_token"] is None


def test_list_runs_pagination():
    exp_id = mlflow.create_experiment("mcp_list_runs_pagination")
    for _ in range(3):
        with mlflow.start_run(experiment_id=exp_id):
            pass

    result = CliRunner().invoke(
        list_run,
        ["--experiment-id", exp_id, "--max-results", "2"],
    )
    assert result.exit_code == 0
    assert "Next page token:" in result.output


def test_experiment_tag_round_trip():
    exp_id = mlflow.create_experiment("mcp_experiment_tags")

    result = CliRunner().invoke(
        experiment_commands.commands["set-tag"],
        ["--experiment-id", exp_id, "--key", "team", "--value", "ml"],
    )
    assert result.exit_code == 0

    experiment = mlflow.get_experiment(exp_id)
    assert experiment.tags["team"] == "ml"

    result = CliRunner().invoke(
        experiment_commands.commands["delete-tag"],
        ["--experiment-id", exp_id, "--key", "team"],
    )
    assert result.exit_code == 0

    experiment = mlflow.get_experiment(exp_id)
    assert "team" not in experiment.tags


def test_run_tag_and_metric_history():
    exp_id = mlflow.create_experiment("mcp_run_metadata")
    with mlflow.start_run(experiment_id=exp_id) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_metric("loss", 1.0, step=0)
        mlflow.log_metric("loss", 0.5, step=1)

    result = CliRunner().invoke(
        run_commands.commands["set-tag"],
        ["--run-id", run_id, "--key", "env", "--value", "test"],
    )
    assert result.exit_code == 0

    result = CliRunner().invoke(
        run_commands.commands["update"],
        ["--run-id", run_id, "--description", "baseline model"],
    )
    assert result.exit_code == 0

    run = mlflow.get_run(run_id)
    assert run.data.tags["env"] == "test"
    assert run.data.tags[MLFLOW_RUN_NOTE] == "baseline model"

    result = CliRunner().invoke(
        run_commands.commands["get-metrics"],
        ["--run-id", run_id],
    )
    assert result.exit_code == 0
    metrics = json.loads(result.output)["metrics"]
    assert metrics["loss"] == 0.5

    result = CliRunner().invoke(
        run_commands.commands["get-metric-history"],
        ["--run-id", run_id, "--metric-key", "loss"],
    )
    assert result.exit_code == 0
    history = json.loads(result.output)["history"]
    assert len(history) == 2
    assert history[0]["value"] == 1.0
    assert history[1]["value"] == 0.5


@pytest.mark.asyncio
async def test_search_runs_mcp_tool(client: Client):
    exp_id = mlflow.create_experiment("mcp_search_runs_tool")
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_metric("accuracy", 0.95)

    result = await client.call_tool(
        "search_runs",
        {"experiment_id": exp_id, "max_results": 1000},
        timeout=5,
    )
    payload = json.loads(result.content[0].text)
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["data"]["metrics"]["accuracy"] == 0.95
