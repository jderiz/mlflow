"""
MCP-exposed Click commands for classical MLflow tracking operations.

These commands supplement mlflow.experiments and mlflow.runs with paginated run
search, tag editing, metrics/parameters retrieval, and related tracking APIs.
"""

import json

import click

from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.mcp.decorator import mlflow_mcp
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE
from mlflow.utils.string_utils import _create_table

EXPERIMENT_ID = click.option("--experiment-id", "-x", type=click.STRING, required=True)
RUN_ID = click.option("--run-id", type=click.STRING, required=True)


def _validate_max_results(ctx, param, value):
    if value is not None and value < 0:
        raise click.BadParameter("max-results must be a non-negative integer")
    return value


def _make_json_serializable(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(value) for value in obj]
    return obj


@click.group("mcp-experiment-tracking")
def experiment_commands():
    """
    Classical ML experiment metadata operations exposed for MCP.
    """


@click.group("mcp-run-tracking")
def run_commands():
    """
    Classical ML run search and metadata operations exposed for MCP.
    """


@experiment_commands.command("set-tag")
@mlflow_mcp(tool_name="set_experiment_tag")
@EXPERIMENT_ID
@click.option("--key", "-k", type=click.STRING, required=True, help="Tag key to set.")
@click.option("--value", type=click.STRING, required=True, help="Tag value to set.")
def set_experiment_tag(experiment_id: str, key: str, value: str) -> None:
    """
    Set a tag on an experiment. Experiment metadata such as descriptions is stored
    as tags; use key ``mlflow.note.content`` for a human-readable description.
    """
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, key, value)
    click.echo(f"Set tag '{key}' on experiment {experiment_id}.")


@experiment_commands.command("delete-tag")
@mlflow_mcp(tool_name="delete_experiment_tag")
@EXPERIMENT_ID
@click.option("--key", "-k", type=click.STRING, required=True, help="Tag key to delete.")
def delete_experiment_tag(experiment_id: str, key: str) -> None:
    """Delete a tag from an experiment."""
    client = MlflowClient()
    client.delete_experiment_tag(experiment_id, key)
    click.echo(f"Deleted tag '{key}' from experiment {experiment_id}.")


@run_commands.command("search")
@mlflow_mcp(tool_name="search_runs")
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID to search runs in.",
)
@click.option(
    "--filter-string",
    type=click.STRING,
    default="",
    help='MLflow search filter string (e.g. "metrics.accuracy > 0.9").',
)
@click.option(
    "--max-results",
    type=click.INT,
    default=SEARCH_MAX_RESULTS_DEFAULT,
    callback=_validate_max_results,
    help=(
        f"Maximum number of runs to return per page (default: {SEARCH_MAX_RESULTS_DEFAULT}). "
        "Use --page-token to fetch additional pages when more runs exist."
    ),
)
@click.option(
    "--page-token",
    type=click.STRING,
    default=None,
    help="Pagination token from a previous search_runs response.",
)
@click.option(
    "--order-by",
    type=click.STRING,
    default=None,
    help="Comma-separated order-by clauses (e.g. 'metrics.accuracy DESC').",
)
@click.option(
    "--view",
    "-v",
    default="active_only",
    help="View type: 'active_only' (default), 'deleted_only', or 'all'.",
)
@click.option(
    "--output",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format: 'json' (default) or 'table'.",
)
def search_runs(
    experiment_id: str,
    filter_string: str,
    max_results: int,
    page_token: str | None,
    order_by: str | None,
    view: str,
    output: str,
) -> None:
    """
    Search runs in an experiment with pagination support.

    Default page size is 1000 runs. When ``next_page_token`` is present in the
    response, call again with ``--page-token`` to retrieve the next page. For
    experiments with more than 1000 runs, always paginate rather than assuming
    a single call returns all data.
    """
    client = MlflowClient()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    order_by_list = order_by.split(",") if order_by else None

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=view_type,
        max_results=max_results,
        order_by=order_by_list,
        page_token=page_token,
    )

    if output == "json":
        result = {
            "runs": [_make_json_serializable(run.to_dictionary()) for run in runs],
            "next_page_token": _make_json_serializable(runs.token),
        }
        click.echo(json.dumps(result, indent=2))
    else:
        table = [
            [
                run.info.run_id,
                run.info.run_name or "",
                run.info.status,
                str(run.data.metrics),
            ]
            for run in runs
        ]
        click.echo(_create_table(table, headers=["Run ID", "Name", "Status", "Metrics"]))
        if runs.token:
            click.echo(f"\nNext page token: {runs.token}")


@run_commands.command("set-tag")
@mlflow_mcp(tool_name="set_run_tag")
@RUN_ID
@click.option("--key", "-k", type=click.STRING, required=True, help="Tag key to set.")
@click.option("--value", type=click.STRING, required=True, help="Tag value to set.")
def set_run_tag(run_id: str, key: str, value: str) -> None:
    """Set a tag on a run."""
    client = MlflowClient()
    client.set_tag(run_id, key, value)
    click.echo(f"Set tag '{key}' on run {run_id}.")


@run_commands.command("delete-tag")
@mlflow_mcp(tool_name="delete_run_tag")
@RUN_ID
@click.option("--key", "-k", type=click.STRING, required=True, help="Tag key to delete.")
def delete_run_tag(run_id: str, key: str) -> None:
    """Delete a tag from a run."""
    client = MlflowClient()
    client.delete_tag(run_id, key)
    click.echo(f"Deleted tag '{key}' from run {run_id}.")


@run_commands.command("update")
@mlflow_mcp(tool_name="update_run")
@RUN_ID
@click.option("--name", type=click.STRING, default=None, help="New run name.")
@click.option(
    "--status",
    type=click.Choice(
        ["RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"],
        case_sensitive=False,
    ),
    default=None,
    help="New run status.",
)
@click.option(
    "--description",
    type=click.STRING,
    default=None,
    help="Run description (stored as the mlflow.note.content tag).",
)
def update_run(
    run_id: str,
    name: str | None,
    status: str | None,
    description: str | None,
) -> None:
    """
    Update a run's name, status, and/or description.

    At least one of --name, --status, or --description must be provided.
    """
    if name is None and status is None and description is None:
        raise click.UsageError("Must specify at least one of --name, --status, or --description.")

    client = MlflowClient()
    changes = []

    if name is not None or status is not None:
        client.update_run(run_id, status=status, name=name)
        if name is not None:
            changes.append(f"renamed to '{name}'")
        if status is not None:
            changes.append(f"status set to {status.upper()}")

    if description is not None:
        client.set_tag(run_id, MLFLOW_RUN_NOTE, description)
        changes.append("description updated")

    click.echo(f"Updated run {run_id}: " + "; ".join(changes) + ".")


@run_commands.command("get-metrics")
@mlflow_mcp(tool_name="get_run_metrics")
@RUN_ID
def get_run_metrics(run_id: str) -> None:
    """Get all latest metrics logged for a run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    click.echo(json.dumps({"run_id": run_id, "metrics": run.data.metrics}, indent=2))


@run_commands.command("get-params")
@mlflow_mcp(tool_name="get_run_params")
@RUN_ID
def get_run_params(run_id: str) -> None:
    """Get all parameters logged for a run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    click.echo(json.dumps({"run_id": run_id, "params": run.data.params}, indent=2))


@run_commands.command("get-metric-history")
@mlflow_mcp(tool_name="get_metric_history")
@RUN_ID
@click.option(
    "--metric-key",
    "-m",
    type=click.STRING,
    required=True,
    help="Metric key to retrieve history for.",
)
def get_metric_history(run_id: str, metric_key: str) -> None:
    """Get the full history of values logged for a single metric on a run."""
    client = MlflowClient()
    history = client.get_metric_history(run_id, metric_key)
    metrics = [
        {"key": m.key, "value": m.value, "timestamp": m.timestamp, "step": m.step} for m in history
    ]
    payload = {"run_id": run_id, "metric_key": metric_key, "history": metrics}
    click.echo(json.dumps(payload, indent=2))
