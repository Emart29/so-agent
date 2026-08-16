"""Tests for the command line.

These run against Typer's test runner with no provider configured, so they cover
the parts that must work before anyone has a key: help text, argument
validation, and the failure messages. The rule they exist to protect is that a
command never exits zero after failing, because a benchmark script that silently
continues past a failed extraction produces numbers nobody should trust.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cli.main import app
from contracts.schemas import CONTRACTS

runner = CliRunner()


class TestHelp:
    def test_every_command_is_listed(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("providers", "probe", "extract", "bench", "metrics",
                        "report", "replay"):
            assert command in result.output

    @pytest.mark.parametrize(
        "command", ["providers", "probe", "extract", "bench", "metrics",
                    "report", "replay"]
    )
    def test_each_command_has_its_own_help(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "command", ["probe", "extract", "bench", "metrics", "replay"]
    )
    def test_every_command_accepts_a_provider(self, command):
        """A command that only works against the default makes the default the
        only measurable thing."""
        result = runner.invoke(app, [command, "--help"])
        assert "--provider" in result.output


class TestProviders:
    def test_it_lists_the_configured_providers(self):
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "groq" in result.output
        assert "openrouter" in result.output

    def test_it_shows_the_budget_for_a_metered_provider(self):
        """The first question on a fresh clone is how much allowance is left."""
        result = runner.invoke(app, ["providers"])
        assert "45" in result.output


class TestValidation:
    def test_an_unknown_provider_is_refused_by_name(self):
        result = runner.invoke(app, ["probe", "--provider", "nope"])
        assert result.exit_code == 1
        assert "nope" in result.output

    def test_an_unknown_schema_names_the_known_ones(self):
        result = runner.invoke(
            app, ["extract", "--schema", "not_a_schema", "--text", "x"]
        )
        assert result.exit_code == 1
        assert any(name in result.output for name in CONTRACTS)

    def test_extract_needs_some_input(self):
        result = runner.invoke(app, ["extract", "--schema", "ticket_summary"])
        assert result.exit_code == 1

    def test_a_missing_results_file_is_refused(self):
        result = runner.invoke(app, ["report", "--results", "no_such_file.json"])
        assert result.exit_code == 1

    def test_bench_will_not_guess_a_model(self):
        """Guessing a model measures the guess."""
        result = runner.invoke(app, ["bench"])
        assert result.exit_code == 1
        assert "--model" in result.output


class TestReplay:
    def test_an_unknown_run_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "config.settings.LOG_DB_PATH", str(tmp_path / "empty.db")
        )
        result = runner.invoke(app, ["replay", "nosuchrun"])
        assert result.exit_code == 1
        assert "nosuchrun" in result.output

    def test_a_run_without_stored_input_says_so(self, tmp_path, monkeypatch):
        """Older rows predate input capture; that is a clear message, not a crash."""
        from store.log import AttemptLog, AttemptRow

        db = tmp_path / "runs.db"
        log = AttemptLog(db)
        log.record(
            AttemptRow(
                run_id="old", attempt_index=1, provider="groq", model="m",
                tier="json_object", schema_name="ticketsummary", success=False,
                failure_type="not_json",
            )
        )
        log.close()
        monkeypatch.setattr("config.settings.LOG_DB_PATH", str(db))

        result = runner.invoke(app, ["replay", "old"])
        assert result.exit_code == 1
        assert "cannot be replayed" in result.output


class TestMetrics:
    def test_an_empty_log_says_what_to_run_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.LOG_DB_PATH", str(tmp_path / "e.db"))
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 1
        assert "bench" in result.output

    def test_it_prints_the_tables_for_a_populated_log(self, tmp_path, monkeypatch):
        from store.log import AttemptLog, AttemptRow

        db = tmp_path / "runs.db"
        log = AttemptLog(db)
        log.record_many([
            AttemptRow(
                run_id=f"r{i}", attempt_index=1, provider="groq", model="m",
                tier="json_object", schema_name="ticketsummary",
                schema_difficulty="simple", success=i % 4 != 0,
                failure_type=None if i % 4 else "not_json",
            )
            for i in range(20)
        ])
        log.close()
        monkeypatch.setattr("config.settings.LOG_DB_PATH", str(db))

        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "first attempt" in result.output
        assert "not_json" in result.output


class TestReport:
    def test_it_builds_from_a_saved_run(self, tmp_path, monkeypatch):
        from bench.results import save_results
        from bench.run import CellResult

        results = tmp_path / "results.json"
        save_results(
            [CellResult(
                provider="groq", model="m", tier="json_object",
                contract="TicketSummary", difficulty="simple",
                first_attempt_ok=9, final_ok=10, total=10,
            )],
            results,
            sampling={"repeats": 3},
        )
        monkeypatch.setattr("config.settings.LOG_DB_PATH", str(tmp_path / "r.db"))
        out = tmp_path / "report.html"

        result = runner.invoke(
            app, ["report", "--results", str(results), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "json_object" in html

    def test_the_saved_run_is_readable_without_the_code(self, tmp_path):
        """The results file is the artefact; it must stand on its own as JSON."""
        from bench.results import save_results
        from bench.run import CellResult

        path = tmp_path / "results.json"
        save_results([CellResult(
            provider="groq", model="m", tier="json_object", contract="c",
            difficulty="simple", total=5, final_ok=5,
        )], path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["cells"][0]["model"] == "m"
