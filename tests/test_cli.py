import pytest

from glotto_lens import cli


def test_without_arguments_prints_the_commands(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in cli.COMMANDS)


def test_unknown_command_is_an_error(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


def test_command_gets_the_remaining_arguments(monkeypatch):
    from glotto_lens import answer

    seen = []
    monkeypatch.setattr(answer, "main", seen.append)
    assert cli.main(["ask", "Сколько падежей?", "-l", "Finnish"]) == 0
    assert seen == [["Сколько падежей?", "-l", "Finnish"]]


def test_subcommand_help_names_the_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["ask", "--help"])
    assert "usage: glotto-lens ask" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["build", "lang", "search", "evidence", "eval"])
def test_every_command_has_help_and_builds_nothing(command, monkeypatch, capsys):
    from glotto_lens import database

    monkeypatch.setattr(database, "build", lambda *a, **k: pytest.fail("build was called"))
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    assert f"usage: glotto-lens {command}" in capsys.readouterr().out


def test_program_name_is_restored_after_an_error(monkeypatch):
    import sys

    from glotto_lens import answer

    def broken(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(answer, "main", broken)
    before = sys.argv[0]
    with pytest.raises(RuntimeError):
        cli.main(["ask", "x"])
    assert sys.argv[0] == before


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("glotto-lens ")


def test_version_outside_an_installation(monkeypatch, capsys):
    def missing(name):
        raise cli.PackageNotFoundError(name)

    monkeypatch.setattr(cli, "version", missing)
    assert cli.main(["--version"]) == 0
    assert "(not installed)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv", [["search", "word", "--limit", "-1"], ["lang"], ["build", "--db", "x.sqlite"]]
)
def test_bad_arguments_are_errors(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2


def test_per_feature_search_refuses_chapters():
    with pytest.raises(SystemExit):
        cli.main(["search", "word", "--per-feature", "--kind", "chapter"])


def test_non_negative_message_names_the_value():
    import argparse

    from glotto_lens.cliargs import non_negative

    assert non_negative("0") == 0
    for text in ("-1", "abc"):
        with pytest.raises(argparse.ArgumentTypeError, match=repr(text)):
            non_negative(text)
