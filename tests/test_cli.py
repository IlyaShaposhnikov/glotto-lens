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
    seen = []
    monkeypatch.setitem(cli.COMMANDS, "ask", (seen.append, "help"))
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

    def broken(argv):
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.COMMANDS, "ask", (broken, "help"))
    before = sys.argv[0]
    with pytest.raises(RuntimeError):
        cli.main(["ask", "x"])
    assert sys.argv[0] == before


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("glotto-lens ")
