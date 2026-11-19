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


def test_lang_and_search_need_arguments(capsys):
    cli.main(["lang"])
    cli.main(["search"])
    out = capsys.readouterr().out
    assert "usage: glotto-lens lang" in out and "usage: glotto-lens search" in out
