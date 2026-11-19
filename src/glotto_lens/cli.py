"""The glotto-lens command: one entry point for building, asking and evaluating.

    glotto-lens build                          download the data and build the database
    glotto-lens ask QUESTION [-l LANGUAGE]     answer a question with cited evidence
    glotto-lens evidence QUERY [-l LANGUAGE]   show the evidence without calling the model
    glotto-lens lang NAME [NAME ...]           look up languages by name or code
    glotto-lens search QUERY                   full-text search over the documents
    glotto-lens eval {check,gaps,run} ...      evaluate on the question set

Each command is the module's own command line (python -m glotto_lens.answer and so on),
so "glotto-lens ask --help" shows its options.
"""

import sys
from collections.abc import Callable

from glotto_lens import answer, context, database, evaluate, languages, search


def _lang(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: glotto-lens lang NAME [NAME ...]  (names, glottocodes, ISO or WALS codes)")
        return
    languages.main(argv)


def _search(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: glotto-lens search QUERY")
        return
    search.main(" ".join(argv))


COMMANDS: dict[str, tuple[Callable[[list[str]], None], str]] = {
    "build": (database.main, "download the data and build the database"),
    "ask": (answer.main, "answer a question with cited evidence (needs an API key)"),
    "evidence": (context.main, "show the evidence for a query, no model call"),
    "lang": (_lang, "look up languages by name or code"),
    "search": (_search, "full-text search over the documents"),
    "eval": (evaluate.main, "evaluate on the question set: check, gaps, run"),
}


def usage() -> str:
    lines = ["usage: glotto-lens COMMAND [ARGS]", "", "commands:"]
    lines += [f"  {name:<9} {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += ["", 'Run "glotto-lens COMMAND --help" for the options of a command.']
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"glotto-lens: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    run, _ = COMMANDS[command]
    program = sys.argv[0]
    sys.argv[0] = f"glotto-lens {command}"  # argparse shows it in usage lines
    try:
        run(rest)
    finally:
        sys.argv[0] = program
    return 0


if __name__ == "__main__":
    sys.exit(main())
