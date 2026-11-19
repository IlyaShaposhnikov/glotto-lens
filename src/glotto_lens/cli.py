"""The glotto-lens command: one entry point for building, asking and evaluating.

    glotto-lens build                          download the data and build the database
    glotto-lens ask QUESTION [-l LANGUAGE]     answer a question with cited evidence
    glotto-lens evidence QUERY [-l LANGUAGE]   show the evidence without calling the model
    glotto-lens lang NAME [NAME ...]           look up languages by name or code
    glotto-lens search QUERY                   full-text search over the documents
    glotto-lens eval {check,gaps,run} ...      evaluate on the question set

Each command is the module's own command line (python -m glotto_lens.answer and so on),
so "glotto-lens ask --help" shows its options, and "glotto-lens --version" the version.
"""

import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

# Command -> (module of glotto_lens whose main() runs it, help). Modules are imported only
# when their command runs: "lang" does not load the LLM client, and tests can patch
# module.main before calling cli.main.
COMMANDS: dict[str, tuple[str, str]] = {
    "build": ("database", "download the data and build the database"),
    "ask": ("answer", "answer a question with cited evidence (needs an API key)"),
    "evidence": ("context", "show the evidence for a query, no model call"),
    "lang": ("languages", "look up languages by name or code"),
    "search": ("search", "full-text search over the documents"),
    "eval": ("evaluate", "evaluate on the question set: check, gaps, run"),
}


def usage() -> str:
    lines = ["usage: glotto-lens [--version] COMMAND [ARGS]", "", "commands:"]
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
    if argv[0] == "--version":
        try:
            print(f"glotto-lens {version('glotto-lens')}")
        except PackageNotFoundError:  # running from a source tree that is not installed
            print("glotto-lens (not installed)")
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"glotto-lens: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, _ = COMMANDS[command]
    run = import_module(f"glotto_lens.{module}").main
    program = sys.argv[0]
    sys.argv[0] = f"glotto-lens {command}"  # argparse shows it in usage lines
    try:
        run(rest)
    finally:
        sys.argv[0] = program
    return 0


if __name__ == "__main__":
    sys.exit(main())
