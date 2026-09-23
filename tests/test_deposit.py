"""
Check this deposit's prose against this deposit's files.

    python tests/test_deposit.py

Run it after any change. It exists because the README drifted: it said the
flexural replicates were missing when all ten curves were present, it gave the
zone boundaries of a partition that had been removed, it listed six scripts out
of eighteen, and it named 25 reproduced values where the script prints 34.
Every one of those was true when it was written. Prose does not fail a test
suite, so it rots quietly while the code around it is kept honest.

So this is the test suite for the prose. It reads the claims out of the two
documents and puts each one against the file it describes:

  row counts       every "| `file.csv` | 1234 |" line in the README, and every
                   "## `file.csv` - 1234 rows" heading in the dictionary
  columns          every column named in a dictionary table must exist in the
                   file, and every column in the file must be documented
  scripts          the README must name every .py in the deposit, and must not
                   name one that is not there
  data files       every file in data/ must have a dictionary section
  imports          every third-party module imported anywhere in the deposit
                   must be in requirements.txt
  reproduced count the number of values the README promises must be the number
                   reproduce_paper_values.py prints, which means running it;
                   this is why the suite takes half a minute rather than a second

The last one is the check that would have caught the real failure: five figure
scripts imported a plotting toolkit that was never in the deposit and only
resolved because the author's machine had it on sys.path.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Import name to distribution name, where the two differ.
DISTRIBUTIONS = {"sklearn": "scikit-learn", "PIL": "pillow",
                 "cv2": "opencv-python", "skimage": "scikit-image",
                 "yaml": "pyyaml"}


# The counts in the README and the dictionary are spelled out, because that is
# how they read on the page. Both checks used to carry their own short list and
# both lists ran out as the deposit grew, which made the checks stop checking
# rather than fail: an unknown word gave None, and None never equals a count.
# One list, long enough to have room, and an unreadable word is now a failure.
NUMBER_WORDS = {w: n for n, w in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
     "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one",
     "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six",
     "twenty-seven", "twenty-eight", "twenty-nine", "thirty", "thirty-one",
     "thirty-two", "thirty-three", "thirty-four", "thirty-five"))}
NUMBER_WORDS.update({w.capitalize(): n for w, n in list(NUMBER_WORDS.items())})


class Deposit:
    """
    The checks, over one deposit directory.

    Takes the root as an argument, not assuming its own location, so
    that the same code can check a copy in a clean room, and so that the
    pre-publication gate, which lives outside the deposit, can run it without
    holding a second copy of the logic.
    """

    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]
        self.data = self.root / "data"
        self.scripts = self.root / "scripts"
        self.package = self.scripts / "hybridmap"
        self.tests = self.root / "tests"
        self.readme = self.root / "README.md"
        self.dictionary = self.data / "DATA_DICTIONARY.md"
        self.problems = []
        self.checks = 0
        self._rows = {}

    # ------------------------------------------------------------------ util

    def fail(self, message):
        self.problems.append(message)

    def text(self, path):
        return path.read_text(encoding="utf8", errors="replace")

    def rows_of(self, name):
        """Row count of a CSV, cached, or None if it is not one we can read."""
        if name not in self._rows:
            path = self.data / name
            self._rows[name] = (len(pd.read_csv(path))
                                if path.exists() and path.suffix == ".csv"
                                else None)
        return self._rows[name]

    def python_files(self):
        """Every Python file that ships, wherever it sits."""
        return sorted(list(self.scripts.rglob("*.py")) +
                      list(self.tests.rglob("*.py")))

    # Directories that can appear inside the deposit but are not part of it.
    # A reader following the README makes a virtualenv here, and a sweep that
    # walks the whole root then reads site-packages: slow, and it reports other
    # people's files as though they were ours.
    NOT_OURS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules",
                ".ipynb_checkpoints", ".pytest_cache"}

    def shipped_files(self, suffixes=None):
        """Every file the deposit itself owns, filtered by extension."""
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(self.root).parts[:-1]
            if any(p in self.NOT_OURS for p in parts):
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue
            yield path

    # ---------------------------------------------------------------- checks

    def check_row_counts(self):
        """Both documents quote row counts. They must match the files."""
        # README table rows look like:  | `shallow_map.csv` | 8553 | what it is |
        for line in self.text(self.readme).splitlines():
            m = re.match(r"\|\s*`([\w.]+\.csv)`\s*\|\s*([\d,]+)\s*\|", line)
            if not m:
                continue
            name, claimed = m.group(1), int(m.group(2).replace(",", ""))
            actual = self.rows_of(name)
            self.checks += 1
            if actual is None:
                self.fail(f"README lists {name}, which is not in data/")
            elif actual != claimed:
                self.fail(f"README says {name} has {claimed} rows; "
                          f"it has {actual}")

        # Dictionary headings look like:  ## `shallow_map.csv` - 8553 rows
        for m in re.finditer(r"^##\s+`([\w.]+\.csv)`\s*[-—]\s*([\d,]+)\s*rows",
                             self.text(self.dictionary), re.M):
            name, claimed = m.group(1), int(m.group(2).replace(",", ""))
            actual = self.rows_of(name)
            self.checks += 1
            if actual is None:
                self.fail(f"dictionary documents {name}, which is not in data/")
            elif actual != claimed:
                self.fail(f"dictionary says {name} has {claimed} rows; "
                          f"it has {actual}")

    def dictionary_sections(self):
        """Split the dictionary into {filename: section text}."""
        text = self.text(self.dictionary)
        heads = list(re.finditer(r"^##\s+`([\w.]+)`", text, re.M))
        out = {}
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            out[h.group(1)] = text[h.start():end]
        return out

    @staticmethod
    def documented_columns(section):
        """
        Column names declared in a section's tables.

        Only the FIRST cell of each table row is read. That is where the column
        name sits; later cells hold units, ranges and prose, and those carry
        backticked *values*, `basalt`, `flax_face`, which are not columns. An
        earlier version of this check read every backtick in the section and
        reported three constituent labels as undocumented columns.

        One cell can declare several columns: | `E_100`, `E_150`, `E_200` | GPa |
        """
        names = set()
        for line in section.splitlines():
            if not line.startswith("|"):
                continue
            first = line.split("|")[1]
            names |= set(re.findall(r"`([A-Za-z_][\w.]*)`", first))
        return names

    def check_columns(self):
        """
        Every column of every CSV must be declared in its own section, and the
        section must not declare a column the file does not have.
        """
        every_column = set()
        for path in self.data.glob("*.csv"):
            every_column |= set(pd.read_csv(path, nrows=1).columns)

        sections = self.dictionary_sections()
        for path in sorted(self.data.glob("*.csv")):
            sec = sections.get(path.name)
            self.checks += 1
            if sec is None:
                self.fail(f"{path.name} has no section in the data dictionary")
                continue

            actual = set(pd.read_csv(path, nrows=1).columns)
            named = self.documented_columns(sec)

            # A section may inherit its shared columns rather than repeat them,
            # as classified_shallow.csv does from shallow_map.csv. Honour that,
            # but only when the section says so in as many words.
            if re.search(r"[Cc]olumns as above", sec):
                for other, text in sections.items():
                    if other != path.name and other.endswith(".csv"):
                        named |= self.documented_columns(text)

            undocumented = sorted(actual - named)
            if undocumented:
                self.fail(f"{path.name}: columns present but not documented: "
                          f"{', '.join(undocumented)}")

            phantom = sorted(named - actual - every_column)
            if phantom:
                self.fail(f"{path.name}: documented but not present: "
                          f"{', '.join(phantom)}")

    def check_scripts(self):
        """The README must list every script in scripts/, and only real ones."""
        text = self.text(self.readme)
        listed = set(re.findall(r"`(\w+\.py)`", text))
        present = {p.name for p in self.scripts.glob("*.py")}

        self.checks += 1
        missing = sorted(present - listed)
        if missing:
            self.fail(f"scripts present but not listed in the README: "
                      f"{', '.join(missing)}")

        self.checks += 1
        everything = {p.name for p in self.python_files()}
        phantom = sorted(listed - everything)
        if phantom:
            self.fail(f"README names scripts that are not in the deposit: "
                      f"{', '.join(phantom)}")

        # The README states how many scripts there are; keep that honest too.
        m = re.search(r"([\w-]+) files, in three groups", text)
        if m:
            self.checks += 1
            claimed = NUMBER_WORDS.get(m.group(1))
            if claimed is None:
                self.fail(f"README says '{m.group(1)} files' and I cannot read "
                          f"that as a number")
            elif claimed != len(present):
                self.fail(f"README says {m.group(1).lower()} scripts; "
                          f"there are {len(present)}")

    def check_modules(self):
        """
        The package and the tests are documented to the same standard.

        A reusable package whose modules are not named anywhere a reader looks
        is not reusable, and this is the same failure mode as the script table
        drifting, so it gets the same treatment.
        """
        listed = set(re.findall(r"`(\w+\.py)`", self.text(self.readme)))
        for directory in (self.package, self.tests):
            if not directory.exists():
                continue
            self.checks += 1
            present = {p.name for p in directory.glob("*.py")}
            missing = sorted(present - listed)
            if missing:
                self.fail(f"{directory.name}/ holds modules the README does "
                          f"not name: {', '.join(missing)}")

    # In data/ but not data: the dictionary describes measurements, and a
    # licence and a dictionary are neither described by it nor counted in it.
    NOT_DATA = {"DATA_DICTIONARY.md", "LICENSE"}

    def data_files(self):
        return [p for p in sorted(self.data.iterdir())
                if p.name not in self.NOT_DATA]

    def check_data_files(self):
        """Nothing in data/ may be undocumented."""
        sections = self.dictionary_sections()
        for path in self.data_files():
            self.checks += 1
            if path.name not in sections:
                self.fail(f"{path.name} is in data/ but has no dictionary "
                          f"section")

        # And the count the dictionary opens with.
        m = re.search(r"^(\w+) files:", self.text(self.dictionary), re.M)
        if m:
            self.checks += 1
            if m.group(1) not in NUMBER_WORDS:
                self.fail(f"dictionary opens with '{m.group(1)} files', a "
                          f"number this check cannot read")
            claimed = NUMBER_WORDS.get(m.group(1))
            actual = len(self.data_files())
            if claimed != actual:
                self.fail(f"dictionary opens with '{m.group(1)} files'; "
                          f"data/ holds {actual}")

    def check_reproduce_count(self):
        """
        The README promises a specific last line from reproduce_paper_values.py.
        Run the script and hold the promise against what it prints.

        Until this round the check read the promise and compared it against
        nothing, so the README could promise any number at all and pass as long
        as it promised the same one in both places, and it also passed when the
        README promised nothing. Both of those are failures now. The cost is the
        time the script takes, half a minute, and it buys the only check in here
        that reaches past the prose and into the analysis.
        """
        text = self.text(self.readme)
        quoted = set(re.findall(r"(\d+) of \1 values reproduce", text))
        quoted |= set(re.findall(r"all (\d+) published values", text))

        self.checks += 1
        if not quoted:
            self.fail("the README does not say how many values "
                      "reproduce_paper_values.py reproduces")
            return
        if len(quoted) > 1:
            self.fail(f"the README quotes more than one count of reproduced "
                      f"values: {', '.join(sorted(quoted))}")
            return
        promised = quoted.pop()

        script = self.scripts / "reproduce_paper_values.py"
        self.checks += 1
        if not script.exists():
            self.fail(f"the README promises {promised} reproduced values and "
                      f"{script.name} is not in the deposit")
            return

        # Byte compilation off: this test is itself run by the pre-publication
        # gate, which fails on a __pycache__ directory inside the deposit.
        proc = subprocess.run([sys.executable, script.name], cwd=self.scripts,
                              capture_output=True, text=True, timeout=900,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        printed = re.findall(r"(\d+) of (\d+) values reproduce", proc.stdout)

        self.checks += 1
        if proc.returncode != 0 or not printed:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            last = f", last line {tail[-1].strip()!r}" if tail else ""
            self.fail(f"{script.name} exited {proc.returncode} without a count "
                      f"to compare the README against{last}")
            return

        reproduced, total = printed[-1]
        if reproduced != total:
            self.fail(f"{script.name} reproduces {reproduced} of {total} "
                      f"values; the README promises all of them")
        elif total != promised:
            self.fail(f"{script.name} prints '{reproduced} of {total} values "
                      f"reproduce' and the README promises {promised}")

    def check_citation_authors_agree(self):
        """
        CITATION.cff names the five authors twice, and the two must match.

        Once for the dataset and once under preferred-citation. Correcting a
        name, adding an ORCID or reordering in one block and not the other is
        the same failure the DOI check exists to catch, and the two lists sit
        forty lines apart.
        """
        citation = self.root / "CITATION.cff"
        if not citation.exists():
            return
        text = self.text(citation)

        # Split at preferred-citation, then read the name pairs on each side.
        head, _, tail = text.partition("preferred-citation:")
        def names(block):
            return re.findall(r"given-names:\s*(\S+)\s*\n\s*family-names:\s*(\S+)",
                              block)

        self.checks += 1
        dataset, article = names(head), names(tail)
        if not dataset or not article:
            self.fail("CITATION.cff should list authors for the dataset and "
                      "again under preferred-citation; one of the two is empty")
        elif dataset != article:
            only_dataset = [f"{g} {f}" for g, f in dataset if (g, f) not in article]
            only_article = [f"{g} {f}" for g, f in article if (g, f) not in dataset]
            detail = (f"dataset only: {', '.join(only_dataset)}; "
                      f"citation only: {', '.join(only_article)}"
                      if only_dataset or only_article else "the order differs")
            self.fail(f"CITATION.cff's two author lists disagree ({detail})")

    def check_doi_agrees(self):
        """
        CITATION.cff and the README must be at the same stage on the DOI.

        The DOI is minted when the paper is accepted and has to be written in
        both places. Filling in one and forgetting the other is the entire
        failure mode of a value recorded twice, and it would ship a citation
        file pointing at a record the README says does not exist yet.
        """
        citation = self.root / "CITATION.cff"
        if not citation.exists():
            return

        # A live DOI is uncommented in the .cff; the README carries it in bold
        # beside the word DOI. Anything else means the placeholder is still in
        # place, which is fine as long as both agree that it is.
        cff = re.search(r"^\s*doi:\s*(\S+)", self.text(citation), re.M)
        readme = re.search(r"DOI[^\n]*?\b(10\.\d{4,}/[^\s*`]+)",
                           self.text(self.readme))

        self.checks += 1
        if bool(cff) != bool(readme):
            has, lacks = ("CITATION.cff", "README.md") if cff else ("README.md",
                                                                    "CITATION.cff")
            self.fail(f"{has} carries a DOI and {lacks} does not; the DOI goes "
                      f"in both places or neither")
        elif cff and readme and cff.group(1).rstrip(".") != readme.group(1).rstrip("."):
            self.fail(f"CITATION.cff says the DOI is {cff.group(1)} and the "
                      f"README says {readme.group(1)}")

    def check_dependencies_agree(self):
        """
        requirements.txt and pyproject.toml must name the same packages.

        Two lists of dependencies is one more than anybody keeps up to date.
        Whichever gets a new entry, this fails until the other does too.
        """
        pyproject = self.root / "pyproject.toml"
        req = self.root / "requirements.txt"
        if not pyproject.exists() or not req.exists():
            return

        def names(text):
            return {re.split(r"[<>=!\[;]", line.strip().strip('",'))[0]
                    .strip().lower()
                    for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("#")}

        declared = names(self.text(req))
        block = re.search(r"dependencies\s*=\s*\[(.*?)\]",
                          self.text(pyproject), re.S)
        self.checks += 1
        if block is None:
            self.fail("pyproject.toml declares no dependencies block")
            return
        packaged = names(block.group(1)) - {""}

        for missing in sorted(declared - packaged):
            self.fail(f"{missing} is in requirements.txt but not in "
                      f"pyproject.toml")
        for missing in sorted(packaged - declared):
            self.fail(f"{missing} is in pyproject.toml but not in "
                      f"requirements.txt")

    def check_house_style(self):
        """
        No em-dashes, anywhere that ships.

        A house rule rather than a matter of correctness, which is exactly why
        it needs a check: nobody notices one arriving, and twenty of them had
        accumulated in the data dictionary before anything looked. A comma, a
        colon or a full stop says the same thing.
        """
        for path in self.shipped_files({".py", ".md", ".txt", ".yaml", ".yml",
                                        ".json"}):
            self.checks += 1
            for i, line in enumerate(self.text(path).splitlines(), 1):
                # The one legitimate use is a pattern that has to match an
                # em-dash in order to reject it.
                if "—" in line and "re." not in line:
                    self.fail(f"{path.relative_to(self.root)}:{i} uses an "
                              f"em-dash; use a comma, a colon or a full stop")

    def check_imports_declared(self):
        """
        Every third-party module the deposit imports must be in
        requirements.txt. Standard library, and the deposit's own modules and
        packages, are excluded.
        """
        declared = set()
        req = self.root / "requirements.txt"
        if req.exists():
            for line in self.text(req).splitlines():
                name = re.split(r"[<>=!\[;]", line.strip())[0].strip().lower()
                if name and not name.startswith("#"):
                    declared.add(name)

        # Anything importable by name from inside the deposit: a module beside
        # the importer, or a package directory.
        own = {p.stem for p in self.python_files()}
        own |= {d.name for d in self.scripts.iterdir() if d.is_dir()}
        stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}

        for path in self.python_files():
            self.checks += 1
            try:
                tree = ast.parse(self.text(path))
            except SyntaxError as exc:
                self.fail(f"{path.name} does not parse: {exc}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    # level > 0 is a relative import, always our own
                    roots = [] if node.level else [(node.module or "").split(".")[0]]
                else:
                    continue
                for root in roots:
                    if not root or root in own or root in stdlib:
                        continue
                    dist = DISTRIBUTIONS.get(root, root).lower()
                    if dist not in declared:
                        self.fail(f"{path.relative_to(self.root)} imports "
                                  f"'{root}', which is not in requirements.txt")

    def run_all(self):
        self.check_row_counts()
        self.check_columns()
        self.check_scripts()
        self.check_modules()
        self.check_data_files()
        self.check_reproduce_count()
        self.check_citation_authors_agree()
        self.check_doi_agrees()
        self.check_dependencies_agree()
        self.check_house_style()
        self.check_imports_declared()
        return self


def main(root=None):
    deposit = Deposit(root).run_all()
    if deposit.problems:
        print(f"{len(deposit.problems)} disagreement(s) between the "
              f"documents and the files\n")
        for p in deposit.problems:
            print(f"  - {p}")
        return 1
    print(f"documentation matches the deposit ({deposit.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
