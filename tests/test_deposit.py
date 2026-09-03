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

The last one is the check that would have caught the real failure: five figure
scripts imported a plotting toolkit that was never in the deposit and only
resolved because the author's machine had it on sys.path.
"""

import ast
import re
import sys
from pathlib import Path

import pandas as pd

# Import name to distribution name, where the two differ.
DISTRIBUTIONS = {"sklearn": "scikit-learn", "PIL": "pillow",
                 "cv2": "opencv-python", "skimage": "scikit-image",
                 "yaml": "pyyaml"}


class Deposit:
    """
    The checks, over one deposit directory.

    Takes the root as an argument rather than assuming its own location, so
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
        m = re.search(r"(\w+) files, in three groups", text)
        if m:
            self.checks += 1
            words = {"Twelve": 12, "Thirteen": 13, "Fourteen": 14,
                     "Fifteen": 15, "Sixteen": 16, "Seventeen": 17,
                     "Eighteen": 18, "Nineteen": 19, "Twenty": 20}
            claimed = words.get(m.group(1))
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
            words = {"Eight": 8, "Nine": 9, "Ten": 10, "Eleven": 11,
                     "Twelve": 12}
            claimed = words.get(m.group(1))
            actual = len(self.data_files())
            if claimed != actual:
                self.fail(f"dictionary opens with '{m.group(1)} files'; "
                          f"data/ holds {actual}")

    def check_reproduce_count(self):
        """
        The README promises a specific last line from reproduce_paper_values.py.
        Read the promise, run nothing: the script itself is a separate gate.
        Here we only check the two places the README states the number agree.
        """
        text = self.text(self.readme)
        quoted = set(re.findall(r"(\d+) of \1 values reproduce", text))
        quoted |= set(re.findall(r"all (\d+) published values", text))
        self.checks += 1
        if len(quoted) > 1:
            self.fail(f"the README quotes more than one count of reproduced "
                      f"values: {', '.join(sorted(quoted))}")

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
