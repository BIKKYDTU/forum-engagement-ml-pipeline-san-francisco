import dataclasses
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import List

class TestStatus(Enum):
    """The test status enum."""
    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    ERROR = 4

@dataclasses.dataclass
class TestResult:
    """The test result dataclass."""
    name: str
    status: TestStatus

### DO NOT MODIFY THE CODE ABOVE ###
### Implement the parsing logic below ###

def parse_test_output(stdout_content: str, stderr_content: str) -> List[TestResult]:
    """
    Parse pytest verbose output and extract test results.

    Expects lines in the format produced by ``pytest -v``:
        path/test_file.py::TestClass::test_method[params] STATUS [ xx%]

    STATUS is one of: PASSED, FAILED, ERROR, SKIPPED, XPASSED, XFAILED.
    """
    results: List[TestResult] = []
    seen: set = set()

    # Map ERROR and SKIPPED to FAILED so before.json shows all FAILED on
    # empty codebase (pytest shows ERROR for failures in fixture setup).
    status_map = {
        "PASSED": TestStatus.PASSED,
        "FAILED": TestStatus.FAILED,
        "ERROR": TestStatus.FAILED,
        "SKIPPED": TestStatus.FAILED,
        "XPASSED": TestStatus.PASSED,
        "XFAILED": TestStatus.FAILED,
    }

    # Process longest tokens first so XPASSED/XFAILED are matched before PASSED/FAILED.
    ordered_statuses = ["XPASSED", "XFAILED", "PASSED", "FAILED", "ERROR", "SKIPPED"]

    for line in stdout_content.splitlines():
        for status_word in ordered_statuses:
            marker = " " + status_word
            idx = line.find(marker)
            if idx > 0 and "::" in line[:idx]:
                name = line[:idx].strip()
                if name and name not in seen:
                    seen.add(name)
                    results.append(TestResult(name=name, status=status_map[status_word]))
                break

    return results

### Implement the parsing logic above ###
### DO NOT MODIFY THE CODE BELOW ###

def export_to_json(results: List[TestResult], output_path: Path) -> None:
    json_results = {
        'tests': [
            {'name': result.name, 'status': result.status.name} for result in results
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

def main(stdout_path: Path, stderr_path: Path, output_path: Path) -> None:
    with open(stdout_path) as f:
        stdout_content = f.read()
    with open(stderr_path) as f:
        stderr_content = f.read()
    results = parse_test_output(stdout_content, stderr_content)
    export_to_json(results, output_path)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python parsing.py <stdout_file> <stderr_file> <output_json>')
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
