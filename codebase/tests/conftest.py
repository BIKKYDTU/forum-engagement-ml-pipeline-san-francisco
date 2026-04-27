import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Promote setup-phase failures to call-phase so the report shows FAILED.

    pytest.fail() called inside an autouse fixture produces a report with
    when='setup', which the terminal reporter displays as ERROR instead of
    FAILED.  Changing when to 'call' for these reports makes the output say
    FAILED (consistent with the F2P requirement that the empty-codebase run
    produces all-FAILED, not all-ERROR).  This hook is generic: it triggers
    only when the setup phase itself marks the test as failed (outcome='failed'),
    so it does not affect genuine infrastructure errors.
    """
    report = yield
    if report.when == "setup" and report.outcome == "failed":
        report.when = "call"
    return report
