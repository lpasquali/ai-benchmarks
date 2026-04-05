"""PagerDuty AI agentic runner — delegates to the pagerduty driver.

Scope:      SRE  |  Rank 3  |  Rating 4.5
Capability: Autonomous alert correlation and triage automation.
Docs:       https://support.pagerduty.com/
            https://developer.pagerduty.com/api-reference/
Ecosystem:  LSF Security Standards

This module re-exports :class:`~rune_bench.drivers.pagerduty.PagerDutyDriverClient`
as ``PagerDutyAIRunner`` so existing call-sites continue to work unchanged.
"""

from rune_bench.drivers.pagerduty import PagerDutyDriverClient

PagerDutyAIRunner = PagerDutyDriverClient
