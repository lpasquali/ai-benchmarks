# SPDX-License-Identifier: Apache-2.0
import pytest

from rune_bench.agents.experimental.safety_interceptor import (
    SafetyInterceptor,
    SafetyViolation,
)


def test_safety_interceptor_safe_command():
    interceptor = SafetyInterceptor()

    # Safe read-only command
    assert interceptor.evaluate("shell", {"command": "ls -la /var/log"}) is True
    assert (
        interceptor.evaluate("kubectl", {"command": "get pods -n kube-system"}) is True
    )


def test_safety_interceptor_destructive_commands():
    interceptor = SafetyInterceptor()

    # rm -rf
    with pytest.raises(SafetyViolation, match="matches destructive pattern"):
        interceptor.evaluate("shell", {"command": "rm -rf /tmp/data"})

    # chmod 777
    with pytest.raises(SafetyViolation):
        interceptor.evaluate("shell", {"command": "chmod -R 777 /etc"})

    # Destructive IAM
    with pytest.raises(SafetyViolation):
        interceptor.evaluate("aws", {"command": "iam delete-user --user-name bob"})

    # Kubernetes NS deletion
    with pytest.raises(SafetyViolation):
        interceptor.evaluate("kubectl", {"command": "delete namespace prod"})


def test_safety_interceptor_whitelist():
    interceptor = SafetyInterceptor()

    # Normally blocked
    with pytest.raises(SafetyViolation):
        interceptor.evaluate("shell", {"command": "rm -rf /tmp/cache"})

    # Whitelist the 'shell' tool outright
    interceptor.add_whitelist("shell")

    # Now it should pass because the tool itself is whitelisted
    assert interceptor.evaluate("shell", {"command": "rm -rf /tmp/cache"}) is True
