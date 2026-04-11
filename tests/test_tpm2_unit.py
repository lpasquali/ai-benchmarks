# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch
from rune_bench.attestation.tpm2 import TPM2Driver

def test_tpm2_driver_init_tools_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="tpm2-tools binaries not found"):
            TPM2Driver()

def test_tpm2_driver_init_with_policy():
    with patch("shutil.which", return_value="/bin/tpm2_tools"):
        driver = TPM2Driver(pcr_policy_path="p")
        assert driver.pcr_policy_path == "p"

def test_tpm2_quote_timeout():
    from subprocess import TimeoutExpired
    with patch("shutil.which", return_value="/bin/tpm2_quote"):
        driver = TPM2Driver()
        with patch("subprocess.run", side_effect=TimeoutExpired(cmd=["tpm2_quote"], timeout=5)):
            res = driver.verify("t1")
            assert not res.passed
            assert "tpm2_quote timed out" in res.message

def test_tpm2_quote_disappeared():
    with patch("shutil.which", return_value="/bin/tpm2_quote"):
        driver = TPM2Driver()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="tpm2_quote binary disappeared"):
                driver.verify("t1")

def test_tpm2_quote_failure():
    from subprocess import CompletedProcess
    with patch("shutil.which", return_value="/bin/tpm2_quote"):
        driver = TPM2Driver()
        with patch("subprocess.run", return_value=CompletedProcess(["tpm2_quote"], 1, stdout="", stderr="failed")):
            res = driver.verify("t1")
            assert not res.passed
            assert "rc=1" in res.message

def test_tpm2_checkquote_timeout(tmp_path):
    from subprocess import TimeoutExpired, CompletedProcess
    policy = tmp_path / "policy"
    policy.write_text("dummy")
    
    with patch("shutil.which", return_value="/bin/tpm2_tools"):
        driver = TPM2Driver(pcr_policy_path=policy)
        
        def mock_run(cmd, **kwargs):
            if "tpm2_quote" in cmd:
                return CompletedProcess(cmd, 0, stdout="pcr_digest: abc", stderr="")
            if "tpm2_checkquote" in cmd:
                raise TimeoutExpired(cmd=cmd, timeout=5)
            return CompletedProcess(cmd, 0)

        with patch("subprocess.run", side_effect=mock_run):
            res = driver.verify("t1")
            assert not res.passed
            assert "tpm2_checkquote timed out" in res.message

def test_tpm2_checkquote_disappeared(tmp_path):
    from subprocess import CompletedProcess
    policy = tmp_path / "policy"
    policy.write_text("dummy")
    
    with patch("shutil.which", return_value="/bin/tpm2_tools"):
        driver = TPM2Driver(pcr_policy_path=policy)
        
        def mock_run(cmd, **kwargs):
            if "tpm2_quote" in cmd:
                return CompletedProcess(cmd, 0, stdout="pcr_digest: abc", stderr="")
            if "tpm2_checkquote" in cmd:
                raise FileNotFoundError()
            return CompletedProcess(cmd, 0)

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(RuntimeError, match="tpm2_checkquote binary disappeared"):
                driver.verify("t1")

def test_tpm2_checkquote_failure(tmp_path):
    from subprocess import CompletedProcess
    policy = tmp_path / "policy"
    policy.write_text("dummy")
    
    with patch("shutil.which", return_value="/bin/tpm2_tools"):
        driver = TPM2Driver(pcr_policy_path=policy)
        
        def mock_run(cmd, **kwargs):
            if "tpm2_quote" in cmd:
                return CompletedProcess(cmd, 0, stdout="pcr_digest: abc", stderr="")
            if "tpm2_checkquote" in cmd:
                return CompletedProcess(cmd, 1, stdout="", stderr="mismatch")
            return CompletedProcess(cmd, 0)

        with patch("subprocess.run", side_effect=mock_run):
            res = driver.verify("t1")
            assert not res.passed
            assert "mismatch" in res.message

def test_tpm2_verify_success_no_policy():
    from subprocess import CompletedProcess
    with patch("shutil.which", return_value="/bin/tpm2_quote"):
        driver = TPM2Driver()
        with patch("subprocess.run", return_value=CompletedProcess(["tpm2_quote"], 0, stdout="abc", stderr="")):
            res = driver.verify("t1")
            assert res.passed
            assert res.pcr_digest == "abc"
