"""The release SEC gate cannot accept skipped, empty or failed qualification."""
import pytest
from scripts.release.rq03_backend_regression import offline_sec_verdict

@pytest.mark.parametrize('code,output,expected',[
 (0,'242 passed, 11 warnings in 78.5s',True),
 (0,'250 passed in 90.0s',True),
 (0,'241 passed in 1s',False),
 (0,'242 passed, 1 skipped in 1s',False),
 (0,'242 passed, 1 failed in 1s',False),
 (1,'242 passed in 1s',False),
 (0,'no tests ran',False),
 (0,'242 passed, 1 error in 1s',False),
])
def test_mandatory_offline_suite_verdict(code,output,expected):
    assert offline_sec_verdict(code,output)[0] is expected
