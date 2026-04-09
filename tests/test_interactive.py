import pytest
import threading
import time
from rune_bench.interactive import InteractiveSessionManager

def test_interactive_session_manager():
    mgr = InteractiveSessionManager()
    
    # Test provide input before request
    with pytest.raises(ValueError):
        mgr.provide_input("job1", {"ans": 1})
        
    prompt = {"action": "test"}
    
    def provide_later():
        time.sleep(0.1)
        mgr.provide_input("job1", {"ans": 2})
        
    threading.Thread(target=provide_later).start()
    
    res = mgr.request_input("job1", prompt, timeout=1)
    assert res == {"ans": 2}
    
    # Test timeout
    with pytest.raises(TimeoutError):
        mgr.request_input("job2", prompt, timeout=0)

    # Test get pending
    mgr.pending_prompts["job3"] = {"p": 1}
    assert mgr.get_pending_prompt("job3") == {"p": 1}
