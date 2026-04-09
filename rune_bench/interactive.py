# SPDX-License-Identifier: Apache-2.0
"""Manager for interactive Manual/Browser driver sessions."""

import threading
import time

class InteractiveSessionManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_prompts = {}  # job_id -> dict
        self.responses = {}  # job_id -> dict
        self.conditions = {} # job_id -> Condition

    def request_input(self, job_id: str, prompt_data: dict, timeout: int = 3600) -> dict:
        with self.lock:
            self.pending_prompts[job_id] = prompt_data
            self.responses.pop(job_id, None)
            if job_id not in self.conditions:
                self.conditions[job_id] = threading.Condition(self.lock)
            cond = self.conditions[job_id]

        with cond:
            # Wait for response
            start = time.time()
            while job_id not in self.responses:
                if time.time() - start >= timeout:
                    self.pending_prompts.pop(job_id, None)
                    raise TimeoutError(f"Interactive session timed out for job {job_id}")
                cond.wait(timeout=min(1.0, timeout))
            
            self.pending_prompts.pop(job_id, None)
            resp = self.responses.pop(job_id)
            return resp

    def provide_input(self, job_id: str, response_data: dict) -> None:
        with self.lock:
            if job_id not in self.pending_prompts:
                raise ValueError(f"No pending prompt for job {job_id}")
            self.responses[job_id] = response_data
            if job_id in self.conditions:
                self.conditions[job_id].notify_all()

    def get_pending_prompt(self, job_id: str) -> dict | None:
        with self.lock:
            return self.pending_prompts.get(job_id)

session_manager = InteractiveSessionManager()
