from typing import Any, Optional
from pydantic import BaseModel


class Tests(BaseModel):
    tests: dict[str, str]
    operation: str = "generate"
    gen_model: Optional[str] = None
    gen_date: Optional[str] = None
    exec_stats: Optional[dict[str, Any]] = None

    def add(self, test_id: str, test: str):
        """Add a test to the tests"""
        self.tests[test_id] = test

    def update_stats(self, stats: dict[str, Any]):
        """Update the execution stats of the tests"""
        self.exec_stats = stats
