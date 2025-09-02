import numpy as np

from pa_core.agents.base import BaseAgent
from pa_core.agents.registry import build_all, register_agent
from pa_core.agents.types import AgentParams
from pa_core.sim.metrics import register_metric, summary_table


class DummyAgent(BaseAgent):
    def run(self, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act):
        return {"Dummy": r_beta}


def test_register_agent_allows_construction() -> None:
    register_agent("Dummy", DummyAgent)
    agents = build_all([AgentParams("Dummy", 1.0, 0.0, 0.0, {})])
    assert any(isinstance(a, DummyAgent) for a in agents)


def test_register_metric_included_in_summary() -> None:
    def mean_return(arr: np.ndarray) -> float:
        return float(np.mean(arr))

    register_metric("Mean", mean_return)
    data = {"A": np.array([[0.1, 0.0], [0.0, 0.1]])}
    df = summary_table(data)
    assert "Mean" in df.columns
