from __future__ import annotations

import os
import unittest
from pathlib import Path

from phase8.app.dashboard import format_percent, numeric_frame, saved_metric_rows, source_label


class DashboardHelperTests(unittest.TestCase):
    def test_plain_language_formatters(self):
        self.assertEqual(format_percent(0.55), "55%")
        self.assertEqual(format_percent(None), "Unavailable")
        self.assertEqual(source_label("discovery_skill_prior"), "General learner pattern")

    def test_saved_metric_selection_and_numeric_chart_frame(self):
        overview = {"saved_metrics": [
            {"source": "phase6_decision.csv", "values": {"Metric": "Recall", "Value": "0.1"}},
            {"source": "other.csv", "values": {"Metric": "NDCG", "Value": "0.2"}},
        ]}
        rows = saved_metric_rows(overview, "phase6_decision.csv")
        self.assertEqual(rows[0]["Metric"], "Recall")
        frame = numeric_frame(rows, "Metric", "Value")
        self.assertEqual(frame.iloc[0]["Value"], 0.1)


class StreamlitFailureStateTests(unittest.TestCase):
    def test_dashboard_renders_service_failure_without_exception(self):
        from streamlit.testing.v1 import AppTest

        previous = os.environ.get("PHASE8_API_URL")
        os.environ["PHASE8_API_URL"] = "http://127.0.0.1:9"
        try:
            get_settings = __import__(
                "phase8.app.config", fromlist=["get_settings"]
            ).get_settings
            get_settings.cache_clear()
            dashboard_path = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
            app = AppTest.from_file(dashboard_path, default_timeout=5).run()
            self.assertEqual(len(app.exception), 0)
            self.assertGreaterEqual(len(app.error), 1)
        finally:
            if previous is None:
                os.environ.pop("PHASE8_API_URL", None)
            else:
                os.environ["PHASE8_API_URL"] = previous
            get_settings.cache_clear()

    def test_both_role_views_render_from_contract_fixture(self):
        from streamlit.testing.v1 import AppTest

        script = '''
from phase8.app.dashboard import render_student_view, render_educator_view

class FakeClient:
    def learners(self, **kwargs):
        return [{'learner_id': 'learner-a'}]
    def profile(self, learner_id):
        return {
            'learner_id': learner_id, 'cold_start_problem_history': False,
            'state_source': 'lstm_dkt_early_history',
            'supported_early_interactions': 20,
        }
    def skills(self, learner_id):
        return [{'skill_id': 'skill:1', 'predicted_correctness': 0.42}]
    def recommendations(self, learner_id):
        return [{
            'learner_id': learner_id, 'rank': 1, 'item_id': 'problem:1',
            'problem_label': 'Problem 1', 'primary_skill_id': 'skill:1',
            'skill_label': 'Fractions', 'predicted_correctness': 0.42,
            'state_source': 'lstm_dkt_early_history',
            'route_label': 'similar learners',
            'dashboard_explanation': 'Recommended because similar learners worked on it.',
            'warning_messages': [],
        }]
    def teacher_overview(self):
        return {
            'learner_count': 1, 'recommendation_count': 1,
            'routes': [{'route': 'cf', 'route_label': 'similar learners', 'recommendation_count': 1}],
            'evidence_sources': [{'state_source': 'lstm_dkt_early_history', 'learner_count': 1}],
            'history_groups': [{'cold_start_problem_history': 0, 'learner_count': 1}],
            'warnings': [{'code': 'none', 'message': 'No limitation', 'recommendation_count': 1}],
            'saved_metrics': [
                {'source': 'phase6_decision.csv', 'values': {'SelectedProjectModel': 'gated_hybrid_with_lstm_dkt'}},
                {'source': 'paired_uncertainty.csv', 'values': {'Metric': 'NDCGAt10', 'Lower95': '-0.1', 'Upper95': '0.1'}},
            ],
        }
    def teacher_learners(self, **kwargs):
        return [{
            'learner_id': 'learner-a', 'cold_start_problem_history': False,
            'state_source': 'lstm_dkt_early_history', 'supported_early_interactions': 20,
        }]

client = FakeClient()
render_student_view(client)
render_educator_view(client)
'''
        app = AppTest.from_string(script, default_timeout=8).run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.title), 2)


if __name__ == "__main__":
    unittest.main()
