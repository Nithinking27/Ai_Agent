import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent import clean_json_text, chat_answer, recommend_for_profile
from recommender import load_catalogue


CATALOGUE_PATH = os.path.join(PROJECT_ROOT, "data", "course_catalogue.json")


class TestAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalogue = load_catalogue(CATALOGUE_PATH)

    def test_clean_json_text_removes_markdown_fences(self):
        """LLM JSON wrapped in a code block should be parsed cleanly."""
        raw = '```json\n{"course_reasons": {}}\n```'
        cleaned = clean_json_text(raw)

        self.assertEqual(cleaned, '{"course_reasons": {}}')

    def test_recommendation_result_contains_ordered_path(self):
        """The agent result should expose numbered course steps."""
        profile = {
            "student_id": "TEST",
            "name": "Test Student",
            "background": "Beginner",
            "goal": "frontend_developer",
            "known_skills": [],
        }

        result = recommend_for_profile(self.catalogue, profile, verbose=False)

        self.assertNotIn("error", result)
        self.assertGreater(len(result["learning_path"]), 0)

        steps = [item["step"] for item in result["learning_path"]]
        self.assertEqual(steps, list(range(1, len(steps) + 1)))

    def test_chat_fallback_recommends_without_api_key(self):
        """The chat fallback should still provide a catalogue-backed path."""
        profile = {
            "student_id": "TEST",
            "name": "Test Student",
            "background": "Beginner",
            "goal": "frontend_developer",
            "known_skills": [],
        }

        old_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            answer, source = chat_answer(
                self.catalogue,
                profile,
                history=[],
                user_text="What should I learn next?",
            )
        finally:
            if old_key is not None:
                os.environ["GROQ_API_KEY"] = old_key

        self.assertEqual(source, "fallback")
        self.assertIn("recommend", answer.lower())
        self.assertIn("HTML & CSS Fundamentals", answer)


if __name__ == "__main__":
    unittest.main()
