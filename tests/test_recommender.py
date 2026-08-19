import os
import sys
import unittest

# Make the project root importable when tests are run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recommender import build_learning_path, path_summary, load_catalogue, load_profiles


CATALOGUE_PATH = os.path.join(PROJECT_ROOT, "data", "course_catalogue.json")
PROFILES_PATH = os.path.join(PROJECT_ROOT, "data", "student_profiles.json")


class TestCourseRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalogue = load_catalogue(CATALOGUE_PATH)
        cls.profiles = load_profiles(PROFILES_PATH)

    def test_valid_goal_returns_learning_path(self):
        """A valid goal should produce courses when skills are missing."""
        path, error = build_learning_path(
            self.catalogue,
            known_skills=[],
            goal_key="data_scientist",
        )

        self.assertIsNone(error)
        self.assertGreater(len(path), 0)

    def test_prerequisites_are_respected(self):
        """Every prerequisite course must appear before the course that needs it."""
        path, error = build_learning_path(
            self.catalogue,
            known_skills=[],
            goal_key="data_scientist",
        )

        self.assertIsNone(error)
        positions = {course["id"]: i for i, course in enumerate(path)}

        for course in path:
            for prereq_skill in course["prerequisites"]:
                prereq_course = next(
                    (
                        c for c in self.catalogue["courses"]
                        if prereq_skill in c["skills_taught"]
                    ),
                    None,
                )
                self.assertIsNotNone(prereq_course)
                self.assertIn(prereq_course["id"], positions)
                self.assertLess(
                    positions[prereq_course["id"]],
                    positions[course["id"]],
                    msg=f"{prereq_course['id']} must come before {course['id']}",
                )

    def test_known_skills_are_not_recommended_again(self):
        """Courses for skills the student already knows should be skipped."""
        path, error = build_learning_path(
            self.catalogue,
            known_skills=["python"],
            goal_key="ml_engineer",
        )

        self.assertIsNone(error)
        course_ids = [course["id"] for course in path]
        self.assertNotIn("C001", course_ids)

    def test_unknown_goal_returns_error(self):
        """An invalid goal should fail gracefully instead of crashing."""
        path, error = build_learning_path(
            self.catalogue,
            known_skills=[],
            goal_key="does_not_exist",
        )

        self.assertEqual(path, [])
        self.assertIsNotNone(error)
        self.assertIn("Unknown goal", error)

    def test_path_summary_is_correct(self):
        """The summary should correctly count courses and total duration."""
        path, error = build_learning_path(
            self.catalogue,
            known_skills=[],
            goal_key="frontend_developer",
        )

        self.assertIsNone(error)

        summary = path_summary(path)

        self.assertEqual(summary["num_courses"], len(path))
        self.assertEqual(
            summary["total_duration_weeks"],
            sum(course["duration_weeks"] for course in path),
        )
        self.assertEqual(
            summary["course_ids_in_order"],
            [course["id"] for course in path],
        )


if __name__ == "__main__":
    unittest.main()
