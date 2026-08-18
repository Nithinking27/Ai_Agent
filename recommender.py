import json


def load_catalogue(path):
    with open(path, "r") as f:
        return json.load(f)


def load_profiles(path):
    with open(path, "r") as f:
        return json.load(f)


def get_skill_to_course(courses):
    skill_course_map = {}
    for course in courses:
        for skill in course["skills_taught"]:
            if skill not in skill_course_map:
                skill_course_map[skill] = course
    return skill_course_map


def build_learning_path(catalogue, known_skills, goal_key):
    goals = catalogue["goals"]
    courses = catalogue["courses"]

    if goal_key not in goals:
        return [], "Unknown goal '" + goal_key + "'. Valid goals: " + str(list(goals.keys()))

    required_skills = goals[goal_key]["required_skills"]
    skill_to_course = get_skill_to_course(courses)

    known = set(known_skills)
    course_order = []
    already_added = set()

    def add_skill(skill):
        if skill in known:
            return

        course = skill_to_course.get(skill)
        if course is None:
            return

        if course["id"] in already_added:
            return

        for prereq in course["prerequisites"]:
            add_skill(prereq)

        already_added.add(course["id"])
        course_order.append(course)
        known.add(skill)

    missing_skills = [s for s in required_skills if s not in known]

    if len(missing_skills) == 0:
        return [], None

    for skill in missing_skills:
        add_skill(skill)

    return course_order, None


def path_summary(course_list):
    total_weeks = 0
    course_ids = []
    for c in course_list:
        total_weeks += c["duration_weeks"]
        course_ids.append(c["id"])

    return {
        "num_courses": len(course_list),
        "total_duration_weeks": total_weeks,
        "course_ids_in_order": course_ids,
    }
