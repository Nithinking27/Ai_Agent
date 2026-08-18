import argparse
import json
import os
import sys
from typing import Any, Dict, List

from recommender import load_catalogue, load_profiles, build_learning_path, path_summary

CATALOGUE_PATH = os.path.join("data", "course_catalogue.json")
PROFILES_PATH = os.path.join("data", "student_profiles.json")
OUTPUT_DIR = "sample_output"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a friendly AI Course Recommendation Agent.
You help students choose courses from a fixed course catalogue.

You have access to:
- the student's current profile and known skills
- the available goals
- the complete course catalogue
- a deterministic recommendation path when one has already been computed

IMPORTANT RULES:
1. Never invent a course, course ID, skill, goal, or catalogue fact.
2. When recommending courses, use only courses from the supplied catalogue.
3. Respect prerequisites. Do not recommend a course before its prerequisites.
4. If a deterministic learning path is supplied, do not change its order or membership.
5. You may answer general career/learning questions using your knowledge, but clearly distinguish general advice from catalogue-backed recommendations.
6. If the user asks what they should learn next, use their known skills and goal to recommend from the catalogue.
7. If the user asks why a course is recommended, explain using the student's profile, course skills, and prerequisites.
8. Be concise, practical, and friendly.
"""


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as exc:
        print(f"[warning] Could not initialize Groq: {exc}", file=sys.stderr)
        return None


def call_groq(messages: List[Dict[str, str]], max_tokens: int = 1200) -> str:
    client = get_client()
    if client is None:
        return ""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.35,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[warning] Groq call failed: {exc}", file=sys.stderr)
        return ""


def make_simple_rationale(profile, goal_label, ordered_courses):
    reasons = {}
    for c in ordered_courses:
        if c["prerequisites"]:
            prereq_text = " It builds on " + ", ".join(c["prerequisites"]) + "."
        else:
            prereq_text = " It has no prerequisites, so it is a good starting point."
        reasons[c["id"]] = (
            "Teaches " + ", ".join(c["skills_taught"]) +
            ", which you need for " + goal_label + "." + prereq_text
        )
    narrative = (
        "Based on " + profile["name"] + "'s background (" + profile["background"] +
        "), this path covers the missing skills needed for " + goal_label + " in prerequisite order."
    )
    return {"overall_narrative": narrative, "course_reasons": reasons}


def call_groq_for_rationale(profile, goal_label, ordered_courses):
    course_info = [
        {
            "id": c["id"],
            "name": c["name"],
            "level": c["level"],
            "skills_taught": c["skills_taught"],
            "prerequisites": c["prerequisites"],
            "description": c["description"],
        }
        for c in ordered_courses
    ]
    user_message = json.dumps({
        "student": profile,
        "goal": goal_label,
        "ordered_courses": course_info,
        "task": "Explain why each already-selected course fits this student. Do not change the path.",
    })
    raw = call_groq([
        {"role": "system", "content": SYSTEM_PROMPT + "\nFor this request, return ONLY valid JSON with keys overall_narrative and course_reasons."},
        {"role": "user", "content": user_message},
    ])
    if not raw:
        return make_simple_rationale(profile, goal_label, ordered_courses), f"fallback (Groq unavailable; configured model: {MODEL})"
    try:
        rationale = json.loads(clean_json_text(raw))
        if not isinstance(rationale, dict):
            raise ValueError("LLM response was not a JSON object")

        # Groq models may return course_reasons either as a mapping:
        # {"C001": "reason", ...} or as a list of objects such as:
        # [{"course_id": "C001", "reason": "..."}, ...]. Normalize both
        # formats so the rest of the application has one predictable shape.
        course_reasons = rationale.get("course_reasons", {})
        if isinstance(course_reasons, list):
            normalized = {}
            for item in course_reasons:
                if isinstance(item, dict):
                    course_id = item.get("course_id") or item.get("id") or item.get("course")
                    reason = item.get("reason") or item.get("explanation") or item.get("rationale")
                    if course_id and reason:
                        normalized[str(course_id)] = str(reason)
            rationale["course_reasons"] = normalized
        elif isinstance(course_reasons, dict):
            rationale["course_reasons"] = {str(k): str(v) for k, v in course_reasons.items()}
        else:
            rationale["course_reasons"] = {}

        return rationale, f"groq ({MODEL})"
    except Exception as exc:
        print(f"[warning] Invalid rationale JSON from Groq: {exc}", file=sys.stderr)
        return make_simple_rationale(profile, goal_label, ordered_courses), f"fallback (invalid Groq response; model: {MODEL})"


def recommend_for_profile(catalogue, profile, verbose=True):
    goal_key = profile["goal"]
    goal_label = catalogue["goals"].get(goal_key, {}).get("label", goal_key)
    ordered_courses, error = build_learning_path(catalogue, profile["known_skills"], goal_key)
    if error:
        return {"student_id": profile.get("student_id"), "name": profile.get("name"), "error": error}

    if not ordered_courses:
        result = {
            "student_id": profile.get("student_id"),
            "name": profile.get("name"),
            "goal": goal_label,
            "message": "This student already has all the skills needed for this goal.",
            "learning_path": [],
        }
        if verbose:
            print(f"\n=== {profile.get('name')} ({profile.get('student_id')}) -> {goal_label} ===")
            print(result["message"])
        return result

    rationale, source = call_groq_for_rationale(profile, goal_label, ordered_courses)
    path = []
    for step_num, c in enumerate(ordered_courses, start=1):
        path.append({
            "step": step_num,
            "course_id": c["id"],
            "course_name": c["name"],
            "level": c["level"],
            "duration_weeks": c["duration_weeks"],
            "skills_gained": c["skills_taught"],
            "reason": rationale.get("course_reasons", {}).get(c["id"], "No reason generated."),
        })

    summary = path_summary(ordered_courses)
    result = {
        "student_id": profile.get("student_id"),
        "name": profile.get("name"),
        "background": profile.get("background"),
        "known_skills": profile.get("known_skills", []),
        "goal": goal_label,
        "overall_narrative": rationale.get("overall_narrative", ""),
        "learning_path": path,
        "summary": summary,
        "rationale_source": source,
    }

    if verbose:
        print_result(result)
    return result


def print_result(result):
    if result.get("error"):
        print("Error:", result["error"])
        return
    print(f"\n=== {result.get('name')} ({result.get('student_id')}) -> {result.get('goal')} ===")
    print(result.get("overall_narrative", result.get("message", "")))
    for step in result.get("learning_path", []):
        print(f"  {step['step']}. [{step['course_id']}] {step['course_name']} ({step['level']}, {step['duration_weeks']}w)")
        print(f"     Reason: {step['reason']}")
    if result.get("summary"):
        s = result["summary"]
        print(f"  Total: {s['num_courses']} courses, {s['total_duration_weeks']} weeks")


def build_context(catalogue, profile, current_path=None):
    compact_courses = []
    for c in catalogue["courses"]:
        compact_courses.append({
            "id": c["id"], "name": c["name"], "level": c["level"],
            "duration_weeks": c["duration_weeks"], "skills_taught": c["skills_taught"],
            "prerequisites": c["prerequisites"], "description": c["description"],
        })
    return {
        "student_profile": profile,
        "goals": catalogue["goals"],
        "course_catalogue": compact_courses,
        "deterministic_learning_path": current_path or [],
    }


def chat_answer(catalogue, profile, history, user_text):
    # Always compute the current catalogue-backed path so recommendation questions
    # are grounded in the same deterministic engine used by --all.
    path, error = build_learning_path(catalogue, profile["known_skills"], profile["goal"])
    path_data = [
        {"id": c["id"], "name": c["name"], "level": c["level"],
         "duration_weeks": c["duration_weeks"], "skills_taught": c["skills_taught"],
         "prerequisites": c["prerequisites"]}
        for c in path
    ] if not error else []

    context = build_context(catalogue, profile, path_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Grounding data:\n" + json.dumps(context)},
    ]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_text})

    answer = call_groq(messages, max_tokens=1400)
    if answer:
        return answer, f"groq ({MODEL})"

    # Useful fallback when no API key is present.
    lower = user_text.lower()
    if any(word in lower for word in ["recommend", "learn", "course", "path", "next", "study"]):
        if error:
            return f"I couldn't build a recommendation path: {error}", "fallback"
        if not path:
            return "You already have all the skills required for your selected goal according to the current catalogue.", "fallback"
        lines = ["Based on your current profile, I recommend these courses in order:"]
        for i, c in enumerate(path, 1):
            lines.append(f"{i}. {c['name']} ({c['level']}, {c['duration_weeks']} weeks)")
        return "\n".join(lines), "fallback"

    return ("I can answer course and career questions when GROQ_API_KEY is configured. "
            "Try asking what you should learn next, why a course is recommended, or what you can skip.", "fallback")


def run_chat(catalogue, profiles, student_id=None):
    if student_id:
        matches = [p for p in profiles if p["student_id"] == student_id]
        if not matches:
            print(f"No profile found with student_id={student_id}")
            return
        profile = matches[0].copy()
    else:
        print("=== Conversational Course Recommendation Agent ===")
        print("Choose a sample student or create your own profile.")
        for p in profiles:
            print(f"  {p['student_id']}: {p['name']} -> {p['goal']}")
        choice = input("Student ID (or NEW): ").strip()
        if choice.upper() == "NEW":
            name = input("Name: ").strip() or "Student"
            background = input("Background: ").strip() or "Student"
            print("Available goals:")
            for key, value in catalogue["goals"].items():
                print(f"  {key}: {value['label']}")
            goal = input("Goal key: ").strip()
            known = input("Known skills, comma-separated: ").strip()
            profile = {
                "student_id": "CHAT",
                "name": name,
                "background": background,
                "goal": goal,
                "known_skills": [s.strip() for s in known.split(",") if s.strip()],
            }
        else:
            matches = [p for p in profiles if p["student_id"].upper() == choice.upper()]
            if not matches:
                print("Unknown student ID.")
                return
            profile = matches[0].copy()

    print(f"\nChatting as {profile['name']} | Goal: {profile['goal']}")
    print("Ask anything about the learning path. Type 'exit' to quit, 'profile' to view the profile, or 'path' to show the computed path.\n")
    history = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if user_text.lower() == "profile":
            print(json.dumps(profile, indent=2))
            continue
        if user_text.lower() == "path":
            result = recommend_for_profile(catalogue, profile, verbose=False)
            print_result(result)
            continue

        answer, source = chat_answer(catalogue, profile, history, user_text)
        print(f"\nAgent ({source}): {answer}\n")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})


def run_interactive(catalogue):
    print("=== Course Recommendation Agent (profile mode) ===")
    name = input("Student name: ").strip() or "Student"
    background = input("Background: ").strip()
    print("Available goals:", list(catalogue["goals"].keys()))
    goal = input("Goal key: ").strip()
    known = input("Known skills, comma-separated (or leave blank): ").strip()
    profile = {
        "student_id": "INTERACTIVE",
        "name": name,
        "background": background,
        "goal": goal,
        "known_skills": [s.strip() for s in known.split(",") if s.strip()],
    }
    result = recommend_for_profile(catalogue, profile)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "interactive_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to", out_path)


def main():
    parser = argparse.ArgumentParser(description="Conversational Course Recommendation Agent")
    parser.add_argument("--catalogue", default=CATALOGUE_PATH)
    parser.add_argument("--profiles", default=PROFILES_PATH)
    parser.add_argument("--student_id", help="Run for one student ID")
    parser.add_argument("--all", action="store_true", help="Run every sample student")
    parser.add_argument("--interactive", action="store_true", help="Build one profile and generate a path")
    parser.add_argument("--chat", action="store_true", help="Start a conversational AI session")
    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue)
    profiles = load_profiles(args.profiles)

    if args.chat:
        run_chat(catalogue, profiles, args.student_id)
        return
    if args.interactive:
        run_interactive(catalogue)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.student_id:
        profiles = [p for p in profiles if p["student_id"] == args.student_id]
        if not profiles:
            print("No profile found with student_id=" + args.student_id)
            return

    results = []
    for profile in profiles:
        result = recommend_for_profile(catalogue, profile)
        results.append(result)
        out_path = os.path.join(OUTPUT_DIR, profile["student_id"] + "_path.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved " + str(len(results)) + " result(s) to " + OUTPUT_DIR + "/")


if __name__ == "__main__":
    main()
