# Conversational Course Recommendation Agent

A grounded AI course advisor that combines deterministic prerequisite planning with Groq LLM-powered conversation.

## What Makes It an Agent?

This project combines structured decision-making with natural-language interaction:

- Computes a valid learning path from the student's goal, known skills, and course prerequisites.
- Uses the deterministic planner as the source of truth for course recommendations.
- Passes the generated learning path to the LLM as grounding context, reducing the chance of invented courses or prerequisites.
- Maintains short conversation history during a chat session.
- Handles follow-up questions naturally, such as:
  - "What should I learn next?"
  - "Why do I need statistics?"
  - "Can I skip SQL if I already know it?"
  - "Explain my learning path in simple terms."
  - "How long will this path take?"
  - "What is the difference between ML and MLOps?"
- Provides a deterministic fallback response when the Groq API is unavailable.

## Architecture

                    User question
                         |
                         v
              +-----------------------+
              | Conversational Agent  |
              +-----------+-----------+
                          |
              +-----------v-----------+
              | Deterministic Planner |
              | skills + prerequisites|
              +-----------+-----------+
                          |
                    Grounding data
                          |
                          v
                  +---------------+
                  |    Groq LLM   |
                  | gpt-oss-120b  |
                  +-------+-------+
                          |
                          v
                   Natural answer

## Key Design Principle

The LLM is not the source of truth for the learning path.

The Python planner determines the valid course order based on:

1. Student goal
2. Current/known skills
3. Course prerequisites
4. Course catalogue

The resulting path is then provided to the LLM as grounding data.

This keeps recommendations consistent with the catalogue while allowing the agent to explain and discuss the path conversationally.

## Setup on Windows

Open Command Prompt in the project directory:

cd course-recommendation-agent
pip install -r requirements.txt

Set the Groq API key:

set GROQ_API_KEY=your_groq_key_here

Optional model configuration:

set GROQ_MODEL=openai/gpt-oss-120b

## Run the Conversational Agent

python agent.py --chat

Choose a sample student, for example S1.

Then try:

You: What should I learn first?
You: Why do I need statistics?
You: What can I skip if I already know Python?
You: Explain my whole path in simple terms.
You: How many weeks will it take?
You: What is the difference between ML and MLOps?

During the session:

- Type "path" to display the deterministic learning path.
- Type "profile" to display the active student profile.
- Type "exit" to quit.

To start chat directly for a specific student:

python agent.py --chat --student_id S1

## Existing Modes

Run all sample students:

python agent.py --all

Run one student:

python agent.py --student_id S2

Create a profile and generate a learning path:

python agent.py --interactive

## Project Structure

course-recommendation-agent/
├── agent.py
├── recommender.py
├── requirements.txt
├── .env
├── data/
│   ├── course_catalogue.json
│   └── student_profiles.json
└── sample_output/

## Main Components

### agent.py

Handles:

- Conversational interaction
- User input
- Conversation history
- Groq LLM integration
- Grounding context
- LLM responses
- Fallback behavior

### recommender.py

Contains the deterministic recommendation logic.

It evaluates:

- Student's known skills
- Student's target goal
- Course prerequisites
- Course catalogue

It then produces an ordered learning path.

### data/course_catalogue.json

Contains the available courses and their prerequisite relationships.

### data/student_profiles.json

Contains sample student profiles with goals and known skills.

## Agent Flow

Student Profile
      |
      v
Goal + Known Skills
      |
      v
Prerequisite Planner
      |
      v
Ordered Learning Path
      |
      v
Grounding Context
      |
      v
Groq LLM
      |
      v
Conversational Explanation

## Why Is This an Agent?

The system is not simply sending a question directly to an LLM.

It follows a workflow:

1. Understands the student's profile and goal.
2. Uses a deterministic planning component to calculate the required learning path.
3. Uses the generated path as grounding information.
4. Sends the grounded context and user's question to the LLM.
5. Generates a conversational response.
6. Maintains conversation history for follow-up questions.
7. Falls back to deterministic responses if the LLM is unavailable.

This combination of planning, deterministic logic, memory, LLM interaction, and fallback behavior makes the system agent-like rather than a simple chatbot.

## Why Use a Hybrid Approach?

A pure LLM-based recommender could generate plausible-sounding courses or prerequisite relationships that do not exist in the catalogue.

This project instead uses a hybrid architecture:

- Deterministic logic → ensures correct prerequisite planning.
- LLM → provides natural-language conversation and explanations.
- Grounding → keeps the LLM aligned with the actual course catalogue.
- Conversation history → allows contextual follow-up questions.
- Fallback logic → keeps the core recommendation functionality available when the API is unavailable.

## Summary

The Conversational Course Recommendation Agent demonstrates a hybrid AI-agent architecture that combines deterministic planning with an LLM.

The recommendation engine provides a reliable and explainable learning path, while the Groq LLM provides a natural conversational interface for asking questions, understanding prerequisites, and exploring the recommended path.
