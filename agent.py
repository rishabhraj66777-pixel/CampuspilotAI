import os
import re
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from uagents import Agent, Context, Protocol

from uagents.experimental.chat_agent.protocol import (
    build_llm_message_history,
)

from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)


# ==========================================
# CAMPUSPILOTAI - STUDENT PRODUCTIVITY AGENT
# ==========================================

agent = Agent()
chat_protocol = Protocol(spec=chat_protocol_spec)

client = OpenAI(
    base_url=os.environ["ASI1_BASE_URL"],
    api_key=os.environ["ASI1_API_KEY"],
)

SYSTEM_PROMPT = """
You are CampusPilotAI, a helpful student productivity agent.

Your responsibilities:
1. Create practical study timetables.
2. Organize subjects and exam preparation.
3. Break large assignments into smaller tasks.
4. Calculate available study hours.
5. Recommend revision and break schedules.
6. Explain college subjects in simple language.

Rules:
- Be friendly, clear and concise.
- Never invent exam dates or syllabus details.
- Ask questions if essential information is missing.
- Respect the student's available study time.
- Use the provided study-planning tools when appropriate.
- Present plans in readable tables or numbered lists.
"""


# ==========================================
# TOOL 1: CALCULATE AVAILABLE STUDY HOURS
# ==========================================

def calculate_study_hours(days: int, hours_per_day: float) -> dict:
    days = max(1, min(days, 365))
    hours_per_day = max(0, min(hours_per_day, 24))

    return {
        "days": days,
        "hours_per_day": hours_per_day,
        "total_hours": days * hours_per_day,
    }


# ==========================================
# TOOL 2: GENERATE STUDY PLAN
# ==========================================

def generate_study_plan(
    subjects: list[str],
    days: int,
    hours_per_day: float,
) -> dict:

    result = calculate_study_hours(days, hours_per_day)

    subjects = [
        subject.strip()
        for subject in subjects
        if subject.strip()
    ]

    if not subjects:
        subjects = ["Subject 1", "Subject 2", "Subject 3"]

    timetable = []

    for day in range(1, result["days"] + 1):
        subject = subjects[(day - 1) % len(subjects)]

        if day == result["days"]:
            activity = "Final revision and practice questions"
        elif day % 3 == 0:
            activity = "Revision and self-test"
        else:
            activity = f"Study {subject} and solve practice questions"

        timetable.append({
            "day": day,
            "subject": subject,
            "activity": activity,
            "hours": result["hours_per_day"],
        })

    return {
        "total_study_hours": result["total_hours"],
        "subjects": subjects,
        "timetable": timetable,
    }


# ==========================================
# TOOL 3: BREAK ASSIGNMENTS INTO TASKS
# ==========================================

def break_into_tasks(assignment: str) -> list[str]:
    assignment = assignment.strip()

    if not assignment:
        return ["Define the assignment topic"]

    return [
        f"Step 1: Understand the requirements of {assignment}",
        f"Step 2: Collect notes and study material for {assignment}",
        f"Step 3: Complete the main work for {assignment}",
        f"Step 4: Review, correct and finalize {assignment}",
    ]


# ==========================================
# TOOL ROUTER
# ==========================================

def run_student_tool(user_text: str) -> str:
    text = user_text.lower()

    # Assignment task breakdown
    if any(word in text for word in [
        "assignment", "break into tasks", "break down", "project tasks"
    ]):
        tasks = break_into_tasks(user_text)

        return (
            "TASK BREAKDOWN TOOL EXECUTED\n"
            + "\n".join(tasks)
        )

    # Study timetable generator
    if any(word in text for word in [
        "study plan", "study timetable", "study schedule",
        "exam preparation", "timetable"
    ]):
        day_match = re.search(r"(\d+)\s*days?", text)
        hour_match = re.search(
            r"(\d+(?:\.\d+)?)\s*hours?\s*(?:daily|per day|a day|each day)",
            text,
        )

        days = int(day_match.group(1)) if day_match else 7
        hours = float(hour_match.group(1)) if hour_match else 4.0

        subjects = []

        match = re.search(
            r"(?:subjects?\s*(?:are|:)?\s*)(.*?)(?:\.|;|$)",
            user_text,
            re.IGNORECASE,
        )

        if match:
            subjects = [
                item.strip()
                for item in re.split(r",| and ", match.group(1))
                if item.strip()
            ]

        plan = generate_study_plan(subjects, days, hours)

        lines = [
            "STUDY PLAN TOOL EXECUTED",
            f"Subjects: {', '.join(plan['subjects'])}",
            f"Total available study time: {plan['total_study_hours']} hours",
            "",
            "DAY-WISE PLAN:",
        ]

        for item in plan["timetable"]:
            lines.append(
                f"Day {item['day']}: {item['activity']} "
                f"({item['hours']} hours)"
            )

        lines.append(
            "\nTake a 5–10 minute break after each focused study session."
        )

        return "\n".join(lines)

    return ""


# ==========================================
# CHAT MESSAGE HELPER
# ==========================================

def create_text_chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=text),
            EndSessionContent(type="end-session"),
        ],
    )


# ==========================================
# MAIN ACP CHAT HANDLER
# ==========================================

@chat_protocol.on_message(ChatMessage)
async def handle_message(
    ctx: Context,
    sender: str,
    msg: ChatMessage,
):
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    user_text = msg.text()

    if not user_text:
        return

    ctx.logger.info(f"User request received: {user_text}")

    # Step 1: Check whether a student productivity tool is needed.
    tool_result = run_student_tool(user_text)

    # Step 2: Combine tool results with the AI's reasoning.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *build_llm_message_history(ctx),
    ]

    if tool_result:
        messages.append({
            "role": "system",
            "content": (
                "A student productivity tool was executed. "
                "Use its actual output to answer the user accurately. "
                "Do not contradict its calculations.\n\n"
                + tool_result
            ),
        })

    # Step 3: Generate the final student-friendly response.
    try:
        response = client.chat.completions.create(
            model="asi1",
            messages=messages,
            max_tokens=900,
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = tool_result or (
                "Tell me your subjects, exam dates and available study hours."
            )

    except Exception:
        ctx.logger.exception("ASI:One request failed")

        answer = tool_result or (
            "I could not reach the AI service right now. "
            "Please try again shortly."
        )

    await ctx.send(sender, create_text_chat(str(answer)))


@chat_protocol.on_message(ChatAcknowledgement)
async def handle_acknowledgement(
    ctx: Context,
    sender: str,
    msg: ChatAcknowledgement,
):
    pass


# Publish the chat protocol manifest for discovery.
agent.include(chat_protocol, publish_manifest=True)


if __name__ == "__main__":
    agent.run()
