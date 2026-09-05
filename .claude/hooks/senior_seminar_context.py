#!/usr/bin/env python3
"""SessionStart hook: surface the Windows-side Senior Seminar School repo
context (research notes + assignment folders) so a session in this WSL
project always sees the latest scope before continuing implementation work.
"""
import json
import os

SENIOR_SEMINAR_DIR = "/mnt/c/Users/eriol/Desktop/School/Senior Seminar"
NOTES_FILE = os.path.join(SENIOR_SEMINAR_DIR, "Project Technical Notes.md")


def build_context() -> str:
    if not os.path.isdir("/mnt/c"):
        return (
            "WARNING: Windows drive not mounted at /mnt/c - could not check "
            "the Senior Seminar folder in the School repo for changes to the "
            "project's research design/scope. Project Technical Notes.md may "
            "be out of date relative to what you're working from."
        )

    if os.path.isfile(NOTES_FILE):
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            notes_content = f.read()
    else:
        notes_content = f"(not found at expected path: {NOTES_FILE})"

    if os.path.isdir(SENIOR_SEMINAR_DIR):
        folders = sorted(
            name
            for name in os.listdir(SENIOR_SEMINAR_DIR)
            if os.path.isdir(os.path.join(SENIOR_SEMINAR_DIR, name))
        )
        assignment_listing = "\n".join(folders) if folders else "(no assignment folders found)"
    else:
        assignment_listing = f"(Senior Seminar folder not found at expected path: {SENIOR_SEMINAR_DIR})"

    return (
        "Senior Seminar project context from the Windows School repo "
        "(re-read fresh this session — check for changes to research "
        "design/scope or new assignment folders before continuing "
        "implementation work):\n\n"
        "=== Project Technical Notes.md ===\n"
        f"{notes_content}\n\n"
        "=== Assignment folders under Senior Seminar/ ===\n"
        f"{assignment_listing}\n"
    )


result = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": build_context(),
    }
}
print(json.dumps(result))
