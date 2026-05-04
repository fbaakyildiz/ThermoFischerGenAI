import asyncio
import json
import logging
import os

from google import genai as google_genai
from google.genai import types as genai_types

from .rate_limiter import limiter
from .schemas import PatientInput, StewardshipReport, AgentOutput
from .prompts import PROMPTS

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


async def call_gemini(system_prompt: str, user_prompt: str) -> str:
    await limiter.wait()

    client = get_client()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        ),
    )
    raw = response.text

    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        logger.warning("Non-JSON response, retrying with explicit instruction")
        await limiter.wait()
        retry_response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    user_prompt
                    + "\n\nCRITICAL: Your previous response was not valid JSON. "
                    "Respond with ONLY a valid JSON object. No markdown, no code fences, no explanation."
                ),
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            ),
        )
        return retry_response.text


async def run_pipeline(patient: PatientInput) -> StewardshipReport:
    patient_json = patient.model_dump_json(indent=2)

    # A1 — Intake & Validation
    logger.info("Running A1: Intake & Validation")
    a1_raw = await call_gemini(
        PROMPTS["A1_SYSTEM"],
        PROMPTS["A1_USER"].format(patient_json=patient_json),
    )
    a1_data = json.loads(a1_raw)
    a1_out = AgentOutput(
        agent_name="A1_Intake",
        reasoning=a1_data.get("reasoning", ""),
        output=a1_data,
        warnings=a1_data.get("warnings", []),
        needs_clinician=a1_data.get("needs_clinician", False),
    )

    # A2 — Clinical Reasoning
    logger.info("Running A2: Clinical Reasoning")
    a2_raw = await call_gemini(
        PROMPTS["A2_SYSTEM"],
        PROMPTS["A2_USER"].format(patient_json=patient_json, a1_output=a1_raw),
    )
    a2_data = json.loads(a2_raw)
    a2_out = AgentOutput(
        agent_name="A2_Clinical",
        reasoning=a2_data.get("reasoning", ""),
        output=a2_data,
        warnings=a2_data.get("warnings", []),
        needs_clinician=a2_data.get("needs_clinician", False),
    )

    # A3 — Kinetic Analysis & Context
    logger.info("Running A3: Kinetic Analysis")
    a3_raw = await call_gemini(
        PROMPTS["A3_SYSTEM"],
        PROMPTS["A3_USER"].format(patient_json=patient_json, a2_output=a2_raw),
    )
    a3_data = json.loads(a3_raw)
    a3_out = AgentOutput(
        agent_name="A3_Kinetic",
        reasoning=a3_data.get("reasoning", ""),
        output=a3_data,
        warnings=a3_data.get("warnings", []),
        needs_clinician=a3_data.get("needs_clinician", False),
    )

    # A4 — Final Report
    logger.info("Running A4: Final Report")
    a4_raw = await call_gemini(
        PROMPTS["A4_SYSTEM"],
        PROMPTS["A4_USER"].format(
            patient_json=patient_json,
            a1_output=a1_raw,
            a2_output=a2_raw,
            a3_output=a3_raw,
        ),
    )
    a4_data = json.loads(a4_raw)
    a4_out = AgentOutput(
        agent_name="A4_Report",
        reasoning=a4_data.get("reasoning", ""),
        output=a4_data,
        warnings=a4_data.get("warnings", []),
        needs_clinician=a4_data.get("needs_clinician", False),
    )

    # Consolidate warnings (deduplicated)
    all_warnings = []
    seen = set()
    for w in (
        a1_data.get("warnings", [])
        + a2_data.get("warnings", [])
        + a3_data.get("warnings", [])
        + a4_data.get("warnings", [])
    ):
        if w and w not in seen:
            seen.add(w)
            all_warnings.append(w)

    return StewardshipReport(
        patient_summary=a4_data.get("patient_summary", ""),
        pct_interpretation=a4_data.get("pct_interpretation", ""),
        recommendation=a4_data.get("recommendation", "clinician_decision"),
        recommendation_strength=a4_data.get("recommendation_strength", "weak"),
        rationale=a4_data.get("rationale", ""),
        kinetic_analysis=a4_data.get("kinetic_analysis"),
        next_steps=a4_data.get("next_steps", []),
        warnings=all_warnings,
        override_flags=a4_data.get("override_flags", []),
        gray_zone=a4_data.get("gray_zone", False),
        clinician_review_required=a4_data.get("clinician_review_required", False),
        agents=[a1_out, a2_out, a3_out, a4_out],
    )
