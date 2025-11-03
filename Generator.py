from pydantic import BaseModel
from typing import List

class RetrievedSection(BaseModel):
    text: str        # The text of the criteria section (snippet from the law/criteria document)
    source: str      # Reference or name of the source document/section (for citation)

class GeneratorInput(BaseModel):
    question: str                # The eligibility question (in Hebrew)
    patient_text: str            # The patient's information/details relevant to the question
    retrieved_sections: List[RetrievedSection]  # List of relevant criteria sections with sources

class GeneratorOutput(BaseModel):
    answer: str                  # The generated answer explaining eligibility with citations


import os
import openai
from dotenv import load_dotenv

# Load environment variables (to get OPENAI_API_KEY, etc.)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# בראש הקובץ (Generator.py), החלף את ה-import הישן:
# import openai
from openai import OpenAI

# אתחול לקוח פעם אחת למודול
client = OpenAI()  # ייקח את המפתח מה-ENV: OPENAI_API_KEY

def generate_answer(input_data: GeneratorInput) -> GeneratorOutput:
    """
    Generate an eligibility answer in Hebrew based on the question, patient info, and retrieved criteria sections.
    The answer will state if the patient is eligible and why, with citations from the provided criteria.
    """
    # Validate that we have criteria to reference
    if not input_data.retrieved_sections:
        answer_text = "מצטערים, לא ניתן לקבוע זכאות כיוון שלא נמצאו קריטריונים רלוונטיים במידע שסופק."
        return GeneratorOutput(answer=answer_text)

    # Construct the system message with instructions for the LLM
    system_message = (
        "אתה עוזר מומחה בתחום הזכאות לגמלאות. "
        "הסתמך אך ורק על המידע המוצג לך וענה בעברית רשמית וברורה. "
        "על כל קביעה לספק סימוכין מהמקורות הנתונים. "
        "אם מידע הדרוש לקביעה חסר במידע שניתן, ציין זאת בתשובתך."
    )

    # Construct the user message with the question, patient info, and the relevant criteria sections
    user_message = f"שאלה: {input_data.question}\n"
    user_message += f"פרטי המטופל: {input_data.patient_text}\n\n"
    user_message += "קריטריונים רלוונטיים:\n"
    for idx, section in enumerate(input_data.retrieved_sections, start=1):
        user_message += f"{idx}. {section.source}: {section.text}\n"
    user_message += (
        "\nבהתבסס על המידע הנ\"ל, קבע האם המטופל זכאי או אינו זכאי והסבר את הסיבות לכך. "
        "יש להתייחס לכל קריטריון רלוונטי ולצטט את המקור המתאים לכל טענה בתשובתך."
    )

    # Prepare the messages payload for the Chat API (v1)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        # v1: השימוש הנכון הוא דרך client.chat.completions.create
        response = client.chat.completions.create(
            model="gpt-4o",       # אפשר להחליף למודל אחר לפי הצורך
            messages=messages,
            temperature=0.2,      # פלט יותר דטרמיניסטי
            max_tokens=1024       # הגבלת אורך תשובה
        )
    except Exception as e:
        # השארת הטיפול בשגיאות; אין עטיפה נוספת למניעת Traceback כפול, אבל אפשר להשאיר אם תרצה
        raise RuntimeError(f"OpenAI API call failed: {e}")

    # Extract the answer text from the response
    answer_text = response.choices[0].message.content.strip()
    return GeneratorOutput(answer=answer_text)
