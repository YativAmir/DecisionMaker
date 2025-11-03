# -*- coding: utf-8 -*-
"""
Router stage (Hebrew, RTL-safe) for an eligibility pipeline.
- מסווג 0..N קטגוריות רלוונטיות על בסיס document_text בלבד
- מחזיר ציון ביטחון (0–1) לכל קטגוריה מותרת
- משתמש ב-OpenAI LLM עם כפיית פלט JSON
- מחזיר (categories, scored, document_text)
- כולל טיפול תקלות ולוגים מסודרים
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI, OpenAIError

# -----------------------------------------------------------------------------
# טעינת משתני סביבה (כולל OPENAI_API_KEY, ROUTER_MODEL)
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# לוגים
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("router")

# -----------------------------------------------------------------------------
# רשימת קטגוריות מותרות + נפילות (fallback)
# -----------------------------------------------------------------------------
ALLOWED_CATEGORIES: List[str] = [
    "ניידות",
    "נכות כללית",
    "תג נכה",
    "תאונת עבודה",
    "תאונת דרכים",
    "שירותים מיוחדים",
    "פטור מס הכנסה",
    "סיעוד חברת ביטוח",
    "סיעוד ביטוח לאומי",
    "נפגעי פעולות איבה",
    "משרד הביטחון",
    "מחלת מקצוע",
    "חברות ביטוח",
]

# סף ביטחון מינימלי להיכלל ב-categories (תאימות לאחור)
MIN_CONFIDENCE: float = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.40"))

# fallback אם יש שגיאה קשה
FALLBACK_CATEGORIES: List[str] = ["לא מסווג"]

# -----------------------------------------------------------------------------
# מודלי קלט/פלט (Pydantic v2)
# -----------------------------------------------------------------------------
class RouterInput(BaseModel):
    """קלט ה-Router: טקסט המסמך המלא בעברית."""
    document_text: str = Field(..., description="Hebrew full-text as extracted by the PDF Intake Parser.")

    @field_validator("document_text")
    @classmethod
    def non_empty_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("document_text cannot be empty.")
        return v


class ScoredCategory(BaseModel):
    """אובייקט דירוג לקטגוריה."""
    name: str = Field(..., description="Category name (must be one of ALLOWED_CATEGORIES).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0,1].")


class RouterOutput(BaseModel):
    """
    פלט ה-Router:
    - categories: קטגוריות שעברו את סף הביטחון (תאימות לאחור)
    - scored: דירוג ביטחון לכל קטגוריה מותרת
    - document_text: הטקסט המקורי
    """
    categories: List[str] = Field(..., description="Zero or more allowed Hebrew categories above confidence threshold.")
    scored: List[ScoredCategory] = Field(..., description="Per-category confidence for all allowed categories.")
    document_text: str = Field(..., description="Original input text (unchanged, Unicode/RTL-safe).")


# -----------------------------------------------------------------------------
# תצורה: מודל ולקוח OpenAI
# -----------------------------------------------------------------------------
OPENAI_MODEL: str = os.getenv("ROUTER_MODEL", "gpt-4o")

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    logger.warning("OPENAI_API_KEY לא מוגדר. יצירת לקוח תיכשל בהרצה בפועל אם לא יוגדר.")
_client = OpenAI(api_key=_api_key) if _api_key else None

# -----------------------------------------------------------------------------
# בניית הודעות (System/User) לפרומפט
# -----------------------------------------------------------------------------
def _build_messages(document_text: str) -> list[dict]:
    """
    System: דירוג כל הקטגוריות המותרות; הימנעות מפספוס (רגישות גבוהה); JSON תקין בלבד.
    User: מספק את רשימת הקטגוריות וטקסט המסמך; דורש JSON עם 'scored' + דוגמה מחייבת.
    """
    allowed_str = ", ".join(ALLOWED_CATEGORIES)

    system_msg = (
        "אתה מנהל תיקים בכיר במשרד עורכי דין לזכויות רפואיות. "
        "היצמד אך ורק לטקסט שסופק. "
        "הטה את סף ההחלטה לרגישות גבוהה: העדף False Positives על פני False Negatives; "
        "במקרה גבולי או אי-ודאות — הוסף/נקד ציון חיובי לקטגוריה. "
        "אל תמציא קטגוריות חדשות. "
        "החזר JSON תקין בלבד."

    )

    # דוגמה מחייבת כדי לקבל בדיוק את המפתח והמבנה נכון
    example_json = {
        "scored": [
            {"name": "תג נכה", "confidence": 0.62},
            {"name": "ניידות", "confidence": 0.35}
        ]
    }

    user_msg = (
        "קטגוריות מותרות (בעברית): "
        f"[{allowed_str}]\n\n"
        "טקסט המסמך (בעברית):\n"
        f"{document_text}\n\n"
        "דרישות פלט (JSON בלבד):\n"
        "1) שדה 'scored' עם רשימה של אובייקטים לכל קטגוריה מותרת בדיוק פעם אחת.\n"
        "2) לכל אובייקט: name (String מתוך הרשימה) ו-confidence (מספר בין 0 ל-1).\n"
        "3) אין לכלול שדות נוספים.\n"
        "דוגמה מבנית מחייבת (שנה את השמות/ציונים בהתאם למסמך):\n"
        f"{json.dumps(example_json, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

# -----------------------------------------------------------------------------
# עזר: נרמול וסינון
# -----------------------------------------------------------------------------
def _normalize_allowed_name(name: str) -> Optional[str]:
    """
    מחזיר את השם הקנוני אם name הוא בדיוק קטגוריה מותרת.
    (אפשר להרחיב כאן מיפויי aliases אם צריך בפועל.)
    """
    if not isinstance(name, str):
        return None
    cc = name.strip()
    return cc if cc in ALLOWED_CATEGORIES else None


def _filter_by_confidence(scored: List[ScoredCategory], min_conf: float) -> List[str]:
    """מחזיר רק קטגוריות מעל סף הביטחון, לפי סדר הופעה."""
    result: List[str] = []
    for sc in scored:
        if sc.confidence >= min_conf and sc.name not in result:
            result.append(sc.name)
    return result


# -----------------------------------------------------------------------------
# חילוץ מה-JSON של המודל
# -----------------------------------------------------------------------------
def _extract_scored_from_json(content: str) -> List[ScoredCategory]:
    """
    מצפה ל-{"scored":[{"name": "...", "confidence": 0.x}, ...]}.
    תאימות לאחור:
      - אם הוחזר {"categories":[...]} בלבד, נייצר scored עם confidence=1.0 לאותן קטגוריות
        ונאפס ל-0.0 את השאר כדי שתמיד יהיו כל הקטגוריות ב-scored.
    אם אין כלום — נחזיר רשימה ריקה.
    """
    data = json.loads(content)

    # מסלול ראשי: יש scored
    if isinstance(data, dict) and "scored" in data and isinstance(data["scored"], list):
        parsed: List[ScoredCategory] = []
        seen_names = set()
        for item in data["scored"]:
            if not isinstance(item, dict):
                continue
            name = _normalize_allowed_name(item.get("name"))
            conf = item.get("confidence")
            if name is None or not isinstance(conf, (int, float)):
                continue
            # גבולות
            if conf < 0.0: conf = 0.0
            if conf > 1.0: conf = 1.0
            if name not in seen_names:
                parsed.append(ScoredCategory(name=name, confidence=float(conf)))
                seen_names.add(name)

        # הבטחה לכל קטגוריה מותרת בדיוק פעם אחת: אם חסר, נוסיף עם 0.0
        missing = [c for c in ALLOWED_CATEGORIES if c not in {p.name for p in parsed}]
        for m in missing:
            parsed.append(ScoredCategory(name=m, confidence=0.0))
        return parsed

    # תאימות לאחור: הוחזרו רק categories
    if isinstance(data, dict) and "categories" in data and isinstance(data["categories"], list):
        chosen = {_normalize_allowed_name(c) for c in data["categories"] if isinstance(c, str)}
        chosen.discard(None)
        parsed: List[ScoredCategory] = []
        for cat in ALLOWED_CATEGORIES:
            conf = 1.0 if cat in chosen else 0.0
            parsed.append(ScoredCategory(name=cat, confidence=conf))
        return parsed

    # לא הצלחנו לפרסר
    return []


# -----------------------------------------------------------------------------
# קריאה ל-LLM והפקת ציונים + קטגוריות מעל סף
# -----------------------------------------------------------------------------
def _call_llm_for_scores(document_text: str) -> List[ScoredCategory]:
    """
    קורא ל-OpenAI עם response_format={"type": "json_object"} ומחזיר רשימת ScoredCategory
    לכל ALLOWED_CATEGORIES (גם אם confidence=0).
    במקרה של כשל/JSON לא תקין -> נחזיר רשימה ריקה (יטופל בהמשך).
    """
    if _client is None:
        logger.error("OpenAI client לא מאותחל (אין API key).")
        return []

    try:
        messages = _build_messages(document_text)
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            timeout=60,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        if not content:
            logger.warning("LLM החזיר תוכן ריק.")
            return []

        logger.debug("LLM raw content: %s", content)
        scored = _extract_scored_from_json(content)
        logger.debug("Parsed scored: %s", [s.model_dump() for s in scored])
        return scored

    except (OpenAIError, json.JSONDecodeError, KeyError) as e:
        logger.exception("כשל קריאה/פענוח LLM. error=%s", e)
        return []
    except Exception as e:
        logger.exception("שגיאה לא צפויה. error=%s", e)
        return []


# -----------------------------------------------------------------------------
# ממשק ציבורי
# -----------------------------------------------------------------------------
def route(payload: RouterInput) -> RouterOutput:
    """
    נקודת הכניסה ל-Router:
    - ולידציה לקלט ב-Pydantic
    - קריאה ל-LLM לקבלת דירוג לכל הקטגוריות
    - גזירה של categories לפי סף MIN_CONFIDENCE
    - במקרה של כשל חמור -> החזר FALLBACK_CATEGORIES ו-scored ריק
    """
    document_text = payload.document_text
    scored = _call_llm_for_scores(document_text=document_text)

    if not scored:
        # כשל חמור: נשמור תאימות לאחור
        return RouterOutput(
            categories=FALLBACK_CATEGORIES,
            scored=[],
            document_text=document_text,
        )

    categories = _filter_by_confidence(scored, MIN_CONFIDENCE)
    return RouterOutput(categories=categories, scored=scored, document_text=document_text)


# -----------------------------------------------------------------------------
# CLI להרצה ידנית
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY לא מוגדר. יש להגדיר משתנה סביבה לפני ההרצה.")
        sys.exit(1)

    example_text = """

שאלה – מה שמך? תשובה – בר יהודה לוי | שאלה – טלפון: תשובה – 0525000258 | שאלה – ת.ז: תשובה – 056570955 | שאלה – תאריך לידה: תשובה – 29/07/1960 | שאלה – האם השאלון עבור גבר או אישה? תשובה – אישה | שאלה – סטטוס תעסוקתי: תשובה – בפנסיה / לא עובד/ת | שאלה – (כותרת) שאלון פטור ממס הכנסה – הכנסה: תשובה – — | שאלה – מאילו מחלות/נכויות הנך סובל? תשובה – סכרת (סוג 2, מטופל באוזמפיק וכדורים, מצוין “מאוזן”), דום נשימה בשינה (לא מסתדר עם CPAP, נוטל כדורי שינה רבים), מחלת ריאות (המלצת מומחה ל־30% נכות; פגיעה בתפקוד), פיברומיאלגיה (כאב יומיומי; לוקח קלונקס), אורתופדיה – גב/צוואר/ברכיים (כתף שמאל קרע גיד/שריר; מועמד להחלפת ברכיים; זריקות תכופות; בעיה גם בברך השנייה; ב־2012 נקבעו 10% לצמיתות בברך בעקבות תאונת עבודה וקיבל מענק חד־פעמי ~5K), וכן “אחר” | שאלה – פירוט “אחר”: תשובה – לפני כחודש וחצי עבר ניתוח להסרת גידול מהמוח; במעקב נוירולוג ואונקולוג; דרגה 1 ושפיר לעת עתה, ממתין לתוצאות; מצוין ש“זה סוג גידול מסוכן”; חוסר יציבות; MRI 03/2025 עם צלקת; מחלת עור כרונית (ליכנפלנוס) החמירה – מתגרד כל היום, מטופל בקרמים; ירידה בשמיעה – נדרש מכשיר שמיעה; יובש בפה; סובל מדיכאון, לוקח תרופות להרגעה אך “לא מצליח להירגע”; היה במעקב פסיכיאטר פעם אחת, כיום לא במעקב | שאלה – סטטוס תעסוקתי (נוסף): תשובה – שכיר | שאלה – מאיזו הכנסה מנוכה מס? תשובה – “אחר” | שאלה – מה גובה המס החודשי? תשובה – כ־2,400 ש״ח; יש עוסק מורשה (השכרת מחסן/מוסך); משלם מקדמות; שוקל להעביר על שם אשתו (נכות כללית) ומתייעץ “מה עדיף”, מצוין 816 ש״ח/חודש | שאלה – כמה שנים אתה משלם מס בגובה זה? תשובה – 5 שנים ומעלה | שאלה – האם הוגשה תביעה בעבר? תשובה – לא.



""".strip()

    try:
        payload = RouterInput(document_text=example_text)
    except Exception as e:
        logger.error("קלט לא תקין: %s", e)
        sys.exit(1)

    output = route(payload)
    print(output.model_dump_json(ensure_ascii=False, indent=2))
