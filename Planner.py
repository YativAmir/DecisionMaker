# -*- coding: utf-8 -*-
"""
Planner stage for a Hebrew (RTL) eligibility pipeline.

- קולט: קטגוריה (מה-Router) + document_text (הטקסט המלא של המסמך בעברית)
- פולט: רשימת שאילתות קריטריונים (criteria_queries) ושאלת זכאות מנוסחת (question) בעברית
- אינו ניגש לשום API חיצוני: הקריטריונים מקודדים כתבניות במפה פנימית
- בנוי מודולרי להרחבה עתידית
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Tuple  # Tuple לא בשימוש כרגע, נשאר לעתיד

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# מודלי קלט/פלט (Pydantic v2)
# ---------------------------------------------------------

class PlannerInput(BaseModel):
    """קלט לשלב ה-Planner."""
    # קטגוריה כפי שנקבעה ב-Router (מחרוזת בעברית/קוד)
    category: str = Field(..., description="קטגוריית הזכאות כפי שזוהתה ע״י ה-Router (מחרוזת/קוד בעברית).")
    # הטקסט המלא של המסמך – יכול לשמש להתאמות קלות בבניית השאילתות
    document_text: str = Field(..., description="טקסט המסמך המלא בעברית, כפי שחולץ בשלבים הקודמים.")

    @field_validator("category")
    @classmethod
    def non_empty_category(cls, v: str) -> str:
        # ולידציה בסיסית: אסור ערך ריק / רווחים
        if not isinstance(v, str) or not v.strip():
            raise ValueError("category cannot be empty.")
        return v.strip()

    @field_validator("document_text")
    @classmethod
    def non_empty_text(cls, v: str) -> str:
        # ולידציה בסיסית: חובה טקסט כלשהו
        if not isinstance(v, str) or not v.strip():
            raise ValueError("document_text cannot be empty.")
        return v


class PlannerOutput(BaseModel):
    """פלט ה-Planner: שאילתות קריטריונים ושאלת הזכאות הסופית (בעברית)."""
    # רשימת שאילתות קצרות וברורות בעברית לשליפת סעיפי קריטריונים ע״י ה-Retriever
    criteria_queries: List[str] = Field(..., description="רשימת שאילתות חיפוש/שליפה בעברית למקטעי קריטריונים רלוונטיים.")
    # השאלה שתנחה את שלב ה-Generator – מנוסחת בעברית ומכילה את שם הקטגוריה
    question: str = Field(..., description="שאלת הזכאות המנחה לשלב המפיק (Generator), בעברית.")


# ---------------------------------------------------------
# נורמליזציית קטגוריות + רשימת קטגוריות מוכרות
# ---------------------------------------------------------

class CanonicalCategory(str, Enum):
    """ייצוג קאנוני (אחיד) לכל קטגוריה – מונע וריאציות כתיב."""
    NAKHUT_KLALIT = "נכות כללית"
    GAMALAT_SIUD = "גמלת סיעוד"
    NAYADUT = "ניידות"
    TAG_NECHET = "תג נכה"
    TEUNAT_AVODA = "תאונת עבודה"
    TEUNAT_DRACHIM = "תאונת דרכים"
    SHERUTIM_MEYUCHADIM = "שירותים מיוחדים"
    PTOR_MAS = "פטור ממס הכנסה"
    NIFGAE_PIGUA = "נפגעי פעולות איבה"
    MISRAD_HABITACHON = "משרד הביטחון"
    MACHALAT_MIKZOA = "מחלת מקצוע"


# מיפוי שמות חלופיים -> שם קאנוני
# מאפשר ל-Planner להבין מגוון ניסוחים שמגיעים מה-Router
CATEGORY_ALIASES: Dict[str, CanonicalCategory] = {
    # סיעוד
    "סיעוד": CanonicalCategory.GAMALAT_SIUD,
    "סיעוד ביטוח לאומי": CanonicalCategory.GAMALAT_SIUD,
    "גמלת סיעוד": CanonicalCategory.GAMALAT_SIUD,

    # נכות כללית
    "נכות כללית": CanonicalCategory.NAKHUT_KLALIT,
    "קצבת נכות": CanonicalCategory.NAKHUT_KLALIT,

    # ניידות / תג נכה
    "ניידות": CanonicalCategory.NAYADUT,
    "תג נכה": CanonicalCategory.TAG_NECHET,

    # תאונות/פגיעה
    "תאונת עבודה": CanonicalCategory.TEUNAT_AVODA,
    "תאונת דרכים": CanonicalCategory.TEUNAT_DRACHIM,
    "נפגעי פעולות איבה": CanonicalCategory.NIFGAE_PIGUA,
    "משרד הביטחון": CanonicalCategory.MISRAD_HABITACHON,

    # שירותים מיוחדים / פטור מס / מחלת מקצוע
    "שירותים מיוחדים": CanonicalCategory.SHERUTIM_MEYUCHADIM,
    "שירותים מיוחדים (סיעוד לפני סיעוד)": CanonicalCategory.SHERUTIM_MEYUCHADIM,
    "פטור מס הכנסה": CanonicalCategory.PTOR_MAS,
    "מחלת מקצוע": CanonicalCategory.MACHALAT_MIKZOA,
}


def normalize_category(raw: str) -> CanonicalCategory:
    """
    נורמליזציה של הערך שמגיע מה-Router לשם קאנוני אחיד.
    - קודם בדיקה מדויקת במפה (alias מדויק)
    - אם לא נמצא: ניסיון match רך (contains) כדי לתפוס וריאציות
    - אם לא מזוהה כלל: זורקים ValueError כדי שהאורקסטרטור ידע לטפל
    """
    key = raw.strip()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    # התאמות רכות – שימוש זהיר כדי לא לתפוס בטעות
    for alias, can in CATEGORY_ALIASES.items():
        if alias in key:
            return can
    raise ValueError(f"קטגוריה לא מוכרת: '{raw}'. יש לבחור אחת מהקטגוריות המוכרות.")


# ---------------------------------------------------------
# תבניות קריטריונים לשאילתות לפי קטגוריה (בעברית)
# ---------------------------------------------------------

# הערה: השאילתות הן ביטויי חיפוש קצרים ומדויקים בעברית.
# המטרה: לאפשר ל-Retriever לאתר סעיפי קריטריונים רלוונטיים במסמכי המקור.
CRITERIA_TEMPLATES: Dict[CanonicalCategory, List[str]] = {
    CanonicalCategory.NAKHUT_KLALIT: [
        "נכות כללית – תנאי זכאות בסיסיים",
        "נכות כללית – אחוז נכות רפואית מינימלי",
        "נכות כללית – כושר השתכרות ועבודה",
        "נכות כללית – תקופת אכשרה/דמי ביטוח",
        "נכות כללית – גיל וזכאות (לפני/אחרי גיל פרישה)",
        "נכות כללית – חריגים והוראות מעבר",
        "נכות כללית – מסמכים רפואיים נדרשים",
    ],
    CanonicalCategory.GAMALAT_SIUD: [
        "גמלת סיעוד – מבחן תלות (ADL)",
        "גמלת סיעוד – מבחן הכנסה",
        "גמלת סיעוד – גיל זכאות",
        "גמלת סיעוד – דרגות זכאות והגדרות",
        "גמלת סיעוד – מסמכים ואישורים נדרשים",
    ],
    CanonicalCategory.NAYADUT: [
        "ניידות – קריטריוני זכאות בסיסיים",
        "ניידות – אחוז מוגבלות בניידות/בדיקת ועדה",
        "ניידות – רכב ורישום בעלות",
        "ניידות – הטבות ותנאים נלווים",
    ],
    CanonicalCategory.TAG_NECHET: [
        "תג נכה – תנאי זכאות רפואיים",
        "תג נכה – הגבלת ניידות ומשמעויות רפואיות",
        "תג נכה – מסוכנות בריאותית בהליכה/ניידות",
        "תג נכה – מסמכים רפואיים ואישורים",
    ],
    CanonicalCategory.TEUNAT_AVODA: [
        "תאונת עבודה – אירוע תוך כדי ועקב העבודה",
        "תאונת עבודה – קשר סיבתי רפואי",
        "תאונת עבודה – דיווח ותביעה במועד",
        "תאונת עבודה – אחוז נכות ונזק תפקודי",
    ],
    CanonicalCategory.TEUNAT_DRACHIM: [
        "תאונת דרכים – הגדרת תאונת דרכים",
        "תאונת דרכים – קשר סיבתי בין התאונה לנזק",
        "תאונת דרכים – פוליסת ביטוח חובה ותנאים",
        "תאונת דרכים – מסמכים רפואיים ודיווחים",
    ],
    CanonicalCategory.SHERUTIM_MEYUCHADIM: [
        "שירותים מיוחדים – קריטריוני תלות וסיעודיות",
        "שירותים מיוחדים – בדיקת תפקוד יומיומי (ADL/IADL)",
        "שירותים מיוחדים – גיל/סטטוס תעסוקתי",
        "שירותים מיוחדים – מסמכים נדרשים",
    ],
    CanonicalCategory.PTOR_MAS: [
        "פטור ממס הכנסה – אחוזי נכות/עיוורון",
        "פטור ממס הכנסה – ועדות רפואיות",
        "פטור ממס הכנסה – תקופת זכאות והחלה",
        "פטור ממס הכנסה – מסמכים ואישורים",
    ],
    CanonicalCategory.NIFGAE_PIGUA: [
        "נפגעי פעולות איבה – הגדרת פגיעה מזכה",
        "נפגעי פעולות איבה – הכרה וגוף מטפל",
        "נפגעי פעולות איבה – קשר סיבתי ונזק",
        "נפגעי פעולות איבה – מסמכים נדרשים",
    ],
    CanonicalCategory.MISRAD_HABITACHON: [
        "משרד הביטחון – הכרה בנכות/פגיעה בשירות",
        "משרד הביטחון – קשר לשירות ותיעוד",
        "משרד הביטחון – ועדות רפואיות ודרגות",
        "משרד הביטחון – זכויות נלוות",
    ],
    CanonicalCategory.MACHALAT_MIKZOA: [
        "מחלת מקצוע – הגדרה ורשימת מחלות",
        "מחלת מקצוע – קשר סיבתי לתנאי עבודה",
        "מחלת מקצוע – תקופת אכשרה/דיווח",
        "מחלת מקצוע – מסמכים ובדיקות",
    ],
}


# ---------------------------------------------------------
# התאמה קלה על בסיס טקסט המסמך (אופציונלי, לא מסבך)
# ---------------------------------------------------------

def extract_age(document_text: str) -> int | None:
    """
    ניסיון גס לחלץ גיל/שנת לידה:
    - תבניות נפוצות: 'גיל 68', 'בן 72', 'בת 70'
    - חיפוש שנה מתוך תאריך לידה (למשל 1985) -> גיל בקירוב (בהנחה של 2025)
    """
    # חיפוש ביטוי גיל מפורש
    m = re.search(r"(?:גיל|בן|בת)\s*[:\-]?\s*(\d{2})", document_text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # חיפוש שנת לידה מתוך טקסט (פשטני) והמרה לגיל בקירוב
    y = re.search(r"(?:תאריך(?:\s*לידה)?|לידה)[^\d]{0,10}(\d{4})", document_text)
    if y:
        birth_year = int(y.group(1))
        approx_age = 2025 - birth_year  # עדכן את השנה לפי סביבת ההרצה
        if 0 < approx_age < 120:
            return approx_age

    return None


def tailor_queries(category: CanonicalCategory, base_queries: List[str], document_text: str) -> List[str]:
    """
    התאמה קלילה של רשימת השאילתות לפי רמזים בטקסט (ללא תלות ב-API):
    - הוספת דגשים בהתאם לגיל/מילות מפתח נפוצות
    - לא משנה את הבסיס – רק מוסיפה או מזיזה קדימה שאילתות חשובות
    """
    queries = list(base_queries)  # שמירה על המקור + יצירת עותק לשינויים
    age = extract_age(document_text)

    if category == CanonicalCategory.NAKHUT_KLALIT:
        # אם הגיל סביב/מעל פרישה – הדגש חריגים לזכאות אחרי גיל פרישה
        if age is not None and age >= 67:
            queries.insert(1, "נכות כללית – זכאות לאחר גיל פרישה/חריגים")

    if category == CanonicalCategory.GAMALAT_SIUD:
        # בגמלת סיעוד – אם מופיע גיל נמוך מ-67, הדגש בדיקת תנאי גיל מינימלי
        if age is not None and age < 67:
            queries.insert(0, "גמלת סיעוד – בדיקת עמידה בתנאי גיל מינימלי")

    if category in (CanonicalCategory.TEUNAT_AVODA, CanonicalCategory.MACHALAT_MIKZOA):
        # אם יש מילים שמרמזות על הקשר לעבודה – הוסף דגש על תיעוד מעסיק/מקום עבודה
        if re.search(r"\b(משמרת|מעסיק|מקום\s*עבודה|תפקיד|שכר)\b", document_text):
            queries.insert(0, f"{category.value} – תיעוד מעסיק/מקום עבודה")

    if category == CanonicalCategory.TAG_NECHET or category == CanonicalCategory.NAYADUT:
        # אם מוזכר קושי בניידות – הדגש תיעוד מגבלה חמורה
        if re.search(r"(קושי\s*בהליכה|קביים|כיסא\s*גלגלים|ניידות\s*מוגבלת)", document_text):
            queries.insert(0, f"{category.value} – תיעוד מגבלת ניידות חמורה")

    return queries


# ---------------------------------------------------------
# בניית השאלה הסופית (בעברית) לכל קטגוריה
# ---------------------------------------------------------

def build_final_question(category: CanonicalCategory) -> str:
    """ניסוח אחיד וברור של שאלת הזכאות לפי הקטגוריה."""
    return f"האם המטופל זכאי ל{category.value} בהתאם לקריטריונים הרלוונטיים במסמכי המקור?"


# ---------------------------------------------------------
# ליבת הלוגיקה: build_plan
# ---------------------------------------------------------

def build_plan(planner_input: PlannerInput) -> PlannerOutput:
    """
    יוצר תכנית בדיקה (שאילתות + שאלה) על בסיס קטגוריה וטקסט חופשי:
    1) נורמליזציה של הקטגוריה לשם קאנוני
    2) שליפת תבניות השאילתות מה-CRITERIA_TEMPLATES
    3) התאמה קלה לפי הטקסט (אופציונלי)
    4) ניסוח שאלת הזכאות הסופית
    """
    # 1) נורמליזציה (וזריקת שגיאה אם לא מוכר)
    category = normalize_category(planner_input.category)

    # 2) שליפת תבניות בסיסיות לקטגוריה
    base_queries = CRITERIA_TEMPLATES.get(category)
    if not base_queries:
        # אמור לא לקרות (כל קטגוריה קאנונית מוגדרת), נשאיר הגנה
        raise ValueError(f"לא נמצאו תבניות קריטריונים עבור הקטגוריה: {category.value}")

    # 3) התאמות קלות לפי הטקסט
    tailored = tailor_queries(category, base_queries, planner_input.document_text)

    # 4) ניסוח השאלה
    question = build_final_question(category)

    # יצירת פלט מובנה ל-Retriever/Generator
    return PlannerOutput(criteria_queries=tailored, question=question)


# ---------------------------------------------------------
# שימוש לדוגמה (ניתן להסרה/השארה לפי הצורך)
# ---------------------------------------------------------
if __name__ == "__main__":
    # דוגמה: קלט מה-Router ל"נכות כללית" עם טקסט שמזכיר גיל 68
    example_input = PlannerInput(
        category="קצבת נכות",
        document_text="""
        פרטי מבוטח: בן 68, מתקשה בעבודה מלאה בעקבות מחלה כרונית.
        תיעוד רפואי מצורף. מבקש בירור זכאות לנכות כללית.
        """
    )
    plan = build_plan(example_input)
    print(plan.model_dump_json(ensure_ascii=False, indent=2))

    # דוגמה נוספת: גמלת סיעוד עם גיל 62 -> הדגשת תנאי גיל
    example_input2 = PlannerInput(
        category="סיעוד ביטוח לאומי",
        document_text="המבוטחת בת 62, תלויה חלקית בעזרת הזולת בפעולות יומיומיות."
    )
    plan2 = build_plan(example_input2)
    print(plan2.model_dump_json(ensure_ascii=False, indent=2))
