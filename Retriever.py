# -*- coding: utf-8 -*-
"""
Retriever stage for a Hebrew (RTL) document-grounded eligibility pipeline.

- קולט:
  * criteria_queries: רשימת שאילתות/קריטריונים בעברית (מ-Planner)
  * criteria_documents: מסמכי קריטריונים (חוקים/נהלים) בעברית לחיפוש

- פולט:
  * retrieved_sections: רשימת קטעים רלוונטיים (מקור, טקסט, סעיף-ייחוס אם נמצא)

עקרונות:
- אין שימוש ב-API חיצוניים או מנועי חיפוש; חיפוש טקסטואלי פשוט מקומי.
- טיפול בסיסי בעברית: הסרת ניקוד, פיצול לשורות/פסקאות/משפטים, התאמות מילות־מפתח.
- ביצועים: סריקה לפי פסקאות/שורות והערכת ציון התאמה (מספר פגיעות מילות־מפתח).
- אם לא נמצאה פגיעה: מחזירים רשומה "לא נמצא מידע" עבור אותה שאילתה (ניתן לשנות למדיניות אחרת).
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple, Iterable

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# לוגים
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] retriever - %(message)s"
)
logger = logging.getLogger("retriever")


# ---------------------------------------------------------
# מודלי קלט/פלט (Pydantic v2)
# ---------------------------------------------------------

class CriteriaDocument(BaseModel):
    """מסמך קריטריונים יחיד (חוק/נהלים)."""
    id: str = Field(..., description="מזהה/שם המסמך (למשל 'חוק הביטוח הלאומי').")
    content: str = Field(..., description="מלל מלא של המסמך בעברית.")

    @field_validator("id", "content")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("שדה מחרוזת לא יכול להיות ריק.")
        return v


class RetrievedSection(BaseModel):
    """קטע רלוונטי שנשלף מתוך מסמך."""
    source_id: str = Field(..., description="מזהה/שם המסמך שממנו נשלף הקטע.")
    text: str = Field(..., description="קטע הטקסט הרלוונטי (משפט/פסקה).")
    section_ref: Optional[str] = Field(
        default=None,
        description="תווית סעיף אם זוהתה (למשל 'סעיף 3(א)')."
    )


class RetrieverInput(BaseModel):
    """קלט לשלב ה-Retriever."""
    criteria_queries: List[str] = Field(..., description="רשימת שאילתות/קריטריונים בעברית לחיפוש.")
    criteria_documents: List[CriteriaDocument] = Field(..., description="רשימת מסמכי קריטריונים לחיפוש.")

    @field_validator("criteria_queries")
    @classmethod
    def non_empty_queries(cls, v: List[str]) -> List[str]:
        if not v or not all(isinstance(q, str) and q.strip() for q in v):
            raise ValueError("חובה לספק לפחות שאילתה אחת לא ריקה.")
        return v

    @field_validator("criteria_documents")
    @classmethod
    def non_empty_docs(cls, v: List[CriteriaDocument]) -> List[CriteriaDocument]:
        if not v:
            raise ValueError("חובה לספק לפחות מסמך קריטריונים אחד.")
        return v


class RetrieverOutput(BaseModel):
    """פלט שלב ה-Retriever."""
    retrieved_sections: List[RetrievedSection] = Field(
        default_factory=list,
        description="רשימת קטעים רלוונטיים שנשלפו עבור השאילתות."
    )


# ---------------------------------------------------------
# כלים לעיבוד טקסט עברי בסיסי
# ---------------------------------------------------------

# טווח יוניקוד של סימני ניקוד בעברית (Niqqud)
_NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")

# זיהוי 'סעיף 3', 'סעיף 3(א)', 'סעיף 12א', 'סעיף 12(ג)(2)' וכו'
_SECTION_REF_RE = re.compile(r"סעיף\s+[\d]+[\w\(\)״״״\"׳׳\-]*")

# מפרקי משפטים/פסקאות בסיסיים (פשוטים כדי להישאר בלי תלויות)
_SENTENCE_SEP_RE = re.compile(r"(?<=[\.!?])\s+|\n+")
_PARAGRAPH_SEP_RE = re.compile(r"\n{2,}|(?:\r?\n)+")

# מילות עצירה עבריות בסיסיות (ניתן להרחיב לפי צורך)
_HE_STOPWORDS = {
    "של", "על", "אל", "עם", "אם", "או", "גם", "וכן", "וכן,", "וכן.", "וכן;", "וכן:", "וכן־",
    "כל", "ללא", "מבלי", "לא", "כן", "יכול", "יכולה", "יכולים", "יהיה", "תהיה",
    "לפי", "על־פי", "לפיה", "בהתאם", "לכך", "כמו", "וכו", "וכו'", "וכו’.", "וכו’", "וכו׳",
    "הוא", "היא", "הם", "הן", "שלו", "שלה", "שלהם", "להיות", "לה", "לו", "אליו", "אליה",
    "זה", "זאת", "אלה", "אשר", "כי", "כך", "כך,", "כך.", "כך;", "כך:",
    "מ-", "ל-", "ב-", "כ-", "ו-", "ש-",
}

def normalize_hebrew(text: str) -> str:
    """ניקוי קל לעברית: הסרת ניקוד, ריווח בסיסי."""
    # הסרת ניקוד
    text = _NIQQUD_RE.sub("", text)
    # איחוד רווחים מיותרים
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_to_paragraphs(text: str) -> List[str]:
    """פיצול למסגרות קריאות (פסקאות/שורות גדולות)."""
    # תחילה פיצול לפי פסקאות כפולות/שבירה מרובה
    parts = [p.strip() for p in _PARAGRAPH_SEP_RE.split(text) if p.strip()]
    # אם מעט מדי פסקאות, אפשר גם לפצל לפי שורות בודדות ארוכות
    if len(parts) <= 1:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts


def split_to_sentences(text: str) -> List[str]:
    """פיצול למשפטים גסים (ללא ספרייה חיצונית)."""
    # שימוש בפסיקים/נקודות/סימני שאלה/שורות
    # (שימו לב: זה Heuristic בלבד)
    candidates = [s.strip() for s in _SENTENCE_SEP_RE.split(text) if s and s.strip()]
    return candidates if candidates else [text.strip()]


def extract_section_ref(around_text: str) -> Optional[str]:
    """חיפוש 'סעיף ...' בקטע/סביבה והחזרת ההתאמה הראשונה."""
    m = _SECTION_REF_RE.search(around_text)
    return m.group(0) if m else None


def keywords_from_query(query: str, min_len: int = 2) -> List[str]:
    """
    הפקה פשוטה של מילות־מפתח מהשאילתה:
    - נירמול עברית (ללא ניקוד)
    - פיצול לפי רווחים/סימני פיסוק
    - הסרת מילות עצירה
    - החלת סף אורך בסיסי
    """
    qn = normalize_hebrew(query)
    tokens = re.split(r"[\s,.;:()\[\]{}\"'־\-–—/\\]+", qn)
    kws = [t for t in tokens if t and len(t) >= min_len and t not in _HE_STOPWORDS]
    # הסרת כפילויות תוך שמירת סדר
    seen = set()
    result = []
    for t in kws:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def match_score(text_norm: str, keywords: List[str]) -> int:
    """
    ניקוד פשוט על בסיס ספירת פגיעות מילות־מפתח (exact substring).
    ניתן להחליף ל-N-gram/regex לפי צורך, אבל נשמור פשוט.
    """
    score = 0
    for kw in keywords:
        if kw and kw in text_norm:
            score += 1
    return score


def best_snippet_for_query_in_doc(
    query: str,
    doc: CriteriaDocument,
    max_snippet_chars: int = 700
) -> Optional[Tuple[str, Optional[str]]]:
    """
    מחפש את הקטע הטוב ביותר למסמך נתון עבור שאילתה:
    - מפצל לפסקאות; אם אין התאמות, מנסה גם משפטים.
    - בוחר את המסגרת עם הניקוד הגבוה ביותר (מספר פגיעות מילות־מפתח).
    - קוצץ לאורך סביר ושולף section_ref אם אפשר.
    מחזיר: (snippet_text, section_ref) או None אם אין התאמות.
    """
    kws = keywords_from_query(query)
    if not kws:
        return None

    # נירמול המסמך (לצורך התאמה); את ה-snippet נחזיר מהטקסט המקורי כדי לשמר ניקוד/סימנים אם קיימים
    content = doc.content
    paragraphs = split_to_paragraphs(content)

    best_para = None
    best_score = 0

    # חיפוש בפסקאות
    for para in paragraphs:
        score = match_score(normalize_hebrew(para), kws)
        if score > best_score:
            best_score = score
            best_para = para

    # אם אין פגיעה בפסקאות, ננסה במשפטים בכל המסמך
    best_sent = None
    if best_score == 0:
        sentences = split_to_sentences(content)
        for sent in sentences:
            score = match_score(normalize_hebrew(sent), kws)
            if score > best_score:
                best_score = score
                best_sent = sent

    if best_score == 0:
        return None

    raw_snippet = best_para if best_para is not None else best_sent
    if not raw_snippet:
        return None

    # חילוץ section_ref מהסביבה (ננסה גם מסביב לפסקה/משפט)
    # כדי להעלות סיכוי, נסתכל גם על כמה שורות לפני/אחרי אם מדובר בפסקה
    section_ref = extract_section_ref(raw_snippet)
    if section_ref is None and best_para is not None:
        # נסה למצוא סעיף בתוך חלון טקסט רחב יותר סביב הפסקה
        try:
            idx = content.index(best_para)
            window = content[max(0, idx - 400): idx + len(best_para) + 400]
            section_ref = extract_section_ref(window)
        except ValueError:
            pass

    # קיצור אורך סביר
    snippet = raw_snippet.strip()
    if len(snippet) > max_snippet_chars:
        snippet = snippet[: max_snippet_chars].rstrip() + "…"

    return snippet, section_ref


# ---------------------------------------------------------
# פונקציית הרצה עיקרית (API פנימי)
# ---------------------------------------------------------

def retrieve(input_data: RetrieverInput, max_per_query: int = 1) -> RetrieverOutput:
    """
    מריץ איתור קטעים לכל שאילתה על פני כלל המסמכים.
    - max_per_query: כמה קטעים לכל שאילתה להחזיר (ברירת מחדל 1 - ההתאמה הטובה ביותר על פני כל המסמכים).
      אם >1, נשמור את ההתאמות הטובות ביותר פר-מסמך ונמיין לפי הציון (פשוט: פגיעות מילות־מפתח).
      כאן, לצמצום מורכבות, נחשב "ציון" באופן פנימי בתוך best_snippet_for_query_in_doc ונאסוף רק התאמות שקיימות.
    - אם לא נמצאה התאמה בשום מסמך, נוסיף רשומה "לא נמצא מידע" (עם source_id = המסמך הראשון או "N/A").
    """
    queries = input_data.criteria_queries
    docs = input_data.criteria_documents

    results: List[RetrievedSection] = []

    for q in queries:
        logger.info(f"מחפש התאמות לשאילתה: {q!r}")

        # אסוף מועמדים מכל המסמכים
        candidates: List[Tuple[str, str, Optional[str]]] = []  # (doc_id, snippet, section_ref)

        for d in docs:
            found = best_snippet_for_query_in_doc(q, d)
            if found is not None:
                snippet, section_ref = found
                candidates.append((d.id, snippet, section_ref))

        if not candidates:
            # לא נמצאה התאמה בשום מסמך: מדיניות – החזר “לא נמצא מידע”
            results.append(RetrievedSection(
                source_id="N/A",
                text="לא נמצא מידע תואם לשאילתה במסמכי הקריטריונים שסופקו.",
                section_ref=None
            ))
            continue

        # אם רוצים יותר מתוצאה אחת לשאילתה, אפשר להרחיב כאן ניקוד/מיון.
        # כרגע נחזיר עד max_per_query הראשונים (ללא ניקוד כמותי נוסף).
        for doc_id, snippet, section_ref in candidates[:max_per_query]:
            results.append(RetrievedSection(
                source_id=doc_id,
                text=snippet,
                section_ref=section_ref
            ))

    return RetrieverOutput(retrieved_sections=results)


# ---------------------------------------------------------
# דוגמת שימוש מינימלית (להדגמה/בדיקות ידניות)
# ---------------------------------------------------------
if __name__ == "__main__":
    example_docs = [
        CriteriaDocument(
            id="חוק הביטוח הלאומי",
            content=(
                "חוק הביטוח הלאומי [נוסח משולב] קובע זכאויות שונות. "
                "סעיף 3(א) מגדיר תנאי זכאות לקצבת נכות, לרבות הגדרת נכות רפואית ואובדן כושר עבודה.\n\n"
                "תנאי סף: גיל מינימלי 18. מבחני הכנסה עשויים לחול בהתאם לתקנות."
            )
        ),
        CriteriaDocument(
            id="תקנות סיעוד",
            content=(
                "במסגרת גמלת סיעוד, הזכאות נקבעת לפי מבחן תלות תפקודית. "
                "סעיף 12(ב) מפרט את רמות הזכאות. "
                "נדרש תושב ישראל, מעל גיל פרישה, ומבחן הערכת תלות."
            )
        ),
    ]

    example_input = RetrieverInput(
        criteria_queries=[
            "מהו הגיל המינימלי לקצבת נכות?",
            "תנאי זכאות לגמלת סיעוד לפי מבחן תלות תפקודית"
        ],
        criteria_documents=example_docs
    )

    out = retrieve(example_input, max_per_query=1)
    for i, sec in enumerate(out.retrieved_sections, 1):
        print(f"[{i}] {sec.source_id} | {sec.section_ref or '-'}\n{sec.text}\n")
