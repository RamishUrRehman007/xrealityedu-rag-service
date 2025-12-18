from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

# --- Primitives ---
class TextSpanRef(BaseModel):
    page: Optional[int] = None
    section_heading: Optional[str] = None
    excerpt: Optional[str] = None

class SourceProvenance(BaseModel):
    kind: Literal["textbook", "past_paper", "reference", "teacher_generated"]
    doc_id: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    paper_id: Optional[str] = None
    year: Optional[int] = None
    session: Optional[str] = None
    question_no: Optional[str] = None

class ReferenceSource(BaseModel):
    ref_id: str
    title: str
    kind: Literal["book", "notes", "website", "video", "paper"]
    author: Optional[str] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    checksum: Optional[str] = None
    coverage_topic_ids: List[str] = []

class Meta(BaseModel):
    version: str
    created_at: str
    language: str = "en"
    grade_band: Optional[Literal["K-2", "3-5", "6-8", "9-10", "11-12", "mixed"]] = None
    curriculum_region: Optional[str] = None
    source: dict = Field(default_factory=dict, description="Original content source metadata")

# --- Core Content Primitives ---
class Prerequisite(BaseModel):
    id: str
    description: str
    importance: Literal["must", "should", "nice"] = "should"

class LearningObjective(BaseModel):
    id: str
    statement: str
    blooms_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"] = "understand"

class KeyTerm(BaseModel):
    term: str
    definition: str
    aliases: List[str] = []
    source_ref: Optional[TextSpanRef] = None

# --- Chapter Profile ---
class ContentBreakdown(BaseModel):
    theory_percent: float
    numericals_percent: float
    definitions_percent: float = 0
    diagrams_percent: float = 0
    experiments_percent: float = 0
    formulas_present: bool

class TeachingPolicy(BaseModel):
    prioritize_numericals_if_formulas: bool = True
    exam_focus_mode: Literal["off", "balanced", "exam_first"] = "balanced"
    past_paper_weight_multiplier: float = 2.0
    max_explain_minutes_per_chunk: int = 7
    default_hint_ladder_levels: int = 4

class ChapterProfile(BaseModel):
    chapter_id: str
    chapter_title: str
    summary: Optional[str] = None
    estimated_total_minutes: Optional[int] = None
    prerequisites: List[Prerequisite] = []
    learning_objectives: List[LearningObjective]
    key_terms: List[KeyTerm] = []
    content_breakdown: ContentBreakdown
    teaching_policy: Optional[TeachingPolicy] = None

# --- Topic Map ---
class Topic(BaseModel):
    topic_id: str
    title: str
    order: int
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    prereq_topic_ids: List[str] = []
    key_ideas: List[str] = []
    misconceptions: List[str] = []
    source_refs: List[TextSpanRef] = []

class Evidence(BaseModel):
    past_paper_count: int
    past_paper_marks_total: float
    years_seen: List[int] = []

class TopicExamWeight(BaseModel):
    topic_id: str
    exam_weight: float
    evidence: Evidence

# --- Formulas & Examples ---
class Variable(BaseModel):
    symbol: str
    meaning: str
    units: Optional[str] = None
    constraints: Optional[str] = None

class SolutionStep(BaseModel):
    step: str
    explanation: str

class WorkedExample(BaseModel):
    example_id: str
    title: str
    problem: str
    given: List[str] = []
    steps: List[SolutionStep]
    final_answer: str
    checks: List[str] = []
    common_mistakes: List[str] = []
    provenance: Optional[SourceProvenance] = None

class Formula(BaseModel):
    formula_id: str
    name: str
    expression: str
    topic_ids: List[str] = []
    variables: List[Variable]
    units_notes: Optional[str] = None
    when_to_use: Optional[str] = None
    common_mistakes: List[str] = []
    worked_examples: List[WorkedExample] = []
    provenance: Optional[SourceProvenance] = None
    source_refs: List[TextSpanRef] = []

# --- Questions & Practice ---
class HintStep(BaseModel):
    level: int
    text: str
    reveal: Literal["concept", "formula", "substitution", "calculation", "next_step", "final_answer"] = "next_step"

class Solution(BaseModel):
    steps: List[SolutionStep]
    final_answer: str
    grading_notes: Optional[str] = None

class AnswerKey(BaseModel):
    correct_choice_index: Optional[int] = None
    correct_text: Optional[str] = None
    numeric_answer: Optional[float] = None
    unit: Optional[str] = None
    tolerance: float = 0

class Question(BaseModel):
    id: str
    type: Literal["mcq", "short", "numeric", "multi_step", "true_false", "match", "fill_blank"]
    prompt: str
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    topic_ids: List[str] = []
    choices: List[str] = [] # Required if type=mcq
    answer_key: Optional[AnswerKey] = None
    provenance: Optional[SourceProvenance] = None
    constraints: Optional[dict] = None # simplified from schema
    source_refs: List[TextSpanRef] = []

# --- Sessions ---
class TeachPoint(BaseModel):
    type: Literal["explain", "definition", "analogy", "diagram_note", "demo", "formula_intro"]
    text: str
    source_refs: List[TextSpanRef] = []
    provenance: Optional[SourceProvenance] = None

class Check(BaseModel):
    check_id: str
    question: Question
    expected: Optional[dict] = None

class PracticeItem(BaseModel):
    question_id: str
    question: Question
    hints: List[HintStep] = []
    solution: Optional[Solution] = None
    tags: List[str] = []
    provenance: Optional[SourceProvenance] = None

class ExamFocusPayload(BaseModel):
    is_exam_priority_session: bool = False
    past_paper_question_ids: List[str] = []

class Session(BaseModel):
    session_id: str
    title: str
    order: int
    duration_minutes: int
    objective: str
    topic_ids: List[str]
    exam_focus: Optional[ExamFocusPayload] = None
    warmup: List[Question] = []
    teach_points: List[TeachPoint]
    guided_examples: List[WorkedExample] = []
    checks: List[Check]
    practice: List[PracticeItem]
    homework: List[Question] = []
    recap_bullets: List[str] = []
    common_mistakes: List[str] = []

# --- Past Papers ---
class PastPaper(BaseModel):
    paper_id: str
    year: int
    board: Optional[str] = None
    exam_name: Optional[str] = None
    session: Optional[str] = None
    variant: Optional[str] = None
    max_marks: Optional[int] = None
    source_doc_id: Optional[str] = None
    checksum: Optional[str] = None

class PastPaperQuestion(BaseModel):
    id: str
    paper_id: str
    question_no: Optional[str] = None
    prompt: str
    type: Literal["mcq", "short", "numeric", "multi_step", "true_false", "match", "fill_blank"]
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    topic_ids: List[str]
    marks: Optional[float] = None
    answer_key: Optional[AnswerKey] = None
    solution_outline: Optional[Solution] = None
    provenance: Optional[SourceProvenance] = None

class TopicCoverageSummary(BaseModel):
    topic_id: str
    count: int
    marks_total: float
    years_seen: List[int] = []
    common_question_types: List[str] = []

# --- Rubrics ---
class Scoring(BaseModel):
    max_points: int = 10
    partial_credit_allowed: bool = True
    show_stepwise_feedback: bool = True

class Rubrics(BaseModel):
    mistake_types: List[str]
    scoring: Scoring
    hint_ladder: List[HintStep] = []

# --- Reference Library ---
class UsagePolicy(BaseModel):
    allowed: bool = True
    priority_order: List[str] = ["textbook", "past_papers", "reference"]
    must_cite_source: bool = True

class ReferenceLibrary(BaseModel):
    available: bool = False
    sources: List[ReferenceSource] = []
    usage_policy: Optional[UsagePolicy] = None

class PastPapersSection(BaseModel):
    available: bool = False
    exam_board: Optional[str] = None
    exam_name: Optional[str] = None
    papers: List[PastPaper] = []
    extracted_questions: List[PastPaperQuestion] = []
    topic_coverage_summary: List[TopicCoverageSummary] = []

class QuestionBank(BaseModel):
    by_topic: Dict[str, List[Question]]
    mixed_review: List[Question] = []
    diagnostic_quiz: List[Question] = []


# --- ROOT OBJECT ---
class TeachingPack(BaseModel):
    meta: Meta
    chapter_profile: ChapterProfile
    topic_map: List[Topic]
    topic_exam_weighting: List[TopicExamWeight] = []
    reference_library: Optional[ReferenceLibrary] = None
    past_papers: Optional[PastPapersSection] = None
    formulas: List[Formula] = []
    sessions: List[Session]
    question_bank: QuestionBank
    rubrics: Rubrics
