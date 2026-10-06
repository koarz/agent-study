"""Core models and orchestration for the autonomous novel agent."""

from .engine import NovelAgent, NovelGenerationError
from .llm import (
    FatalLLMError,
    InvalidLLMResponse,
    LLMClient,
    OpenAICompatibleBackend,
    parse_json_object,
)
from .retrieval import (
    ChromaBGERetriever,
    ContextRetriever,
    LocalBGEEmbedder,
    RAGConfig,
    RAGDependencyError,
    RAGIndexError,
    chunk_text,
)

from .models import (
    Chapter,
    ChapterPlan,
    Character,
    Foreshadow,
    ModelValidationError,
    NovelBrief,
    NovelProject,
    StoryBible,
)
from .style import analyze_prose
from .style_profile import (
    DEFAULT_ANALYSIS_CHARS,
    STYLE_DIMENSIONS,
    StyleProfile,
    StyleProfileError,
    extract_style_profile,
    load_style_profile_selection,
    mix_style_profiles,
)

__all__ = [
    "Chapter",
    "ChapterPlan",
    "Character",
    "Foreshadow",
    "ModelValidationError",
    "NovelBrief",
    "NovelProject",
    "StoryBible",
    "NovelAgent",
    "NovelGenerationError",
    "FatalLLMError",
    "InvalidLLMResponse",
    "LLMClient",
    "OpenAICompatibleBackend",
    "parse_json_object",
    "ChromaBGERetriever",
    "ContextRetriever",
    "LocalBGEEmbedder",
    "RAGConfig",
    "RAGDependencyError",
    "RAGIndexError",
    "chunk_text",
    "analyze_prose",
    "DEFAULT_ANALYSIS_CHARS",
    "STYLE_DIMENSIONS",
    "StyleProfile",
    "StyleProfileError",
    "extract_style_profile",
    "load_style_profile_selection",
    "mix_style_profiles",
]
