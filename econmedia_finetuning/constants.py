TOPIC_LABELS = [
    "prices_inflation",
    "employment",
    "income_purchasing_power",
    "housing",
    "consumption",
    "growth_outlook",
    "economic_policy",
    "household_finance",
]

TONE_LABELS = [
    "positive",
    "negative",
    "neutral",
    "mixed",
    "insufficient_information",
]

PRIMARY_TOPIC_LABELS = TOPIC_LABELS + ["multiple"]

T5_INSTRUCTION = (
    "Classify this U.S. economic news article. Return exactly one line in this "
    "format: primary_topic: <label>; tone: <label>. "
    "Use only the allowed labels and do not explain."
)
