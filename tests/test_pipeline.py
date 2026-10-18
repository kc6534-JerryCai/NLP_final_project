from __future__ import annotations

import unittest

from news_pipeline.core import (
    classify_relevance,
    deduplicate,
    normalize_guardian,
    normalize_nyt,
    normalized_url,
)
from news_pipeline.settings import load_keyword_config, quoted_query


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_keyword_config()

    def test_queries_quote_phrases_and_use_or(self) -> None:
        query = quoted_query(["inflation", "consumer prices", "cost-of-living"])
        self.assertEqual(query, '(inflation OR "consumer prices" OR "cost-of-living")')

    def test_normalized_url_removes_tracking(self) -> None:
        self.assertEqual(
            normalized_url("https://www.example.com/news/item/?utm_source=x#top"),
            "https://example.com/news/item",
        )

    def test_nyt_metadata_is_not_full_text(self) -> None:
        doc = {
            "_id": "nyt://article/test",
            "web_url": "https://www.nytimes.com/test",
            "headline": {"main": "U.S. inflation slows as consumer prices ease"},
            "abstract": "Federal Reserve officials watched the change in prices.",
            "lead_paragraph": "American households saw some relief.",
            "pub_date": "2025-01-02T10:00:00+0000",
            "section_name": "Business",
            "keywords": [{"name": "glocations", "value": "United States"}],
            "type_of_material": "News",
        }
        record = normalize_nyt(doc, self.config, "2026-10-14T00:00:00+00:00")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["text_scope"], "headline_abstract_lead")
        self.assertEqual(record["body"], "")
        self.assertEqual(record["us_relevance"], "relevant")
        self.assertIn("C", record["matched_query_groups"])

    def test_guardian_body_and_direct_sentiment(self) -> None:
        item = {
            "id": "business/2025/jan/01/test",
            "webUrl": "https://www.theguardian.com/business/2025/jan/01/test",
            "webPublicationDate": "2025-01-01T12:00:00Z",
            "sectionName": "Business",
            "type": "article",
            "fields": {
                "headline": "US consumer confidence rises",
                "trailText": "The Conference Board reported a change.",
                "bodyText": "American consumers were more confident as the job market improved.",
                "firstPublicationDate": "2025-01-01T12:00:00Z",
                "lastModified": "2025-01-01T13:00:00Z",
            },
            "tags": [{"id": "world/usa", "webTitle": "United States", "type": "keyword"}],
        }
        record = normalize_guardian(item, self.config, "2026-10-14T00:00:00+00:00")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["text_scope"], "full_body")
        self.assertTrue(record["direct_sentiment_coverage"])
        self.assertEqual(record["us_relevance"], "relevant")

    def test_foreign_only_item_is_irrelevant(self) -> None:
        relevance, reason = classify_relevance(
            source="guardian",
            title="British economy enters recession",
            abstract="Households in the United Kingdom face higher prices.",
            body="",
            section="Business",
            keywords=[],
            geographic_tags=[],
            config=self.config,
        )
        self.assertEqual(relevance, "irrelevant")
        self.assertIn("Foreign focus", reason)

    def test_dedup_preserves_all_group_matches(self) -> None:
        base = {
            "article_id": "one", "source": "nyt", "url": "https://nytimes.com/a",
            "title": "US economy and inflation", "abstract": "", "body": "",
            "text_scope": "headline_only", "published_at": "2025-01-01T00:00:00Z",
            "updated_at": "", "retrieved_at": "2026-10-14T00:00:00Z", "section": "Business",
            "keywords": [], "geographic_tags": [], "matched_query_groups": ["A"],
            "us_relevance": "relevant", "relevance_reason": "test",
            "direct_sentiment_coverage": False, "article_type": "news", "duplicate_group_id": "",
        }
        second = {**base, "matched_query_groups": ["C"], "direct_sentiment_coverage": True}
        records = deduplicate([base, second])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["matched_query_groups"], ["A", "C"])
        self.assertTrue(records[0]["direct_sentiment_coverage"])


if __name__ == "__main__":
    unittest.main()
