"""
Unit tests for backend/models.py — Alert ORM model.
No database connection required; only validates model structure.
"""
import pytest
from sqlalchemy import Integer, String, Float, Boolean, JSON, DateTime

from models import Alert
from database import Base


class TestAlertModel(): 
    """Validate the Alert ORM model definition."""

    def test_tablename(self):
        assert Alert.__tablename__ == "alerts"

    def test_inherits_base(self):
        assert issubclass(Alert, Base)

    def test_has_all_expected_columns(self):
        col_names = {c.name for c in Alert.__table__.columns}
        expected = {"id",
         "timestamp",
          "created_at",
           "source",
            "is_attack",

                    "confidence", "attack_type", "original_features"}
        assert expected == col_names

    def test_id_is_primary_key(self):
        col = Alert.__table__.c.id
        assert col.primary_key

    def test_column_types(self):
        cols = {c.name: type(c.type) for c in Alert.__table__.columns}
        assert cols["id"] == Integer
        assert cols["timestamp"] == Float
        assert cols["created_at"] == DateTime
        assert cols["source"] == String
        assert cols["is_attack"] == Boolean
        assert cols["confidence"] == Float
        assert cols["attack_type"] == String
        assert cols["original_features"] == JSON

    def test_indexed_columns(self):
        """source and attack_type should be indexed for query performance."""
        indexed = {idx.columns.keys()[0] for idx in Alert.__table__.indexes}
        assert "source" in indexed
        assert "attack_type" in indexed

    def test_instance_creation(self):
        """Verify an Alert object can be instantiated with typical data."""
        alert = Alert(
            timestamp=1234567890.0,
            source="10.0.0.1",
            is_attack=True,
            confidence=0.99,
            attack_type="SQL Injection",
            original_features={"payload": "test"},
        )
        assert alert.source == "10.0.0.1"
        assert alert.is_attack is True
        assert alert.confidence == 0.99
