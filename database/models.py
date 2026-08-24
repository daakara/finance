"""SQLAlchemy Database Models for Target Architecture Migration."""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class AssetMetadataModel(Base):
    """Asset core metadata table."""
    __tablename__ = "asset_metadata"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    asset_name = Column(String(255))
    asset_type = Column(String(50), nullable=False)
    sector = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyScreeningRecordModel(Base):
    """Daily Hidden Gem screening history table."""
    __tablename__ = "daily_screening_records"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    composite_score = Column(Float, nullable=False)
    risk_rating = Column(String(50), nullable=False)
    primary_catalyst = Column(Text)
    raw_payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

