"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Online prediction audit trail (replaces CSV append)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    credit_score: Mapped[int] = mapped_column(Integer)
    geography: Mapped[str] = mapped_column(String(32))
    gender: Mapped[str] = mapped_column(String(16))
    age: Mapped[int] = mapped_column(Integer)
    tenure: Mapped[int] = mapped_column(Integer)
    balance: Mapped[float] = mapped_column(Float)
    num_of_products: Mapped[int] = mapped_column(Integer)
    has_cr_card: Mapped[int] = mapped_column(Integer)
    is_active_member: Mapped[int] = mapped_column(Integer)
    estimated_salary: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    probability_exited: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))


class TrainingRun(Base):
    """One row per `train_and_save()` — manifest + classification reports (no JSON files)."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    classification_reports: Mapped[dict[str, Any]] = mapped_column(JSON)
