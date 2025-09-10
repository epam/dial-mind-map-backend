import json
from datetime import datetime, timezone
from typing import List, Optional

from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import Generation
from langchain_community.cache import SQLAlchemyCache
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    and_,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class LLMCacheWithTimestamp(Base):
    """
    SQLAlchemy model for a cache with a timestamp, supporting multiple
    generations.
    """

    __tablename__ = "llm_cache_with_timestamp"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)
    response = Column(String, nullable=False)
    last_accessed_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class TimestampedSQLiteCache(SQLAlchemyCache):
    """
    A cache that mirrors LangChain's standard SQLAlchemyCache but adds a
    'last_accessed_at' timestamp to each entry for time-based
    invalidation.
    """

    def __init__(self, database_path: str):
        """Initialize by creating the engine and our custom table."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine, cache_schema=LLMCacheWithTimestamp)

    def lookup(
        self, prompt: str, llm_string: str
    ) -> Optional[List[Generation]]:
        """
        Look up a prompt in the cache. If found, all matching entries
        will have their 'last_accessed_at' timestamp updated before
        being returned.
        """
        with Session(self.engine) as session:
            # CORRECTED: Use and_() for type-safe, modern SQLAlchemy
            # query
            stmt = (
                select(self.cache_schema.response)
                .where(
                    and_(
                        self.cache_schema.prompt == prompt,
                        self.cache_schema.llm == llm_string,
                    )
                )
                .order_by(self.cache_schema.idx)
            )
            rows = session.execute(stmt).fetchall()

            if rows:
                # If we found rows, update their timestamp to now.
                update_stmt = (
                    update(self.cache_schema).where(
                        and_(
                            self.cache_schema.prompt == prompt,
                            self.cache_schema.llm == llm_string,
                        )
                    )
                    # CORRECTED: Use timezone-aware UTC datetime
                    .values(last_accessed_at=datetime.now(timezone.utc))
                )
                session.execute(update_stmt)
                session.commit()

                # CORRECTED: Catch specific deserialization errors
                try:
                    return [loads(row[0]) for row in rows]
                except (json.JSONDecodeError, TypeError):
                    # Fallback for older, plain-text cache formats
                    return [Generation(text=row[0]) for row in rows]
        return None

    def update(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ) -> None:
        """
        Update the cache with new entries using an efficient "upsert"
        (update or insert) operation for each generation.
        """
        with Session(self.engine) as session, session.begin():
            for i, gen in enumerate(return_val):
                item = self.cache_schema(
                    prompt=prompt,
                    llm=llm_string,
                    idx=i,
                    response=dumps(gen),
                    # The timestamp is set automatically by the column's
                    # default
                )
                # session.merge handles both INSERT and UPDATE
                # efficiently.
                session.merge(item)
