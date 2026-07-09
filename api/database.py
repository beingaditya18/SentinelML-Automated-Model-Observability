import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "live")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'sentinelml.db')}"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InferenceRecord(Base):
    __tablename__ = "inferences"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Feature columns
    age = Column(Integer, nullable=True)
    workclass = Column(String, nullable=True)
    education = Column(String, nullable=True)
    education_num = Column(Integer, nullable=True)
    marital_status = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    relationship = Column(String, nullable=True)
    race = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    capital_gain = Column(Integer, nullable=True)
    capital_loss = Column(Integer, nullable=True)
    hours_per_week = Column(Integer, nullable=True)
    native_country = Column(String, nullable=True)
    
    # Outputs
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    feedback_label = Column(Integer, nullable=True)
    imputed_features = Column(String, nullable=True)
    is_anomaly = Column(Integer, default=0)
    anomaly_reasons = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import text
    with engine.connect() as conn:
        for col_name, col_type in [
            ("feedback_label", "INTEGER"),
            ("imputed_features", "TEXT"),
            ("is_anomaly", "INTEGER"),
            ("anomaly_reasons", "TEXT")
        ]:
            try:
                conn.execute(text(f"ALTER TABLE inferences ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
