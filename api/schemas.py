from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., description="Age of the individual")
    workclass: Optional[str] = Field(None, alias="workclass")
    education: Optional[str] = Field(None, alias="education")
    education_num: Optional[int] = Field(None, alias="education-num")
    marital_status: Optional[str] = Field(None, alias="marital-status")
    occupation: Optional[str] = Field(None, alias="occupation")
    relationship: Optional[str] = Field(None, alias="relationship")
    race: Optional[str] = Field(None, alias="race")
    sex: Optional[str] = Field(None, alias="sex")
    capital_gain: Optional[int] = Field(None, alias="capital-gain")
    capital_loss: Optional[int] = Field(None, alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: Optional[str] = Field(None, alias="native-country")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    id: str
    status: str = "success"
    is_anomaly: Optional[bool] = False
    anomaly_reasons: Optional[list[str]] = None
    imputed_features: Optional[list[str]] = None
