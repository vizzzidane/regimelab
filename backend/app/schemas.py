from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PipelineRunResponse(BaseModel):
    status: str
    message: str