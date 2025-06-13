from pydantic import BaseModel, SecretStr


# TODO: Migrate to pydantic v2 and langchain 0.3
class DialConfig(BaseModel):
    dial_url: str
    api_key: SecretStr
