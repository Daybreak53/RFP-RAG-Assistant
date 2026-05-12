from pydantic import BaseModel

class DocumentResponse(BaseModel):
    id: str
    name: str
    size: int
    status: str

class DeleteResponse(BaseModel):
    status: str
    id: str