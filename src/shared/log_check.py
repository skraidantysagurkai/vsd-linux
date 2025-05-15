from pydantic import BaseModel
from typing import List, Optional

class Log(BaseModel):
    timestamp: str
    success: int
    uid: str
    pid: str
    command: str
    arguments: Optional[List[str]]
    
    