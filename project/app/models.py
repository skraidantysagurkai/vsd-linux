from pydantic import BaseModel
from typing import List, Optional
  
class Log(BaseModel):
    timestamp: float
    success: int
    uid: int
    euid: int 
    syscall: int
    ppid: int
    pid: int
    command: str
    arguments: Optional[List[str]]
    CWD: Optional[str]


class TestingLog(BaseModel):
    id: int
    target: float
    content: Log