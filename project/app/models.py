from pydantic import BaseModel
from typing import List, Optional

# class Log(BaseModel):
#     timestamp: str
#     success: int
#     uid: str
#     pid: str
#     command: str
#     is_bash: int
#     arguments: Optional[List[str]]
    
    
class Log(BaseModel):
    timestamp: float
    success: int
    uid: str
    euid: str 
    syscall: str
    ppid: str
    pid: str
    command: str
    arguments: Optional[List[str]]
    is_bash: int
    CWD: Optional[str]
    
class LogPrompt(BaseModel):
    command: Log
    history: List[Log]
    