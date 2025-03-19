from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any

class InputSchema(BaseModel):
    func_name: Literal[
        "initialize", 
        "run_query", 
        "add_data", 
        "delete_table", 
        "delete_row", 
        "list_rows",
        "ingest_knowledge",
        "search",
        "get_by_id",
        "get_findings",
        "clear"
    ]
    func_input_data: Optional[Dict[str, Any]] = None

class RetrievedMemory(BaseModel):
    chunk: str
    chunk_start: int
    chunk_end: int
    full_text: str
    metadata: Dict[str, Any]
    source: str
    timestamp: str