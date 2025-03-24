from pydantic import BaseModel
from typing import Dict, Any, Optional

class InputSchema(BaseModel):
    func_name: str
    func_input_data: Optional[Dict[str, Any]] = None
