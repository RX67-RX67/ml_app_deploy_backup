from pydantic import BaseModel
from typing import List
from datetime import datetime

class ItemInfo(BaseModel):
    item_id: str
    category: str
    crop_path: str
    embedding_path: str

    class Config:
        extra = "allow"   

class YoloEmbedResponse(BaseModel):
    timestamp: datetime
    num_items: int
    items: List[ItemInfo]
    message: str
