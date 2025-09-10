import os
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# --- 1. Load .env variables ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("⚠️ OPENAI_API_KEY not found in .env file")

# --- 2. Define Pydantic models ---
class POI(BaseModel):
    arrival: Optional[int] = None
    name: str
    service: Optional[int] = None
    travel: Optional[int] = None

class POIResult(BaseModel):
    result: List[POI]

# --- 3. Initialize parser ---
parser = PydanticOutputParser(pydantic_object=POIResult)

# --- 4. Initialize model (uses env var automatically) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

# --- 5. Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a travel planner. Given a list of POIs, create a feasible itinerary in structured JSON. "
     "Arrival time must be expressed in minutes, where 480 = 8:00 AM and 1200 = 8:00 PM. "
     "Travel represents the traffic time between two POIs, and service represents the stay time at a POI. "
     "The itinerary must include the order of visits and clearly mark start/end nodes. "
     "Follow the provided JSON schema exactly."),
    ("human", """Here is the list of POIs:

{poi_list}

Format them into the JSON schema below:
{format_instructions}""")
])



# --- 6. Example variable list ---
poi_list = [
    "嘉義火車站",
    "嘉義市環市自行車道",
    "貳陸陸杉space",
    "嘉義製材所",
    "嘉義公園",
    "射日塔",
    "嘉義文化創意產業園區",
    "嘉人酒場",
    "嘉義火車站"
]

# --- 7. Format messages ---
messages = prompt.format_messages(
    poi_list=", ".join(poi_list),
    format_instructions=parser.get_format_instructions()
)

# --- 8. Run GPT and parse ---
response = llm.invoke(messages)
parsed = parser.parse(response.content)

# --- 9. Save JSON file ---
with open("pois.json", "w", encoding="utf-8") as f:
    json.dump(parsed.dict(), f, ensure_ascii=False, indent=2)

print("✅ JSON file created: pois.json")

