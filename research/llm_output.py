import os
import json
import requests

from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from routing import main as routing_main

from collections import defaultdict, deque
from typing import Dict, Any, List
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

def plan_itinerary_with_routing(poi_list: List[str]):
    return routing_main(1, poi_list, 480, 1200)

def plan_itinerary_with_llm(poi_list: List[str]):
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
        "Use '嘉義火車站' as the starting point and '嘉義火車站' as the ending point. "
        "Follow the provided JSON schema exactly."),
        ("human", """Here is the list of POIs:

    {poi_list}

    Format them into the JSON schema below:
    {format_instructions}""")
    ])


    # --- 7. Format messages ---
    messages = prompt.format_messages(
        poi_list=", ".join(poi_list),
        format_instructions=parser.get_format_instructions()
    )

    # --- 8. Run GPT and parse ---
    response = llm.invoke(messages)
    parsed = parser.parse(response.content)
    result = parsed.model_dump()['result']   # Pydantic v2 way
    # print(result)
    return result
    # --- 9. Save JSON file ---
    #with open("pois.json", "w", encoding="utf-8") as f:
    #     json.dump(parsed.dict(), f, ensure_ascii=False, indent=2)

    # print("✅ JSON file created: pois.json")

def get_recommend_poi(preference:str):
    url = "http://localhost:5001/recommend"
    headers = {"Content-Type": "application/json"}
    payload = {
        "day": 1,
        "preference": preference
    }

    response = requests.post(url, json=payload, headers=headers)
    try:
        #print(response.json()['result'], type(response.json()['result']))
        return response.json()['result']
    except ValueError:
        print("Response is not JSON")
    #return response.json()['result']

def get_sites_data():
    with open('data/sitesData.json', 'r') as file:
        sitesData = json.load(file)
    return sitesData

def get_time_matrix(sites, time_matrix):
    d = {}
    n = len(sites)
    for i in range(n):
        d[sites[i]] = {}
        for j in range(n):
            d[sites[i]][sites[j]] = time_matrix[i][j]["raw"]
    return d

def planning(preference):
    recommend_poi = get_recommend_poi(preference)

    routing_res = plan_itinerary_with_routing(recommend_poi)
    if routing_res and isinstance(routing_res, tuple) and len(routing_res) == 4 and routing_res[0] is not None:
        plan_itinerary, sites, raw_time_matrix, time_windows = routing_res
        print('time windows:')
        print(time_windows)
    else:
        print("⚠️ VRPTW found no solution; continuing with LLM-only itinerary.")
        return None
        # sites = [p["name"] if isinstance(p, dict) and "name" in p else str(p) for p in recommend_poi]
        # plan_itinerary, raw_time_matrix = [], None

    plan_itinerary_llm = plan_itinerary_with_llm(sites)

    sitesData = get_sites_data()
    time_matrix = get_time_matrix(sites, raw_time_matrix)


    for poi in plan_itinerary_llm:
        poi['metadata'] = sitesData[poi['name']]['metadata']['type']
        time_spent = sitesData[poi['name']]['time_spent']
        poi['real_service'] = sum(time_spent) // len(time_spent)
    for poi in plan_itinerary:
        poi['metadata'] = sitesData[poi['name']]['metadata']['type']
        time_spent = sitesData[poi['name']]['time_spent']
        poi['real_service'] = sum(time_spent) // len(time_spent)

    for i in range(len(plan_itinerary_llm)-1):
        cur_poi = plan_itinerary_llm[i]['name']
        nxt_poi = plan_itinerary_llm[i+1]['name']
        plan_itinerary_llm[i]['travel_raw'] = time_matrix[cur_poi][nxt_poi]

    print('\n\n')
    print('VRPTW system itinerary')
    print(plan_itinerary)
    print('\nLLM system itinerary')
    print(plan_itinerary_llm)

def calculation_matrix():
    preference_list =   [
            "親子旅遊",
            "文化古蹟",
            "自然風光",
            "冒險體驗",
            "購物旅遊",
            "療癒放鬆",
            "生態旅遊",
            "藝術之旅",
        ]

    example = "生態旅遊"
    for i in range(5):
        ans = planning(example)
        print(f'The answer of {i}: {ans}')




def main():
    calculation_matrix()



if __name__ == "__main__":
    main()

time windows:
{'嘉義火車站': (290, 1380), '嘉義樹木園': (0, 1440), '嘉義市立博物館': (540, 1020), '嘉義製材所': (540, 1020), 'KANO遊客中心': (600, 1080), '嘉義文化創意產業園區': (600, 1080), '貳陸陸杉space': (540, 1020), '嘉木居': (600, 1020)}



VRPTW system itinerary
[{'name': '嘉義火車站', 'arrival': 0, 'service': 0, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 12, 'metadata': 'site', 'real_service': 0}, {'name': '貳陸陸杉space', 'arrival': 100, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 6, 'metadata': 'food', 'real_service': 37}, {'name': '嘉木居', 'arrival': 160, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 8, 'metadata': 'food', 'real_service': 37}, {'name': '嘉義文化創意產業園區', 'arrival': 210, 'service': 70, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 10, 'metadata': 'site', 'real_service': 70}, {'name': '嘉義製材所', 'arrival': 290, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 11, 'metadata': 'site', 'real_service': 47}, {'name': 'KANO遊客中心', 'arrival': 360, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 4, 'metadata': 'site', 'real_service': 47}, {'name': '嘉義樹木園', 'arrival': 420, 'service': 60, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 10, 'metadata': 'site', 'real_service': 60}, {'name': '嘉義市立博物館', 'arrival': 490, 'service': 120, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 8, 'metadata': 'site', 'real_service': 120}, {'name': '嘉義火車站', 'arrival': 1100, 'end_node': True, 'vehicle': 0, 'metadata': 'site', 'real_service': 0}]

LLM system itinerary
[{'arrival': 480, 'name': '嘉義火車站', 'service': None, 'travel': 10, 'metadata': 'site', 'real_service': 0, 'travel_raw': 14}, {'arrival': 490, 'name': '嘉義樹木園', 'service': 60, 'travel': 15, 'metadata': 'site', 'real_service': 60, 'travel_raw': 10}, {'arrival': 565, 'name': '嘉義市立博物館', 'service': 45, 'travel': 10, 'metadata': 'site', 'real_service': 120, 'travel_raw': 1}, {'arrival': 620, 'name': '嘉義製材所', 'service': 30, 'travel': 10, 'metadata': 'site', 'real_service': 47, 'travel_raw': 11}, {'arrival': 660, 'name': 'KANO遊客中心', 'service': 60, 'travel': 20, 'metadata': 'site', 'real_service': 47, 'travel_raw': 16}, {'arrival': 740, 'name': '嘉義文化創意產業園區', 'service': 90, 'travel': 15, 'metadata': 'site', 'real_service': 70, 'travel_raw': 14}, {'arrival': 845, 'name': '貳陸陸杉space', 'service': 45, 'travel': 10, 'metadata': 'food', 'real_service': 37, 'travel_raw': 6}, {'arrival': 900, 'name': '嘉木居', 'service': 60, 'travel': 20, 'metadata': 'food', 'real_service': 37, 'travel_raw': 6}, {'arrival': 980, 'name': '嘉義火車站', 'service': None, 'travel': None, 'metadata': 'site', 'real_service': 0}]
The answer of 0: None
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No solution found!
⚠️ VRPTW found no solution; continuing with LLM-only itinerary.
The answer of 1: None
time windows:
{'嘉義火車站': (290, 1380), '愛木村休閒觀光工廠': (540, 1050), '大溪厝水環境教育園區': (0, 1440), '埤子頭植物園': (510, 960), '嘉義樹木園': (0, 1440), '嘉大植物園': (0, 1440)}



VRPTW system itinerary
[{'name': '嘉義火車站', 'arrival': 0, 'service': 0, 'vehicle': 0, 'end_node': False, 'travel': 30, 'travel_raw': 30, 'metadata': 'site', 'real_service': 0}, {'name': '嘉大植物園', 'arrival': 30, 'service': 55, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 15, 'metadata': 'site', 'real_service': 55}, {'name': '嘉義樹木園', 'arrival': 105, 'service': 60, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 17, 'metadata': 'site', 'real_service': 60}, {'name': '愛木村休閒觀光工廠', 'arrival': 185, 'service': 90, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 17, 'metadata': 'site', 'real_service': 90}, {'name': '大溪厝水環境教育園區', 'arrival': 295, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 30, 'travel_raw': 21, 'metadata': 'site', 'real_service': 50}, {'name': '埤子頭植物園', 'arrival': 375, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 8, 'metadata': 'site', 'real_service': 50}, {'name': '嘉義火車站', 'arrival': 915, 'end_node': True, 'vehicle': 0, 'metadata': 'site', 'real_service': 0}]

LLM system itinerary
[{'arrival': 480, 'name': '嘉義火車站', 'service': None, 'travel': 15, 'metadata': 'site', 'real_service': 0, 'travel_raw': 15}, {'arrival': 495, 'name': '愛木村休閒觀光工廠', 'service': 60, 'travel': 20, 'metadata': 'site', 'real_service': 90, 'travel_raw': 17}, {'arrival': 575, 'name': '大溪厝水環境教育園區', 'service': 45, 'travel': 25, 'metadata': 'site', 'real_service': 50, 'travel_raw': 21}, {'arrival': 645, 'name': '埤子頭植物園', 'service': 60, 'travel': 15, 'metadata': 'site', 'real_service': 50, 'travel_raw': 13}, {'arrival': 720, 'name': '嘉義樹木園', 'service': 90, 'travel': 10, 'metadata': 'site', 'real_service': 60, 'travel_raw': 15}, {'arrival': 820, 'name': '嘉大植物園', 'service': 60, 'travel': 20, 'metadata': 'site', 'real_service': 55, 'travel_raw': 30}, {'arrival': 900, 'name': '嘉義火車站', 'service': None, 'travel': None, 'metadata': 'site', 'real_service': 0}]
The answer of 2: None
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No duration_in_traffic in the element, something go wrong!
No solution found!
⚠️ VRPTW found no solution; continuing with LLM-only itinerary.
The answer of 3: None
time windows:
{'嘉義火車站': (290, 1380), '嘉義樹木園': (0, 1440), '嘉義製材所': (540, 1020), '嘉義市立博物館': (540, 1020), 'KANO遊客中心': (600, 1080), '射日塔': (540, 1080), '貳陸陸杉space': (540, 1020), 'Morikoohii 森咖啡': (600, 1080)}



VRPTW system itinerary
[{'name': '嘉義火車站', 'arrival': 0, 'service': 0, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 14, 'metadata': 'site', 'real_service': 0}, {'name': 'KANO遊客中心', 'arrival': 170, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 4, 'metadata': 'site', 'real_service': 47}, {'name': '嘉義樹木園', 'arrival': 230, 'service': 60, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 6, 'metadata': 'site', 'real_service': 60}, {'name': '射日塔', 'arrival': 300, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 5, 'metadata': 'site', 'real_service': 50}, {'name': '貳陸陸杉space', 'arrival': 360, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 7, 'metadata': 'food', 'real_service': 37}, {'name': '嘉義製材所', 'arrival': 410, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 1, 'metadata': 'site', 'real_service': 47}, {'name': '嘉義市立博物館', 'arrival': 470, 'service': 120, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 3, 'metadata': 'site', 'real_service': 120}, {'name': 'Morikoohii 森咖啡', 'arrival': 600, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 10, 'metadata': 'food', 'real_service': 40}, {'name': '嘉義火車站', 'arrival': 1130, 'end_node': True, 'vehicle': 0, 'metadata': 'site', 'real_service': 0}]

LLM system itinerary
[{'arrival': 480, 'name': '嘉義火車站', 'service': None, 'travel': 10, 'metadata': 'site', 'real_service': 0, 'travel_raw': 15}, {'arrival': 490, 'name': '嘉義樹木園', 'service': 60, 'travel': 15, 'metadata': 'site', 'real_service': 60, 'travel_raw': 10}, {'arrival': 565, 'name': '嘉義製材所', 'service': 45, 'travel': 10, 'metadata': 'site', 'real_service': 47, 'travel_raw': 1}, {'arrival': 620, 'name': '嘉義市立博物館', 'service': 60, 'travel': 10, 'metadata': 'site', 'real_service': 120, 'travel_raw': 10}, {'arrival': 690, 'name': 'KANO遊客中心', 'service': 30, 'travel': 20, 'metadata': 'site', 'real_service': 47, 'travel_raw': 3}, {'arrival': 740, 'name': '射日塔', 'service': 45, 'travel': 15, 'metadata': 'site', 'real_service': 50, 'travel_raw': 5}, {'arrival': 800, 'name': '貳陸陸杉space', 'service': 30, 'travel': 10, 'metadata': 'food', 'real_service': 37, 'travel_raw': 6}, {'arrival': 840, 'name': 'Morikoohii 森咖啡', 'service': 60, 'travel': 20, 'metadata': 'food', 'real_service': 40, 'travel_raw': 10}, {'arrival': 920, 'name': '嘉義火車站', 'service': None, 'travel': None, 'metadata': 'site', 'real_service': 0}]
The answer of 4: None
