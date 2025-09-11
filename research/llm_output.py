import os
import json
import requests
import pandas as pd
import statistics
from datetime import datetime

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
        #print('time windows:')
        #print(time_windows)
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

    #print('\n\n')
    #print('VRPTW system itinerary')
    #print(plan_itinerary)
    return  plan_itinerary, plan_itinerary_llm, time_windows


def calculate_indicators(plan_itinerary, plan_itinerary_llm, time_windows):
    """
    Calculate four indicators for comparing VRPTW and LLM itinerary systems:
    1. travel / travel_raw ratio (average and variation)
    2. service / real_service ratio for sites (average and variation)
    3. service / real_service ratio for food (average and variation)
    4. Number of sites/food out of service time
    """
    import statistics

    def calculate_travel_ratios(itinerary):
        """Calculate travel/travel_raw ratios"""
        ratios = []
        for i in range(len(itinerary) - 1):
            if itinerary[i].get('travel') and itinerary[i].get('travel_raw'):
                ratio = itinerary[i]['travel'] / itinerary[i]['travel_raw']
                ratios.append(ratio)
        return ratios

    def calculate_service_ratios(itinerary, metadata_type):
        """Calculate service/real_service ratios for specific metadata type"""
        ratios = []
        for poi in itinerary:
            if (poi.get('metadata') == metadata_type and
                poi.get('service') and poi.get('real_service') and
                poi['real_service'] > 0):
                ratio = poi['service'] / poi['real_service']
                ratios.append(ratio)
        return ratios

    def count_out_of_service_time(itinerary, time_windows):
        """Count sites/food that are out of service time"""
        count = 0
        for poi in itinerary:
            if poi.get('name') in time_windows and poi.get('arrival') is not None:
                time_window = time_windows[poi['name']]
                arrival = poi['arrival']
                service_time = poi.get('service', 0)

                # Handle None service time
                if service_time is None:
                    service_time = 0

                # Check if arrival time is before opening or departure time is after closing
                if (arrival < time_window[0] or
                    arrival + service_time > time_window[1]):
                    count += 1
        return count

    # Calculate indicators for both systems
    vrptw_indicators = {}
    llm_indicators = {}

    # 1. Travel ratios
    vrptw_travel_ratios = calculate_travel_ratios(plan_itinerary)
    llm_travel_ratios = calculate_travel_ratios(plan_itinerary_llm)

    vrptw_indicators['travel_ratio'] = {
        'average': statistics.mean(vrptw_travel_ratios) if vrptw_travel_ratios else 0,
        'variation': statistics.stdev(vrptw_travel_ratios) if len(vrptw_travel_ratios) > 1 else 0
    }

    llm_indicators['travel_ratio'] = {
        'average': statistics.mean(llm_travel_ratios) if llm_travel_ratios else 0,
        'variation': statistics.stdev(llm_travel_ratios) if len(llm_travel_ratios) > 1 else 0
    }

    # 2. Service ratios for sites
    vrptw_site_ratios = calculate_service_ratios(plan_itinerary, 'site')
    llm_site_ratios = calculate_service_ratios(plan_itinerary_llm, 'site')

    vrptw_indicators['site_service_ratio'] = {
        'average': statistics.mean(vrptw_site_ratios) if vrptw_site_ratios else 0,
        'variation': statistics.stdev(vrptw_site_ratios) if len(vrptw_site_ratios) > 1 else 0
    }

    llm_indicators['site_service_ratio'] = {
        'average': statistics.mean(llm_site_ratios) if llm_site_ratios else 0,
        'variation': statistics.stdev(llm_site_ratios) if len(llm_site_ratios) > 1 else 0
    }

    # 3. Service ratios for food
    vrptw_food_ratios = calculate_service_ratios(plan_itinerary, 'food')
    llm_food_ratios = calculate_service_ratios(plan_itinerary_llm, 'food')

    vrptw_indicators['food_service_ratio'] = {
        'average': statistics.mean(vrptw_food_ratios) if vrptw_food_ratios else 0,
        'variation': statistics.stdev(vrptw_food_ratios) if len(vrptw_food_ratios) > 1 else 0
    }

    llm_indicators['food_service_ratio'] = {
        'average': statistics.mean(llm_food_ratios) if llm_food_ratios else 0,
        'variation': statistics.stdev(llm_food_ratios) if len(llm_food_ratios) > 1 else 0
    }

    # 4. Out of service time count
    vrptw_indicators['out_of_service_count'] = count_out_of_service_time(plan_itinerary, time_windows)
    llm_indicators['out_of_service_count'] = count_out_of_service_time(plan_itinerary_llm, time_windows)

    return {
        'VRPTW': vrptw_indicators,
        'LLM': llm_indicators
    }

def test_indicators_with_example(plan_itinerary, plan_itinerary_llm, time_windows):
    # Calculate indicators
    indicators = calculate_indicators(plan_itinerary, plan_itinerary_llm, time_windows)

    print("=== INDICATOR ANALYSIS RESULTS ===")
    print()

    for system, data in indicators.items():
        print(f"--- {system} System ---")
        print(f"1. Travel/Travel_raw Ratio:")
        print(f"   Average: {data['travel_ratio']['average']:.4f}")
        print(f"   Variation: {data['travel_ratio']['variation']:.4f}")
        print()

        print(f"2. Service/Real_service Ratio (Sites):")
        print(f"   Average: {data['site_service_ratio']['average']:.4f}")
        print(f"   Variation: {data['site_service_ratio']['variation']:.4f}")
        print()

        print(f"3. Service/Real_service Ratio (Food):")
        print(f"   Average: {data['food_service_ratio']['average']:.4f}")
        print(f"   Variation: {data['food_service_ratio']['variation']:.4f}")
        print()

        print(f"4. Out of Service Time Count:")
        print(f"   Count: {data['out_of_service_count']}")
        print()
        print("-" * 50)
        print()

def calculation_matrix(iterations_per_category=100):
    preference_list = [
        "親子旅遊",
        "古蹟之旅",
        "冒險體驗",
        "購物旅遊",
        "生態旅遊",
        "藝術之旅",
    ]

    # Dictionary to store all results
    all_results = {}

    print(f"Starting comprehensive testing for {len(preference_list)} categories with {iterations_per_category} iterations each...")
    print(f"Total iterations: {len(preference_list) * iterations_per_category}")
    print("=" * 80)

    for category_idx, preference in enumerate(preference_list, 1):
        print(f"\n[{category_idx}/{len(preference_list)}] Testing category: {preference}")
        print("-" * 50)

        # Store results for this category
        category_results = {
            'VRPTW': {
                'travel_ratio_avg': [],
                'travel_ratio_var': [],
                'site_service_ratio_avg': [],
                'site_service_ratio_var': [],
                'food_service_ratio_avg': [],
                'food_service_ratio_var': [],
                'out_of_service_count': []
            },
            'LLM': {
                'travel_ratio_avg': [],
                'travel_ratio_var': [],
                'site_service_ratio_avg': [],
                'site_service_ratio_var': [],
                'food_service_ratio_avg': [],
                'food_service_ratio_var': [],
                'out_of_service_count': []
            }
        }

        successful_runs = 0

        for iteration in range(iterations_per_category):
            if (iteration + 1) % max(1, iterations_per_category // 5) == 0:  # Show progress every 20% of iterations
                print(f"  Progress: {iteration + 1}/{iterations_per_category} iterations completed")

            try:
                ans = planning(preference)
                if ans is not None:
                    plan_itinerary, plan_itinerary_llm, time_windows = ans
                    indicators = calculate_indicators(plan_itinerary, plan_itinerary_llm, time_windows)

                    # Store VRPTW results
                    category_results['VRPTW']['travel_ratio_avg'].append(indicators['VRPTW']['travel_ratio']['average'])
                    category_results['VRPTW']['travel_ratio_var'].append(indicators['VRPTW']['travel_ratio']['variation'])
                    category_results['VRPTW']['site_service_ratio_avg'].append(indicators['VRPTW']['site_service_ratio']['average'])
                    category_results['VRPTW']['site_service_ratio_var'].append(indicators['VRPTW']['site_service_ratio']['variation'])
                    category_results['VRPTW']['food_service_ratio_avg'].append(indicators['VRPTW']['food_service_ratio']['average'])
                    category_results['VRPTW']['food_service_ratio_var'].append(indicators['VRPTW']['food_service_ratio']['variation'])
                    category_results['VRPTW']['out_of_service_count'].append(indicators['VRPTW']['out_of_service_count'])

                    # Store LLM results
                    category_results['LLM']['travel_ratio_avg'].append(indicators['LLM']['travel_ratio']['average'])
                    category_results['LLM']['travel_ratio_var'].append(indicators['LLM']['travel_ratio']['variation'])
                    category_results['LLM']['site_service_ratio_avg'].append(indicators['LLM']['site_service_ratio']['average'])
                    category_results['LLM']['site_service_ratio_var'].append(indicators['LLM']['site_service_ratio']['variation'])
                    category_results['LLM']['food_service_ratio_avg'].append(indicators['LLM']['food_service_ratio']['average'])
                    category_results['LLM']['food_service_ratio_var'].append(indicators['LLM']['food_service_ratio']['variation'])
                    category_results['LLM']['out_of_service_count'].append(indicators['LLM']['out_of_service_count'])

                    successful_runs += 1

            except Exception as e:
                print(f"  Error in iteration {iteration + 1}: {str(e)}")
                continue

        print(f"  Completed: {successful_runs}/{iterations_per_category} successful runs for {preference}")

        # Calculate averages for this category
        category_averages = {}
        for system in ['VRPTW', 'LLM']:
            category_averages[system] = {}
            for metric in category_results[system]:
                if category_results[system][metric]:  # Check if list is not empty
                    category_averages[system][metric] = statistics.mean(category_results[system][metric])
                else:
                    category_averages[system][metric] = 0

        all_results[preference] = category_averages

    # Create DataFrame and export to CSV (Excel export removed)
    export_results_to_csv(all_results)

    return all_results






def export_results_to_csv(all_results):
    """Export results to CSV files in both wide and long formats for plotting 7 figures.

    Wide columns:
      - Category, System,
      - Travel_Avg, Travel_Std,
      - SiteService_Avg, SiteService_Std,
      - FoodService_Avg, FoodService_Std,
      - OutOfService_Count_Avg

    Long columns (tidy):
      - Category, System, Indicator, Metric, Value
        Indicator in {travel, site_service, food_service, out_of_service}
        Metric in {avg, std, count}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wide_filename = f"indicator_results_summary_{timestamp}.csv"
    long_filename = f"indicator_results_long_{timestamp}.csv"

    # Mapping: Chinese preference category -> English
    category_en_map = {
        '親子旅遊': 'Family Travel',
        '古蹟之旅': 'Historical Sites',
        '冒險體驗': 'Adventure Experience',
        '購物旅遊': 'Shopping Travel',
        '生態旅遊': 'Ecotourism',
        '藝術之旅': 'Art Journey',
    }

    # Prepare data for wide CSV
    wide_rows = []
    long_rows = []

    for category, systems in all_results.items():
        for system, metrics in systems.items():
            category_en = category_en_map.get(category, category)
            # Wide
            wide_row = {
                'Category': category,
                'Category_EN': category_en,
                'System': system,
                'Travel_Avg': metrics['travel_ratio_avg'],
                'Travel_Std': metrics['travel_ratio_var'],
                'SiteService_Avg': metrics['site_service_ratio_avg'],
                'SiteService_Std': metrics['site_service_ratio_var'],
                'FoodService_Avg': metrics['food_service_ratio_avg'],
                'FoodService_Std': metrics['food_service_ratio_var'],
                'OutOfService_Count_Avg': metrics['out_of_service_count'],
            }
            wide_rows.append(wide_row)

            # Long – 6 metrics for three indicators + count
            long_rows.extend([
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'travel',
                    'Metric': 'avg',
                    'Value': metrics['travel_ratio_avg'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'travel',
                    'Metric': 'std',
                    'Value': metrics['travel_ratio_var'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'site_service',
                    'Metric': 'avg',
                    'Value': metrics['site_service_ratio_avg'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'site_service',
                    'Metric': 'std',
                    'Value': metrics['site_service_ratio_var'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'food_service',
                    'Metric': 'avg',
                    'Value': metrics['food_service_ratio_avg'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'food_service',
                    'Metric': 'std',
                    'Value': metrics['food_service_ratio_var'],
                },
                {
                    'Category': category,
                    'Category_EN': category_en,
                    'System': system,
                    'Indicator': 'out_of_service',
                    'Metric': 'count',
                    'Value': metrics['out_of_service_count'],
                },
            ])

    # Save wide CSV
    pd.DataFrame(wide_rows).to_csv(wide_filename, index=False, encoding='utf-8-sig')
    # Save long CSV
    pd.DataFrame(long_rows).to_csv(long_filename, index=False, encoding='utf-8-sig')

    print(f"\n✅ Results exported to CSV (wide): {wide_filename}")
    print(f"✅ Results exported to CSV (long): {long_filename}")
    return wide_filename


def main(iterations_per_category=100):
    """
    Main function to run comprehensive testing across all preference categories.

    Args:
        iterations_per_category (int): Number of iterations to run for each category (default: 100)
    """
    result = calculation_matrix(iterations_per_category)


def quick_test(iterations_per_category=5):
    """
    Quick test function for development and debugging.

    Args:
        iterations_per_category (int): Number of iterations to run for each category (default: 5)
    """
    print(f"Running quick test with {iterations_per_category} iterations per category...")
    result = calculation_matrix(iterations_per_category)
    return result




if __name__ == "__main__":
    import sys

    # Check if iterations parameter is provided via command line
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
            print(f"Running with {iterations} iterations per category")
            main(iterations)
        except ValueError:
            print("Invalid iterations parameter. Using default 100 iterations per category.")
            main()
    else:
        print("Using default 100 iterations per category.")
        print("To specify different iterations: python llm_output.py <number_of_iterations>")
        main()
