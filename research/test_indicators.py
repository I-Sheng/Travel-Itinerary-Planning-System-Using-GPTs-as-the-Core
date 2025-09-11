#!/usr/bin/env python3
"""
Test script for calculating the four indicators for VRPTW vs LLM itinerary comparison
"""

import statistics

def calculate_indicators(plan_itinerary, plan_itinerary_llm, time_windows):
    """
    Calculate four indicators for comparing VRPTW and LLM itinerary systems:
    1. travel / travel_raw ratio (average and variation)
    2. service / real_service ratio for sites (average and variation)
    3. service / real_service ratio for food (average and variation)
    4. Number of sites/food out of service time
    """
    
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

def test_indicators_with_example():
    """Test the indicator calculation with the provided example data"""
    
    # Example time windows
    time_windows = {
        '嘉義火車站': (290, 1380), 
        '嘉義樹木園': (0, 1440), 
        '嘉義製材所': (540, 1020), 
        '嘉義市立博物館': (540, 1020), 
        'KANO遊客中心': (600, 1080), 
        '射日塔': (540, 1080), 
        '貳陸陸杉space': (540, 1020), 
        'Morikoohii 森咖啡': (600, 1080)
    }
    
    # VRPTW system itinerary
    plan_itinerary = [
        {'name': '嘉義火車站', 'arrival': 0, 'service': 0, 'vehicle': 0, 'end_node': False, 'travel': 20, 'travel_raw': 14, 'metadata': 'site', 'real_service': 0}, 
        {'name': 'KANO遊客中心', 'arrival': 170, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 4, 'metadata': 'site', 'real_service': 47}, 
        {'name': '嘉義樹木園', 'arrival': 230, 'service': 60, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 6, 'metadata': 'site', 'real_service': 60}, 
        {'name': '射日塔', 'arrival': 300, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 5, 'metadata': 'site', 'real_service': 50}, 
        {'name': '貳陸陸杉space', 'arrival': 360, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 7, 'metadata': 'food', 'real_service': 37}, 
        {'name': '嘉義製材所', 'arrival': 410, 'service': 50, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 1, 'metadata': 'site', 'real_service': 47}, 
        {'name': '嘉義市立博物館', 'arrival': 470, 'service': 120, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 3, 'metadata': 'site', 'real_service': 120}, 
        {'name': 'Morikoohii 森咖啡', 'arrival': 600, 'service': 40, 'vehicle': 0, 'end_node': False, 'travel': 10, 'travel_raw': 10, 'metadata': 'food', 'real_service': 40}, 
        {'name': '嘉義火車站', 'arrival': 1130, 'end_node': True, 'vehicle': 0, 'metadata': 'site', 'real_service': 0}
    ]
    
    # LLM system itinerary
    plan_itinerary_llm = [
        {'arrival': 480, 'name': '嘉義火車站', 'service': None, 'travel': 10, 'metadata': 'site', 'real_service': 0, 'travel_raw': 15}, 
        {'arrival': 490, 'name': '嘉義樹木園', 'service': 60, 'travel': 15, 'metadata': 'site', 'real_service': 60, 'travel_raw': 10}, 
        {'arrival': 565, 'name': '嘉義製材所', 'service': 45, 'travel': 10, 'metadata': 'site', 'real_service': 47, 'travel_raw': 1}, 
        {'arrival': 620, 'name': '嘉義市立博物館', 'service': 60, 'travel': 10, 'metadata': 'site', 'real_service': 120, 'travel_raw': 10}, 
        {'arrival': 690, 'name': 'KANO遊客中心', 'service': 30, 'travel': 20, 'metadata': 'site', 'real_service': 47, 'travel_raw': 3}, 
        {'arrival': 740, 'name': '射日塔', 'service': 45, 'travel': 15, 'metadata': 'site', 'real_service': 50, 'travel_raw': 5}, 
        {'arrival': 800, 'name': '貳陸陸杉space', 'service': 30, 'travel': 10, 'metadata': 'food', 'real_service': 37, 'travel_raw': 6}, 
        {'arrival': 840, 'name': 'Morikoohii 森咖啡', 'service': 60, 'travel': 20, 'metadata': 'food', 'real_service': 40, 'travel_raw': 10}, 
        {'arrival': 920, 'name': '嘉義火車站', 'service': None, 'travel': None, 'metadata': 'site', 'real_service': 0}
    ]
    
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

if __name__ == "__main__":
    test_indicators_with_example()
