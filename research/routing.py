from distance_matrix import travel_time
from vrptw import routing
from data_time import convert_time_windows, get_stay_time, get_real_stay_time
import json

def get_sites_data():
    with open('data/sitesData.json', 'r') as file:
        sitesData = json.load(file)
    return sitesData


def create_data_model(day:int, sites: list, start_time: int, end_time:int):
    sitesData = get_sites_data()
    data = {}
    sites, data["time_windows"] = convert_time_windows(sites)
    sites, data["time_matrix"], data['time_windows'] = travel_time(sites, data["time_windows"])
    data["numlocations_"] = len(data["time_matrix"])
    data["name"] = sites
    data["service"] = get_stay_time(sites)
    data["num_vehicles"] = day
    data["service_unit"] = 1
    data["depot"] = 0
    data["start_time"] = start_time
    data["end_time"] = end_time
    data["real_service"] = get_real_stay_time(sites)
    return data, sites



def main(day:int, sites:str, start_time:int = 480, end_time:int = 1200, start_point = '嘉義火車站'):
    #sites = sites.split(', ')
    sites = [start_point] + [site for site in sites if site != start_point]
    #print(sites)
    data, verify_sites = create_data_model(day, sites, start_time, end_time)
    #print(data)
    data2 = routing(data)
    time_windows: dict() = {}
    for i in range(len(verify_sites)):
        time_windows[verify_sites[i]] = data['time_windows'][i]

    return data2, verify_sites, data["time_matrix"], time_windows



if __name__ == "__main__":
    main(1,  ["嘉義壺豆花", "嘉義製材所", "貳陸陸杉space", "嘉義公園", "拾間文化", "嘉義市環市自行車道", "嘉義樹木園", "射日塔"], 480, 1200 )

