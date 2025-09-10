from distance_matrix import travel_time
from vrptw import routing
from data_time import convert_time_windows, get_stay_time, get_real_stay_time

def create_data_model(day:int, sites: list, start_time: int, end_time:int):
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
    return data



def main(day:int, sites:str, start_time:int = 480, end_time:int = 1200, start_point = '嘉義火車站'):
    sites = sites.split(', ')
    sites = [start_point] + [site for site in sites if site != start_point]
    #print(sites)
    data = create_data_model(day, sites, start_time, end_time)
    #print(data)
    data2 = routing(data)
    return data2



if __name__ == "__main__":
    main(1,  "嘉義壺豆花, 嘉義製材所, 貳陸陸杉space, 嘉義公園, 拾間文化, 嘉義市環市自行車道, 嘉義樹木園, 射日塔", 480, 1200 )

