import json
import re
import random

def getJson(fileName: str):
    with open(fileName, 'r') as file:
        json_data = file.read()
        data = json.loads(json_data)
    return data


def get_stay_time(sites: list):
    data = getJson('./data/sitesData.json')
    arr = []
    for site in sites:
        timelist = data[site]['time_spent']
        time = sum(timelist) // len(timelist)
        remainder = time % 5
        if remainder != 0:
            time += 5 - remainder
        arr.append(time)

    return arr

def get_real_stay_time(sites: list):
    data = getJson('./data/sitesData.json')
    arr = []
    for site in sites:
        timelist = data[site]['time_spent']
        arr.append(sum(timelist) // len(timelist))
    return arr

def convert_time_windows(sites):
    data = getJson('./data/sitesData.json')
    time_windows:list = []
    sites2 = sites.copy()
    for site in sites:
        weekend = data[site]["opening_hours"][-2:]
        sat = time_to_minutes(weekend[0])
        sun = time_to_minutes(weekend[1])
        if sum(sat) < sum(sun):
            tmp = sat
        else:
            tmp = sun
        if sum(tmp) == 0:
            sites2 = [si for si in sites2 if si != site]
        else:
            time_windows.append(tmp)
            
    return sites2, time_windows


def time_to_minutes(time_str):

    time_str = time_str.upper()
    if "CLOSED" in time_str:
        return (0, 0)

    if "OPEN 24" in time_str:
        return (0, 1440)

    # Regular expression to match the time ranges
    time_pattern = r'(\d{1,2}):(\d{2})\s*([AP]M)?'

    # Extract time ranges
    time_ranges = re.findall(time_pattern, time_str)

    if not time_ranges:
        raise ValueError("No valid time ranges found in the input string.")

    # Initialize the last known period
    last_period = None

    # Helper function to convert time to minutes since midnight
    def convert_to_minutes(hours, minutes, period):
        nonlocal last_period
        hours = int(hours)
        minutes = int(minutes)

        # Use the last known period if the current period is missing
        if not period:
            period = last_period
        else:
            last_period = period  # Update the last known period

        if period == 'PM' and hours != 12:
            hours += 12
        if period == 'AM' and hours == 12:
            hours = 0
        return hours * 60 + minutes

    # Convert the extracted times to minutes
    times = [convert_to_minutes(h, m, p) for h, m, p in time_ranges]

    # Group the times into pairs (ranges)
    ranges = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

    # Return one range randomly
    return random.choice(ranges)

# Example usage
if __name__ == "__main__":
    time_str = "Saturday: 11:30 AM – 2:30 PM, 5:30 – 8:30 PM"
    try:
        random_range = time_to_minutes(time_str)
        print("Randomly selected time range (in minutes since midnight):", random_range)
    except ValueError as e:
        print("Error:", e)
