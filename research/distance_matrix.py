from datetime import datetime
import os
import googlemaps
import math

#from urllib.parse import urlparse, parse_qsl
from dotenv import load_dotenv

load_dotenv()


class DistanceMatrix():  # Extend unittest.TestCase
    def __init__(self, *args, **kwargs):
        super(DistanceMatrix, self).__init__(*args, **kwargs)  # Initialize superclass
        self.key = os.getenv('GOOGLE_MAP_API_KEY')  # Load API key from .env
        self.client = googlemaps.Client(key=self.key)

    def testAllParams(self, ls):
        # Real request to the Google Distance Matrix API

        origins = ls
        destinations = ls

        now = datetime.now()

        # manual site: https://developers.google.com/maps/documentation/distance-matrix/distance-matrix?hl=zh-tw
        matrix = self.client.distance_matrix(
            origins,
            destinations,
            mode="driving",
            language="zh-TW",
            avoid="tolls",
            units="imperial",
            departure_time=now,
            traffic_model="pessimistic",
        )

        # Print the real API response
        #print(matrix, type(matrix))
        return matrix

    def matrixTo2dArray(self, sites: list, time_windows):
        matrix = self.testAllParams(sites)
        if 'destination_addresses' not in matrix or not isinstance(matrix['destination_addresses'], list):
            raise ValueError("Invalid matrix structure: 'destination_addresses' is missing or not a list.")

        # Identify indices to neglect
        neglect = [i for i, address in enumerate(matrix['destination_addresses']) if address == '']

        # Validate time_windows length
        if len(time_windows) != len(matrix['destination_addresses']):
            raise ValueError("Mismatch between time_windows and destination_addresses length.")

        # Filter time windows
        time_windows2 = [time_windows[i] for i in range(len(time_windows)) if i not in neglect]

        arr = []
        for i in range(len(matrix['rows'])):
            if i in neglect:
                continue
            tmp = []
            src = matrix['rows'][i]

            for j in range(len(src['elements'])):
                if j in neglect:
                    continue
                dist = src['elements'][j]
                # print(dist)
                if 'duration_in_traffic' in dist:
                    raw_minutes:int = math.ceil(dist['duration_in_traffic']['value'] / 60)
                    adjusted = raw_minutes + (5 - (raw_minutes % 5)) if raw_minutes % 10 != 0 else raw_minutes
                    tmp.append({
                        "raw": raw_minutes,
                        "adjusted": adjusted
                    })                
                else:
                    tmp.append({"raw": 1441, "adjusted": 1441})
                    print("No duration_in_traffic in the element, something go wrong!")
            arr.append(tmp)


        return [sites[i] for i in range(len(matrix['destination_addresses'])) if matrix['destination_addresses'][i] != ''], arr, time_windows2


def travel_time(sites: list, time_windows):
    # print(sites)
    distance_matrix = DistanceMatrix()
    exist_sites, arr, time_windows = distance_matrix.matrixTo2dArray(sites, time_windows)
    # print('successfully get the 2d array!')
    # for ele in arr:
    #     for i in ele:
    #         print(i, end=' ')
    #     print()
    return exist_sites, arr, time_windows

if __name__ == "__main__":
    distance_matrix = DistanceMatrix()
    sites: list = [
        "嘉義公園",
        "檜意森活村",
        "蒜頭糖廠",
        ]
    arr = distance_matrix.matrixTo2dArray(sites)
    for ele in arr:
        for i in ele:
            print(i, end=' ')
        print()

