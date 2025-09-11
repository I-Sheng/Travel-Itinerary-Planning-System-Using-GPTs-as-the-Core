#!/usr/bin/env python3
# Copyright 2010-2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START program]
"""Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).

   This is a sample using the routing library python wrapper to solve a CVRPTW
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters and time in minutes.
"""

# [START import]
import functools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
# [END import]

def create_time_evaluator(data):
    """Creates callback to get total times between locations."""

    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data["service"][node] * data["service_unit"]

    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = data["time_matrix"][from_node][to_node]['adjusted']
        return travel_time

    total_time_ = {}
    # precompute total time to have time callback in O(1)
    for from_node in range(data["numlocations_"]):
        total_time_[from_node] = {}
        for to_node in range(data["numlocations_"]):
            if from_node == to_node:
                total_time_[from_node][to_node] = 0
            else:
                total_time_[from_node][to_node] = int(
                        service_time(data, from_node)
                        + travel_time(data, from_node, to_node)
                        )

    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes."""
        return total_time_[manager.IndexToNode(from_node)][manager.IndexToNode(to_node)]

    return time_evaluator


def add_time_window_constraints(routing, manager, data, time_evaluator_index):
    """Add Global Span constraint."""
    start_time = data['start_time']
    end_time = data['end_time']


    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data["service"][node] * data["service_unit"]


    time = "Time"
    horizon = end_time - start_time # set waiting time for time window & total routing time
    routing.AddDimension(
            time_evaluator_index,
            horizon,  # allow waiting time
            horizon,  # maximum time per vehicle
            False,  # don't force start cumul to zero
            time,
            )
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        time1 = int(time_window[0])
        time2 = int(time_window[1])
        if time1 == 0 and time2 == 0:
            time_dimension.CumulVar(index).SetRange(open_time, close_time)
            continue
        index = manager.NodeToIndex(location_idx)
        open_time = max(time1 + service_time(data, location_idx) - start_time, 0)
        close_time = min(time2 - start_time, horizon)
        if open_time >= close_time:
            # print(f'location: {location_idx}')
            # print(f'open_time: {open_time}')
            # print(f'close_time: {close_time}')
            # assert("something wrong")
            close_time = 1440

        open_time = int(open_time)
        close_time = int(close_time)
        # print("open", open_time, "close",  close_time)
        time_dimension.CumulVar(index).SetRange(open_time, close_time)
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
                min(0, data['time_windows'][0][0] - start_time), min(horizon, max(data['time_windows'][0][1] - start_time, 0))
                # data["time_windows"][0][0], data["time_windows"][0][1]
                )
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # The time window at the end node was impliclty set in the time dimension
        # definition to be [0, horizon].
        # Warning: Slack var is not defined for vehicle end nodes and should not
        # be added to the assignment.


# [START solution_printer]

def print_solution(data, manager, routing, assignment):  # pylint:disable=too-many-locals
    """Prints assignment on console with arrival and departure times."""
    # print(f"Objective: {assignment.ObjectiveValue()}")
    arr:list = []
    idx = -1
    total_time = 0
    time_dimension = routing.GetDimensionOrDie("Time")
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_time = 0

        while not routing.IsEnd(index):
            idx += 1
            arr.append({})
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            arrival_time = assignment.Min(time_var)
            departure_time = arrival_time + data["service"][node]
            slack_var = time_dimension.SlackVar(index)
            arr[idx]= {'name': data['name'][node], 'arrival': arrival_time, 'service': data['service'][node], 'vehicle': vehicle_id, 'end_node': False}
            plan_output += (
                f" Node {data['name'][node]} - Arrival: {arrival_time}, \n"
                #f" Time({assignment.Min(time_var)}, {assignment.Max(time_var)})\n"
                #f" Slack({assignment.Min(slack_var)}, {assignment.Max(slack_var)})\n"
                f" Node Service time: {data['service'][node]}\n"
                f"Departure: {departure_time}  \n\n"
            )

            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            # if not routing.IsEnd(index):
            travel_time = data["time_matrix"][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]['adjusted']
            plan_output += f"Travel Time: {travel_time} -> "
            arr[idx]['travel'] = travel_time
            arr[idx]['travel_raw'] = data["time_matrix"][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]['raw']

        # Handle the last node
        idx += 1
        arr.append({})
        time_var = time_dimension.CumulVar(index)
        node = manager.IndexToNode(index)
        arrival_time = assignment.Min(time_var)
        plan_output += f" Node {data['name'][node]} - Arrival: {arrival_time}, \n"
        arr[idx] = {'name': data['name'][node], 'arrival': arrival_time + 480, 'end_node': True, 'vehicle': vehicle_id} # +480 for 8 a.m.

        route_time += assignment.Value(time_var)
        plan_output += f"Total Time of the route: {route_time} minutes\n"
        # print(plan_output)

        total_time += route_time


    # print(f"Total Time of all routes: {total_time} minutes")
    return arr
# [END solution_printer]



def routing(data:dict):
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]
    # data = create_data_model()
    # [END data]

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
            data["numlocations_"], data["num_vehicles"], data["depot"]
            )

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Add Time Window constraint
    time_evaluator_index = routing.RegisterTransitCallback(
            functools.partial(create_time_evaluator(data), manager)
            )
    add_time_window_constraints(routing, manager, data, time_evaluator_index)


    # Setting first solution heuristic (cheapest addition).
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED
            )  # pylint: disable=no-member
    # [END parameters]

    # Solve the problem.
    # [START solve]
    assignment = routing.SolveWithParameters(search_parameters)
    # [END solve]

    # Print solution on console.
    # [START print_solution]
    if assignment:
        return print_solution(data, manager, routing, assignment)
    else:
        print("No solution found!")
        return None
    # [END print_solution]


if __name__ == "__main__":
    data:dict = create_data_model()
    routing(data)
# [END program]
