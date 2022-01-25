import math

def calculate_initial_compass_bearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def name_angle(angle):
    if angle <= 22.5 or angle > 337.5:
        return "NORTH"
    elif angle > 22.5 and angle <= 67.5:
        return "NORTHEAST"
    elif angle > 67.5 and angle <= 112.5:
        return "EAST"
    elif angle > 112.5 and angle <= 157.5:
        return "SOUTHEAST"
    elif angle > 157.5 and angle <= 202.5:
        return "SOUTH"
    elif angle > 202.5 and angle <= 247.5:
        return "SOUTHWEST"
    elif angle > 247.5 and angle <= 292.5:
        return "WEST"
    elif angle > 292.5 and angle <= 337.5:
        return "NORTHWEST"

def id_angle(angle):
    if angle <= 22.5 or angle > 337.5:
        return 0
    elif angle > 22.5 and angle <= 67.5:
        return 1
    elif angle > 67.5 and angle <= 112.5:
        return 2
    elif angle > 112.5 and angle <= 157.5:
        return 3
    elif angle > 157.5 and angle <= 202.5:
        return 4
    elif angle > 202.5 and angle <= 247.5:
        return 5
    elif angle > 247.5 and angle <= 292.5:
        return 6
    elif angle > 292.5 and angle <= 337.5:
        return 7