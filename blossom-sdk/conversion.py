# Modified from pypot's conversion.py file
# Source: https://github.com/poppy-project/pypot/blob/master/pypot/dynamixel/conversion.py

# Motor type : (Max pos, max degree)
position_range = {
    350: (1024, 300.0),
    1200: (4096, 360.0) 
}

def dxl_to_degree(value, model):
    max_pos, max_deg = position_range[model]

    return round(((max_deg * float(value)) / (max_pos - 1)) - (max_deg / 2), 1)

def degree_to_dxl(value, model): 
    max_pos, max_deg = position_range[model]

    pos = int(round((max_pos - 1) * ((max_deg / 2 + float(value)) / max_deg), 0))
    pos = min(max(pos, 0), max_pos - 1)

    return pos