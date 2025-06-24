from typing import List
import csv

def size_of_list(list: List):
    result = 1
    for i in list:
        result *= i
    return result

def size(list):
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return list.size

def closest_factors(n):
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0



class TraceLogger:
    _instance = None  # ← class-level variable, holds the single instance

    def __init__(self):
        self.memory_trace = []

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # ← create the one and only instance
        return cls._instance

    def emit_trace(self, access_type, base_addr, offset, size, timestamp_ns, tensor_name, tile_coords, function):
        self.memory_trace.append({
            "timestamp_ns": int(timestamp_ns),
            "address": hex((base_addr + offset) &  0xFFFFFFFF),
            "bytes": size,
            "type": access_type,  # "read" or "write"
            "tensor": tensor_name,
            "tile": tile_coords,
            "function":function
        })

        print(
            "{"
            f'"timestamp_ns": {int(timestamp_ns)}, '
            f'"address": "{hex((base_addr + offset) &  0xFFFFFFFF)}", '
            f'"bytes": {size}, '
            f'"type": "{access_type}", '
            f'"tensor": "{tensor_name}", '
            f'"tile": {tile_coords}'
            "}"
        )

    def save_trace(self, filepath="memory_trace.csv"):
        fieldnames = ["timestamp_ns", "address", "bytes", "type", "tensor", "tile", "function"]

        with open(filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.memory_trace:
                # Convert tile tuple to string for CSV
                entry_copy = entry.copy()
                entry_copy["tile"] = str(entry_copy["tile"])
                writer.writerow(entry_copy)

        print(f"Memory trace saved to {filepath}")
