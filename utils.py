from typing import List
import csv
from collections import OrderedDict

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
        self.on_chip_memory = 40 * 1024 * 1024  # 20MB total on-chip (e.g., L2 + SRAM)
        self.on_chip_used = 0
        self.cache_lru: "OrderedDict[int, int]" = OrderedDict()  # address -> size (most recent last)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # ← create the one and only instance
        return cls._instance

    def cached(self, address, size, tile_size = 4096) -> bool:
        """
        Track this tile in the LRU cache. Return True if fully cached, False if DRAM needed.
        """
        # tile_size = min(max_tile_size, max(256, tensor.element_size() * d_head))
        is_fully_on_chip = True
        aligned_addresses = range(address, address + size, tile_size)

        for aligned_addr in aligned_addresses:
            if aligned_addr in self.cache_lru:
                self.cache_lru.move_to_end(aligned_addr)
                continue

            # Evict until space is available
            while self.on_chip_used + tile_size > self.on_chip_memory:
                evicted_addr, evicted_size = self.cache_lru.popitem(last=False)
                self.on_chip_used -= evicted_size

            self.cache_lru[aligned_addr] = tile_size
            self.on_chip_used += tile_size
            is_fully_on_chip = False  # at least one tile was not already cached

        return is_fully_on_chip

    def emit_trace(self, access_type, base_addr, offset, size, timestamp_ns, tensor_name, tile_coords, function):
        address = base_addr + offset

        if not self.cached(address, size):
            self.memory_trace.append({
                "timestamp_ns": int(timestamp_ns),
                "address": hex(address &  0xFFFFFFFF),
                "bytes": size,
                "type": access_type,  # "read" or "write"
                "tensor": tensor_name,
                "tile": tile_coords,
                "function":function
            })
        # else:
        #     self.memory_trace.append({
        #         "timestamp_ns": "cached",
        #         "address": hex(address &  0xFFFFFFFF),
        #         "bytes": size,
        #         "type": access_type,  # "read" or "write"
        #         "tensor": tensor_name,
        #         "tile": tile_coords,
        #         "function":function
        #     })

        # print(
        #     "{"
        #     f'"timestamp_ns": {int(timestamp_ns)}, '
        #     f'"address": "{hex((base_addr + offset) &  0xFFFFFFFF)}", '
        #     f'"bytes": {size}, '
        #     f'"type": "{access_type}", '
        #     f'"tensor": "{tensor_name}", '
        #     f'"tile": {tile_coords}'
        #     "}"
        # )

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

    @staticmethod
    def coalesce_memory_traces(traces, max_gap_bytes = 0):
        """
        Merge adjacent memory traces that:
        - Have same timestamp, type, tensor, and function
        - Are contiguous (or within `max_gap_bytes`) in address space
        """
        if not traces:
            return []

        # traces.sort(key=lambda x: (x["timestamp_ns"], x["tensor"], x["type"], int(x["address"], 16)))

        merged = []
        group = [traces[0]]

        def flush_group(group):
            if not group:
                return
            first = group[0]
            merged.append({
                "timestamp_ns": first["timestamp_ns"],
                "address": first["address"],
                "bytes": sum(t["bytes"] for t in group),
                "type": first["type"],
                "tensor": first["tensor"],
                "tile": (first["tile"][0], "coalesced"),
                # "on_chip": all(t.get("on_chip", False) for t in group),
                "function": first.get("function", "")
            })

        for curr in traces[1:]:
            prev = group[-1]
            prev_addr = int(prev["address"], 16)
            curr_addr = int(curr["address"], 16)
            prev_end = prev_addr + prev["bytes"]

            is_adjacent = (
                # curr["timestamp_ns"] == prev["timestamp_ns"] and
                curr["type"] == prev["type"] and
                curr["tensor"] == prev["tensor"] and
                curr.get("function", "") == prev.get("function", "") and
                curr_addr <= prev_end + max_gap_bytes
            )

            if is_adjacent:
                group.append(curr)
            else:
                flush_group(group)
                group = [curr]

        flush_group(group)
        return merged

    @staticmethod
    def make_timestamps_monotonic(traces):
        if not traces:
            return traces

        current_time = traces[0]["timestamp_ns"]

        for i in range(1, len(traces)):
            raw = traces[i]["timestamp_ns"]
            if raw < current_time:
                current_time += raw
                traces[i]["timestamp_ns"] = current_time
            else:
                current_time = raw
                traces[i]["timestamp_ns"] = current_time

        return traces

    
    def coalesce(self, filepath="memory_trace.csv"):
        # === 1. Load trace from original CSV ===
        trace = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace.append({
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "address": row["address"],
                    "bytes": int(row["bytes"]),
                    "type": row["type"],
                    "tensor": row["tensor"],
                    "tile": eval(row["tile"]),  # convert stringified tuple back to tuple
                    "function": row.get("function", "")
                    # "on_chip": row.get("on_chip", "False") == "True"
                })

        # === 2. Coalesce the trace ===
        time_trace = self.make_timestamps_monotonic(trace)
        coalesced_trace = self.coalesce_memory_traces(time_trace)
        # coalesced_trace = self.coalesce_memory_traces(trace)

        # === 3. Save coalesced version ===
        with open("coalesced_trace.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_ns", "address", "bytes", "type", "tensor", "tile", "function"])
            writer.writeheader()
            for entry in coalesced_trace:
                entry_copy = entry.copy()
                entry_copy["tile"] = str(entry_copy["tile"])
                writer.writerow(entry_copy)
