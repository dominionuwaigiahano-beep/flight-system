"""
generate_data.py
Generates a realistic simulated dataset modelled on Benin Airport (BNI) operations,
used to train the two machine learning models in the system:
  1. Missed-boarding risk classifier  (primary AI core, objective 3)
  2. Flight-delay prediction classifier (supporting AI, objective 4)

No real passenger or airline data is used. Patterns (boarding behaviour, delay
drivers) are modelled on documented airport operations so the models learn
meaningful relationships rather than noise.
"""
import csv
import os
import random

random.seed(42)  # reproducible dataset for the thesis

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

CARRIERS = ["Air Peace", "Arik Air", "Ibom Air", "Dana Air", "United Nigeria", "Overland"]
ZONES = ["Check-in Hall", "Security", "Departure Lounge", "Food Court", "Gate Area", "Restroom"]
# Walking minutes from each zone to a boarding gate (simulated terminal geography)
ZONE_DISTANCE_MIN = {
    "Check-in Hall": 12, "Security": 9, "Departure Lounge": 5,
    "Food Court": 7, "Gate Area": 1, "Restroom": 4,
}

# ---------------------------------------------------------------------------
# 1. Missed-boarding risk dataset
# ---------------------------------------------------------------------------
# Features per passenger, captured at the moment boarding opens:
#   minutes_to_departure : time left until scheduled departure
#   checked_in_at_gate    : 1 if passenger has scanned at the gate, else 0
#   zone_distance_min     : walking minutes from current zone to the gate
#   prior_late_count      : how many times this traveller profile was late before
#   is_frequent_flyer     : 1 if frequent flyer, else 0
# Label: missed_boarding (1 = missed / very late, 0 = boarded on time)

def gen_missed_boarding(n=4000):
    rows = []
    for _ in range(n):
        minutes_to_departure = random.randint(5, 60)
        checked_in_at_gate = random.choices([0, 1], weights=[55, 45])[0]
        zone = random.choice(ZONES)
        zone_distance_min = ZONE_DISTANCE_MIN[zone] + random.randint(-1, 3)
        zone_distance_min = max(1, zone_distance_min)
        prior_late_count = random.choices([0, 1, 2, 3], weights=[60, 22, 12, 6])[0]
        is_frequent_flyer = random.choices([0, 1], weights=[70, 30])[0]

        # Risk logic (hidden ground truth the model must learn):
        # a passenger misses if the time they still need exceeds time available,
        # softened by check-in status and sharpened by poor punctuality history.
        time_needed = zone_distance_min + 6  # walk + queue/boarding buffer
        slack = minutes_to_departure - time_needed
        risk = 0.0
        if slack < 0:
            risk += 0.55
        elif slack < 5:
            risk += 0.30
        elif slack < 10:
            risk += 0.12
        if not checked_in_at_gate:
            risk += 0.22
        risk += 0.08 * prior_late_count
        if is_frequent_flyer:
            risk -= 0.10
        risk += random.uniform(-0.08, 0.08)  # irreducible noise

        missed = 1 if risk >= 0.5 else 0
        rows.append([minutes_to_departure, checked_in_at_gate, zone_distance_min,
                     prior_late_count, is_frequent_flyer, missed])

    path = os.path.join(DATA_DIR, "missed_boarding.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minutes_to_departure", "checked_in_at_gate", "zone_distance_min",
                    "prior_late_count", "is_frequent_flyer", "missed_boarding"])
        w.writerows(rows)
    pos = sum(r[-1] for r in rows)
    print(f"missed_boarding.csv : {len(rows)} rows, {pos} missed ({100*pos/len(rows):.1f}%)")

# ---------------------------------------------------------------------------
# 2. Flight-delay dataset
# ---------------------------------------------------------------------------
# Features per scheduled flight:
#   dep_hour          : scheduled departure hour (0-23)
#   carrier_idx       : encoded carrier
#   weather_bad       : 1 if adverse weather, else 0
#   inbound_late      : 1 if the inbound aircraft arrived late, else 0
#   is_weekend        : 1 if weekend, else 0
# Label: delayed (1 = delayed 30+ min, 0 = on time)

def gen_delay(n=3000):
    rows = []
    for _ in range(n):
        dep_hour = random.randint(5, 22)
        carrier_idx = random.randint(0, len(CARRIERS) - 1)
        weather_bad = random.choices([0, 1], weights=[78, 22])[0]
        inbound_late = random.choices([0, 1], weights=[70, 30])[0]
        is_weekend = random.choices([0, 1], weights=[71, 29])[0]

        p = 0.10
        if weather_bad:
            p += 0.33
        if inbound_late:
            p += 0.30
        if dep_hour >= 18:          # evening congestion/knock-on effect
            p += 0.14
        if is_weekend:
            p += 0.05
        p += random.uniform(-0.05, 0.05)
        delayed = 1 if random.random() < p else 0
        rows.append([dep_hour, carrier_idx, weather_bad, inbound_late, is_weekend, delayed])

    path = os.path.join(DATA_DIR, "flight_delay.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dep_hour", "carrier_idx", "weather_bad", "inbound_late",
                    "is_weekend", "delayed"])
        w.writerows(rows)
    pos = sum(r[-1] for r in rows)
    print(f"flight_delay.csv    : {len(rows)} rows, {pos} delayed ({100*pos/len(rows):.1f}%)")

if __name__ == "__main__":
    gen_missed_boarding()
    gen_delay()
    print("Datasets written to", os.path.abspath(DATA_DIR))
