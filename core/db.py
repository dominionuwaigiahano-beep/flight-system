"""
db.py
SQLite data layer for the prototype. Kept deliberately simple and parameterised.
The schema (flights, gates, zones, passengers, notifications, users) mirrors the
ERD in Chapter Three. Written so it could be repointed at PostgreSQL later.
"""
import sqlite3
import os
import hashlib

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "..", "data", "flight_system.db")

CARRIERS = ["Air Peace", "Arik Air", "Ibom Air", "Dana Air", "United Nigeria", "Overland"]
ZONES = ["Check-in Hall", "Security", "Departure Lounge", "Food Court", "Gate Area", "Restroom"]
ZONE_DISTANCE_MIN = {"Check-in Hall": 12, "Security": 9, "Departure Lounge": 5,
                     "Food Court": 7, "Gate Area": 1, "Restroom": 4}

def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    c = conn()
    c.executescript("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT);
    CREATE TABLE flights (
        id INTEGER PRIMARY KEY, flight_no TEXT, carrier TEXT, destination TEXT,
        dep_hour INTEGER, gate TEXT, status TEXT, delay_mins INTEGER DEFAULT 0);
    CREATE TABLE passengers (
        id INTEGER PRIMARY KEY, name TEXT, flight_id INTEGER, lang TEXT,
        zone TEXT, checked_in_at_gate INTEGER, prior_late_count INTEGER,
        is_frequent_flyer INTEGER,
        FOREIGN KEY(flight_id) REFERENCES flights(id));
    CREATE TABLE notifications (
        id INTEGER PRIMARY KEY, passenger_id INTEGER, flight_id INTEGER,
        event_type TEXT, lang TEXT, message TEXT, zone TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')));
    """)

    # users
    for u, p, r in [("admin", "admin", "administrator"),
                    ("staff", "staff", "staff")]:
        c.execute("INSERT INTO users(username,password,role) VALUES(?,?,?)",
                  (u, hash_pw(p), r))

    # flights (simulated Benin Airport departures)
    flights = [
        ("P47123", "Air Peace", "Lagos", 9, "1", "Scheduled"),
        ("W30455", "Arik Air", "Abuja", 11, "2", "Scheduled"),
        ("QI０12", "Ibom Air", "Uyo", 14, "3", "Scheduled"),
        ("9J208", "Dana Air", "Abuja", 18, "2", "Scheduled"),
        ("U5601", "United Nigeria", "Lagos", 20, "1", "Scheduled"),
    ]
    # fix accidental fullwidth chars if any
    flights = [(fn.replace("０","0").replace("１","1").replace("２","2"), ca, de, dh, g, s)
               for (fn, ca, de, dh, g, s) in flights]
    for fn, ca, de, dh, g, s in flights:
        c.execute("""INSERT INTO flights(flight_no,carrier,destination,dep_hour,gate,status)
                     VALUES(?,?,?,?,?,?)""", (fn, ca, de, dh, g, s))

    # passengers spread across flights, zones, languages
    import random
    random.seed(7)
    names = ["Osaro I.","Amaka N.","Bello M.","Efe O.","Grace A.","Tunde K.",
             "Ngozi U.","Sani D.","Peace E.","Ola B.","Ifeoma C.","Musa G.",
             "Blessing O.","Chidi A.","Hauwa S.","Eseosa U."]
    langs = ["en","pcm","hau","edo"]
    fids = [r["id"] for r in c.execute("SELECT id FROM flights")]
    for i, nm in enumerate(names):
        fid = random.choice(fids)
        zone = random.choice(ZONES)
        c.execute("""INSERT INTO passengers(name,flight_id,lang,zone,checked_in_at_gate,
                     prior_late_count,is_frequent_flyer) VALUES(?,?,?,?,?,?,?)""",
                  (nm, fid, langs[i % 4], zone,
                   random.choices([0,1],weights=[55,45])[0],
                   random.choices([0,1,2,3],weights=[60,22,12,6])[0],
                   random.choices([0,1],weights=[70,30])[0]))
    c.commit()
    c.close()
    print("Database initialised at", os.path.abspath(DB_PATH))

if __name__ == "__main__":
    init_db()
