"""
app.py  -  AI-Driven Flight Announcement and Passenger Notification System
Prototype, Benin Airport context. Flask + SQLite + scikit-learn.

Run:
    python app.py
then open http://127.0.0.1:5000
Logins: admin/admin (administrator), staff/staff (staff)
"""
import os
import math
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from core.db import conn, hash_pw, init_db, DB_PATH, ZONE_DISTANCE_MIN, CARRIERS, ZONES
from core.ml.engine import (predict_missed_boarding, predict_delay,
                           generate_announcement, LANG_NAMES, load_models)

app = Flask(__name__,
            template_folder="core/templates",
            static_folder="core/static")
app.secret_key = "flight-prototype-benin-2026"

CARRIER_IDX = {c: i for i, c in enumerate(CARRIERS)}

def ensure_ready():
    # On a fresh server the dataset and trained models may not exist yet.
    # Build them once if missing, so the app can start unattended.
    models_dir = os.path.join(os.path.dirname(__file__), "core", "models")
    mb = os.path.join(models_dir, "missed_boarding_model.pkl")
    dl = os.path.join(models_dir, "flight_delay_model.pkl")
    if not (os.path.exists(mb) and os.path.exists(dl)):
        from core.ml import generate_data, train
        generate_data.gen_missed_boarding()
        generate_data.gen_delay()
        train.train_one("missed_boarding.csv", "missed_boarding",
                        "missed_boarding_model.pkl", "Missed-Boarding Risk Model")
        train.train_one("flight_delay.csv", "delayed",
                        "flight_delay_model.pkl", "Flight-Delay Prediction Model")
    if not os.path.exists(DB_PATH):
        init_db()
    load_models()

# ---------- helpers ----------
def current_minutes_to_departure(dep_hour, delay_mins=0):
    """Simulated 'now' uses the earliest flight hour as reference so the demo
    always has live, meaningful countdowns regardless of real clock time."""
    now = datetime.now()
    # Map real minutes into the simulated schedule: treat 'now' minute-of-hour
    # as the live tick within the boarding window.
    sim_minutes = (dep_hour * 60) - (now.hour * 60 + now.minute) + delay_mins
    # keep it inside a sensible boarding window for the demo
    if sim_minutes > 60 or sim_minutes < 0:
        sim_minutes = 5 + (now.minute % 55)
    return int(sim_minutes)

def passenger_risk(p, flight):
    mins = current_minutes_to_departure(flight["dep_hour"], flight["delay_mins"])
    feats = {
        "minutes_to_departure": mins,
        "checked_in_at_gate": p["checked_in_at_gate"],
        "zone_distance_min": ZONE_DISTANCE_MIN.get(p["zone"], 6),
        "prior_late_count": p["prior_late_count"],
        "is_frequent_flyer": p["is_frequent_flyer"],
    }
    r = predict_missed_boarding(feats)
    r["minutes_to_departure"] = mins
    return r

# ---------- auth ----------
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("passenger"))

@app.route("/login", methods=["GET", "POST"])
def login():
    ensure_ready()
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        c = conn()
        row = c.execute("SELECT * FROM users WHERE username=? AND password=?",
                        (u, hash_pw(p))).fetchone()
        c.close()
        if row:
            session["user"] = row["username"]
            session["role"] = row["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------- admin / staff console ----------
@app.route("/dashboard")
def dashboard():
    ensure_ready()
    if "user" not in session:
        return redirect(url_for("login"))
    c = conn()
    flights = c.execute("SELECT * FROM flights ORDER BY dep_hour").fetchall()
    data = []
    for f in flights:
        d = predict_delay({
            "dep_hour": f["dep_hour"], "carrier_idx": CARRIER_IDX.get(f["carrier"], 0),
            "weather_bad": 1 if f["status"] == "Delayed" else 0,
            "inbound_late": 1 if f["delay_mins"] > 0 else 0,
            "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
        })
        pax = c.execute("SELECT * FROM passengers WHERE flight_id=?", (f["id"],)).fetchall()
        risks = []
        for p in pax:
            r = passenger_risk(p, f)
            risks.append({"id": p["id"], "name": p["name"], "lang": LANG_NAMES[p["lang"]],
                          "zone": p["zone"], "checked_in": p["checked_in_at_gate"],
                          "risk": r["risk"], "band": r["band"],
                          "mins": r["minutes_to_departure"]})
        risks.sort(key=lambda x: x["risk"], reverse=True)
        high = sum(1 for x in risks if x["band"] == "HIGH")
        data.append({"flight": dict(f), "delay": d, "pax": risks, "high": high})
    c.close()
    return render_template("dashboard.html", data=data, role=session.get("role"),
                           user=session.get("user"))

@app.route("/fire_event", methods=["POST"])
def fire_event():
    """Staff triggers a flight event. System generates targeted, multilingual
    notifications only for the affected flight's passengers (zone-aware)."""
    if "user" not in session:
        return jsonify({"error": "unauthorised"}), 401
    flight_id = int(request.form["flight_id"])
    event_type = request.form["event_type"]
    newgate = request.form.get("newgate", "")
    c = conn()
    f = c.execute("SELECT * FROM flights WHERE id=?", (flight_id,)).fetchone()
    if event_type == "gate_change" and newgate:
        c.execute("UPDATE flights SET gate=? WHERE id=?", (newgate, flight_id))
    if event_type == "delay":
        c.execute("UPDATE flights SET status='Delayed', delay_mins=45 WHERE id=?", (flight_id,))
    if event_type == "boarding_open":
        c.execute("UPDATE flights SET status='Boarding' WHERE id=?", (flight_id,))
    if event_type == "final_call":
        c.execute("UPDATE flights SET status='Final Call' WHERE id=?", (flight_id,))
    f = c.execute("SELECT * FROM flights WHERE id=?", (flight_id,)).fetchone()

    pax = c.execute("SELECT * FROM passengers WHERE flight_id=?", (flight_id,)).fetchall()
    count = 0
    for p in pax:
        mins = current_minutes_to_departure(f["dep_hour"], f["delay_mins"])
        fields = {"flight": f["flight_no"], "gate": f["gate"],
                  "dest": f["destination"], "newgate": newgate, "mins": str(abs(mins))}
        msg = generate_announcement(event_type, lang=p["lang"], **fields)
        c.execute("""INSERT INTO notifications(passenger_id,flight_id,event_type,lang,message,zone)
                     VALUES(?,?,?,?,?,?)""",
                  (p["id"], flight_id, event_type, p["lang"], msg, p["zone"]))
        count += 1
    c.commit(); c.close()
    return jsonify({"ok": True, "notified": count,
                    "flight": f["flight_no"], "event": event_type})

@app.route("/send_risk_alerts", methods=["POST"])
def send_risk_alerts():
    """Pre-emptive: the ML model finds HIGH-risk passengers and alerts them early."""
    if "user" not in session:
        return jsonify({"error": "unauthorised"}), 401
    c = conn()
    flights = {f["id"]: f for f in c.execute("SELECT * FROM flights").fetchall()}
    pax = c.execute("SELECT * FROM passengers").fetchall()
    sent = 0
    for p in pax:
        f = flights[p["flight_id"]]
        r = passenger_risk(p, f)
        if r["band"] == "HIGH":
            fields = {"flight": f["flight_no"], "gate": f["gate"],
                      "dest": f["destination"], "mins": str(abs(r["minutes_to_departure"]))}
            msg = generate_announcement("risk_alert", lang=p["lang"], **fields)
            c.execute("""INSERT INTO notifications(passenger_id,flight_id,event_type,lang,message,zone)
                         VALUES(?,?,?,?,?,?)""",
                      (p["id"], f["id"], "risk_alert", p["lang"], msg, p["zone"]))
            sent += 1
    c.commit(); c.close()
    return jsonify({"ok": True, "alerts": sent})

# ---------- passenger view ----------
@app.route("/passenger")
def passenger():
    ensure_ready()
    c = conn()
    pax = c.execute("SELECT * FROM passengers ORDER BY name").fetchall()
    c.close()
    return render_template("passenger.html", pax=pax, langs=LANG_NAMES)

@app.route("/passenger/<int:pid>")
def passenger_feed(pid):
    c = conn()
    p = c.execute("SELECT * FROM passengers WHERE id=?", (pid,)).fetchone()
    f = c.execute("SELECT * FROM flights WHERE id=?", (p["flight_id"],)).fetchone()
    notes = c.execute("""SELECT * FROM notifications WHERE passenger_id=?
                         ORDER BY id DESC""", (pid,)).fetchall()
    r = passenger_risk(p, f)
    c.close()
    return render_template("passenger_feed.html", p=p, f=f,
                           notes=notes, risk=r, langs=LANG_NAMES)

@app.route("/api/feed/<int:pid>")
def api_feed(pid):
    """Polled by passenger view to show live targeted notifications."""
    c = conn()
    notes = c.execute("""SELECT event_type,lang,message,zone,created_at
                         FROM notifications WHERE passenger_id=?
                         ORDER BY id DESC LIMIT 20""", (pid,)).fetchall()
    c.close()
    return jsonify([dict(n) for n in notes])

@app.route("/api/update_passenger", methods=["POST"])
def update_passenger():
    """Passenger changes language or zone (simulated movement)."""
    pid = int(request.form["pid"])
    lang = request.form.get("lang")
    zone = request.form.get("zone")
    c = conn()
    if lang:
        c.execute("UPDATE passengers SET lang=? WHERE id=?", (lang, pid))
    if zone:
        c.execute("UPDATE passengers SET zone=? WHERE id=?", (zone, pid))
    c.commit(); c.close()
    return jsonify({"ok": True})

# Ensure models and database exist when the app is imported by a production
# server such as gunicorn (which does not run the __main__ block below).
ensure_ready()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # host 0.0.0.0 so the app is reachable when hosted; debug off in production
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
