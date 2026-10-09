"""
app.py - AI-Driven Flight Announcement and Passenger Notification System
Prototype, Benin Airport context. Flask + SQLite + scikit-learn.

Run:
    python app.py

Then open:
    http://127.0.0.1:5000

Logins:
    admin/admin
    staff/staff
"""

import os
import math
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash
)

import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.db import (
    conn,
    hash_pw,
    init_db,
    DB_PATH,
    ZONE_DISTANCE_MIN,
    CARRIERS,
    ZONES
)

from core.ml.engine import (
    predict_missed_boarding,
    predict_delay,
    generate_announcement,
    LANG_NAMES,
    load_models
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="core/templates",
    static_folder="core/static"
)

app.secret_key = "flight-prototype-benin-2026"


# Carrier index used by the delay prediction model
CARRIER_IDX = {
    c: i for i, c in enumerate(CARRIERS)
}


# ============================================================
# STARTUP / DATABASE / MODEL SETUP
# ============================================================

def ensure_ready():
    """
    Make sure the database and trained ML models exist.

    If the models are missing, generate the training data,
    train the models, and then load them.
    """

    models_dir = os.path.join(
        os.path.dirname(__file__),
        "core",
        "models"
    )

    mb = os.path.join(
        models_dir,
        "missed_boarding_model.pkl"
    )

    dl = os.path.join(
        models_dir,
        "flight_delay_model.pkl"
    )

    # --------------------------------------------------------
    # Generate and train models if they do not exist
    # --------------------------------------------------------

    if not (os.path.exists(mb) and os.path.exists(dl)):

        from core.ml import generate_data, train

        generate_data.gen_missed_boarding()
        generate_data.gen_delay()

        train.train_one(
            "missed_boarding.csv",
            "missed_boarding",
            "missed_boarding_model.pkl",
            "Missed-Boarding Risk Model"
        )

        train.train_one(
            "flight_delay.csv",
            "delayed",
            "flight_delay_model.pkl",
            "Flight-Delay Prediction Model"
        )

    # --------------------------------------------------------
    # Initialize database if it does not exist
    # --------------------------------------------------------

    if not os.path.exists(DB_PATH):
        init_db()

    # --------------------------------------------------------
    # Load trained models
    # --------------------------------------------------------

    load_models()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def current_minutes_to_departure(dep_hour, delay_mins=0):
    """
    Simulated countdown to departure.

    The prototype uses a simulated time calculation so that
    the demo remains meaningful regardless of the real clock.
    """

    now = datetime.now()

    sim_minutes = (
        (dep_hour * 60)
        - (now.hour * 60 + now.minute)
        + delay_mins
    )

    # Keep the countdown inside a sensible demo window
    if sim_minutes > 60 or sim_minutes < 0:
        sim_minutes = 5 + (now.minute % 55)

    return int(sim_minutes)


def passenger_risk(p, flight):
    """
    Calculate missed-boarding risk for a passenger.
    """

    mins = current_minutes_to_departure(
        flight["dep_hour"],
        flight["delay_mins"]
    )

    features = {
        "minutes_to_departure": mins,
        "checked_in_at_gate": p["checked_in_at_gate"],
        "zone_distance_min": ZONE_DISTANCE_MIN.get(
            p["zone"],
            6
        ),
        "prior_late_count": p["prior_late_count"],
        "is_frequent_flyer": p["is_frequent_flyer"],
    }

    result = predict_missed_boarding(features)

    result["minutes_to_departure"] = mins

    return result


# ============================================================
# AUTHENTICATION
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """
    Redirect homepage to passenger interface.
    """

    return redirect(url_for("passenger"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Staff/admin login.
    """

    ensure_ready()

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        )

        password = request.form.get(
            "password",
            ""
        )

        c = conn()

        row = c.execute(
            """
            SELECT *
            FROM users
            WHERE username=?
            AND password=?
            """,
            (
                username,
                hash_pw(password)
            )
        ).fetchone()

        c.close()

        if row:

            session["user"] = row["username"]
            session["role"] = row["role"]

            return redirect(
                url_for("dashboard")
            )

        flash("Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """
    Log the current user out.
    """

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# STAFF / ADMIN DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    """
    Staff dashboard.

    Displays flights, delay prediction, passengers,
    and missed-boarding risk.
    """

    ensure_ready()

    if "user" not in session:
        return redirect(
            url_for("login")
        )

    c = conn()

    flights = c.execute(
        """
        SELECT *
        FROM flights
        ORDER BY dep_hour
        """
    ).fetchall()

    data = []

    for f in flights:

        # ----------------------------------------------------
        # Flight delay prediction
        # ----------------------------------------------------

        delay_prediction = predict_delay({
            "dep_hour": f["dep_hour"],

            "carrier_idx": CARRIER_IDX.get(
                f["carrier"],
                0
            ),

            "weather_bad":
                1 if f["status"] == "Delayed" else 0,

            "inbound_late":
                1 if f["delay_mins"] > 0 else 0,

            "is_weekend":
                1 if datetime.now().weekday() >= 5 else 0,
        })

        # ----------------------------------------------------
        # Get passengers for flight
        # ----------------------------------------------------

        passengers = c.execute(
            """
            SELECT *
            FROM passengers
            WHERE flight_id=?
            """,
            (f["id"],)
        ).fetchall()

        risks = []

        for p in passengers:

            risk = passenger_risk(
                p,
                f
            )

            risks.append({
                "id": p["id"],
                "name": p["name"],
                "lang": LANG_NAMES[p["lang"]],
                "zone": p["zone"],
                "checked_in":
                    p["checked_in_at_gate"],
                "risk": risk["risk"],
                "band": risk["band"],
                "mins":
                    risk["minutes_to_departure"]
            })

        # Highest-risk passengers first
        risks.sort(
            key=lambda x: x["risk"],
            reverse=True
        )

        high = sum(
            1
            for x in risks
            if x["band"] == "HIGH"
        )

        data.append({
            "flight": dict(f),
            "delay": delay_prediction,
            "pax": risks,
            "high": high
        })

    c.close()

    return render_template(
        "dashboard.html",
        data=data,
        role=session.get("role"),
        user=session.get("user")
    )


# ============================================================
# FIRE FLIGHT EVENT
# ============================================================

@app.route("/fire_event", methods=["POST"])
def fire_event():
    """
    Staff triggers a flight event.

    Supports:

        - boarding_open
        - final_call
        - delay
        - gate_change
        - risk_alert

    If a target zone is supplied, notifications are sent only
    to passengers currently in that zone.

    Gate-change announcements are sent to all passengers on
    the affected flight because a gate change affects everyone.
    """

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if "user" not in session:

        return jsonify({
            "ok": False,
            "error": "unauthorised"
        }), 401

    # --------------------------------------------------------
    # Validate flight ID
    # --------------------------------------------------------

    try:

        flight_id = int(
            request.form["flight_id"]
        )

    except (KeyError, ValueError):

        return jsonify({
            "ok": False,
            "error": "Invalid flight ID"
        }), 400

    # --------------------------------------------------------
    # Read submitted values
    # --------------------------------------------------------

    event_type = request.form.get(
        "event_type",
        ""
    ).strip()

    target_zone = request.form.get(
        "zone",
        ""
    ).strip()

    newgate = request.form.get(
        "newgate",
        ""
    ).strip()

    # --------------------------------------------------------
    # Database connection
    # --------------------------------------------------------

    c = conn()

    try:

        # ----------------------------------------------------
        # Find flight
        # ----------------------------------------------------

        flight = c.execute(
            """
            SELECT *
            FROM flights
            WHERE id=?
            """,
            (flight_id,)
        ).fetchone()

        if not flight:

            return jsonify({
                "ok": False,
                "error": "Flight not found"
            }), 404

        # ----------------------------------------------------
        # GATE CHANGE
        # ----------------------------------------------------

        if event_type == "gate_change":

            if newgate:

                c.execute(
                    """
                    UPDATE flights
                    SET gate=?
                    WHERE id=?
                    """,
                    (
                        newgate,
                        flight_id
                    )
                )

        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        if event_type == "delay":

            c.execute(
                """
                UPDATE flights
                SET status='Delayed',
                    delay_mins=45
                WHERE id=?
                """,
                (flight_id,)
            )

        # ----------------------------------------------------
        # BOARDING OPEN
        # ----------------------------------------------------

        if event_type == "boarding_open":

            c.execute(
                """
                UPDATE flights
                SET status='Boarding'
                WHERE id=?
                """,
                (flight_id,)
            )

        # ----------------------------------------------------
        # FINAL CALL
        # ----------------------------------------------------

        if event_type == "final_call":

            c.execute(
                """
                UPDATE flights
                SET status='Final Call'
                WHERE id=?
                """,
                (flight_id,)
            )

        # ----------------------------------------------------
        # Refresh flight information after update
        # ----------------------------------------------------

        flight = c.execute(
            """
            SELECT *
            FROM flights
            WHERE id=?
            """,
            (flight_id,)
        ).fetchone()

        # ====================================================
        # SELECT PASSENGERS TO NOTIFY
        # ====================================================

        # Gate changes affect everyone on the flight
        if event_type == "gate_change":

            passengers = c.execute(
                """
                SELECT *
                FROM passengers
                WHERE flight_id=?
                """,
                (flight_id,)
            ).fetchall()

        # Zone-specific announcement
        elif target_zone:

            passengers = c.execute(
                """
                SELECT *
                FROM passengers
                WHERE flight_id=?
                AND zone=?
                """,
                (
                    flight_id,
                    target_zone
                )
            ).fetchall()

        # No zone selected = everyone on flight
        else:

            passengers = c.execute(
                """
                SELECT *
                FROM passengers
                WHERE flight_id=?
                """,
                (flight_id,)
            ).fetchall()

        # ====================================================
        # CREATE NOTIFICATIONS
        # ====================================================

        count = 0

        announcement_message = ""

        for p in passengers:

            # ------------------------------------------------
            # Calculate current countdown
            # ------------------------------------------------

            mins = current_minutes_to_departure(
                flight["dep_hour"],
                flight["delay_mins"]
            )

            # ------------------------------------------------
            # Announcement fields
            # ------------------------------------------------

            fields = {
                "flight":
                    flight["flight_no"],

                "gate":
                    flight["gate"],

                "dest":
                    flight["destination"],

                "newgate":
                    newgate,

                "mins":
                    str(abs(mins))
            }

            # ------------------------------------------------
            # Generate multilingual announcement
            # ------------------------------------------------

            message = generate_announcement(
                event_type,
                lang=p["lang"],
                **fields
            )

            # Keep the first message for the terminal
            # announcement response.
            if not announcement_message:

                announcement_message = message

            # ------------------------------------------------
            # Save notification
            # ------------------------------------------------

            c.execute(
                """
                INSERT INTO notifications
                (
                    passenger_id,
                    flight_id,
                    event_type,
                    lang,
                    message,
                    zone
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    p["id"],
                    flight_id,
                    event_type,
                    p["lang"],
                    message,
                    p["zone"]
                )
            )

            count += 1

        # ----------------------------------------------------
        # Save changes
        # ----------------------------------------------------

        c.commit()

        # ----------------------------------------------------
        # Return response to dashboard JavaScript
        # ----------------------------------------------------

        return jsonify({
            "ok": True,
            "notified": count,
            "flight": flight["flight_no"],
            "event": event_type,
            "zone": target_zone,
            "message": announcement_message
        })

    except Exception as e:

        # ----------------------------------------------------
        # Rollback if anything goes wrong
        # ----------------------------------------------------

        c.rollback()

        print(
            "fire_event error:",
            e
        )

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

    finally:

        c.close()


# ============================================================
# SEND ML RISK ALERTS
# ============================================================

@app.route(
    "/send_risk_alerts",
    methods=["POST"]
)
def send_risk_alerts():
    """
    Pre-emptive notification system.

    The ML model identifies HIGH-risk passengers and
    sends them an early warning.
    """

    if "user" not in session:

        return jsonify({
            "ok": False,
            "error": "unauthorised"
        }), 401

    c = conn()

    # --------------------------------------------------------
    # Load all flights
    # --------------------------------------------------------

    flights = {
        f["id"]: f
        for f in c.execute(
            "SELECT * FROM flights"
        ).fetchall()
    }

    # --------------------------------------------------------
    # Load all passengers
    # --------------------------------------------------------

    passengers = c.execute(
        "SELECT * FROM passengers"
    ).fetchall()

    sent = 0

    # --------------------------------------------------------
    # Evaluate passengers
    # --------------------------------------------------------

    for p in passengers:

        flight = flights.get(
            p["flight_id"]
        )

        if not flight:
            continue

        risk = passenger_risk(
            p,
            flight
        )

        # ----------------------------------------------------
        # Notify HIGH risk passengers
        # ----------------------------------------------------

        if risk["band"] == "HIGH":

            fields = {
                "flight":
                    flight["flight_no"],

                "gate":
                    flight["gate"],

                "dest":
                    flight["destination"],

                "mins":
                    str(
                        abs(
                            risk[
                                "minutes_to_departure"
                            ]
                        )
                    )
            }

            message = generate_announcement(
                "risk_alert",
                lang=p["lang"],
                **fields
            )

            c.execute(
                """
                INSERT INTO notifications
                (
                    passenger_id,
                    flight_id,
                    event_type,
                    lang,
                    message,
                    zone
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    p["id"],
                    flight["id"],
                    "risk_alert",
                    p["lang"],
                    message,
                    p["zone"]
                )
            )

            sent += 1

    c.commit()

    c.close()

    return jsonify({
        "ok": True,
        "alerts": sent
    })


# ============================================================
# PASSENGER VIEW
# ============================================================

@app.route("/passenger")
def passenger():
    """
    Passenger selection page.
    """

    ensure_ready()

    c = conn()

    passengers = c.execute(
        """
        SELECT *
        FROM passengers
        ORDER BY name
        """
    ).fetchall()

    c.close()

    return render_template(
        "passenger.html",
        pax=passengers,
        langs=LANG_NAMES
    )


# ============================================================
# PASSENGER NOTIFICATION FEED
# ============================================================

@app.route("/passenger/<int:pid>")
def passenger_feed(pid):
    """
    Individual passenger notification page.
    """

    c = conn()

    passenger_data = c.execute(
        """
        SELECT *
        FROM passengers
        WHERE id=?
        """,
        (pid,)
    ).fetchone()

    # --------------------------------------------------------
    # Passenger not found
    # --------------------------------------------------------

    if not passenger_data:

        c.close()

        return "Passenger not found", 404

    # --------------------------------------------------------
    # Get passenger flight
    # --------------------------------------------------------

    flight = c.execute(
        """
        SELECT *
        FROM flights
        WHERE id=?
        """,
        (
            passenger_data["flight_id"],
        )
    ).fetchone()

    if not flight:

        c.close()

        return "Flight not found", 404

    # --------------------------------------------------------
    # Notifications
    # --------------------------------------------------------

    notifications = c.execute(
        """
        SELECT *
        FROM notifications
        WHERE passenger_id=?
        ORDER BY id DESC
        """,
        (pid,)
    ).fetchall()

    # --------------------------------------------------------
    # ML risk
    # --------------------------------------------------------

    risk = passenger_risk(
        passenger_data,
        flight
    )

    c.close()

    return render_template(
        "passenger_feed.html",
        p=passenger_data,
        f=flight,
        notes=notifications,
        risk=risk,
        langs=LANG_NAMES
    )


# ============================================================
# PASSENGER LIVE API
# ============================================================

@app.route("/api/feed/<int:pid>")
def api_feed(pid):
    """
    API endpoint polled by the passenger interface.

    Returns the latest notifications for the passenger.
    """

    c = conn()

    notifications = c.execute(
        """
        SELECT
            event_type,
            lang,
            message,
            zone,
            created_at
        FROM notifications
        WHERE passenger_id=?
        ORDER BY id DESC
        LIMIT 20
        """,
        (pid,)
    ).fetchall()

    c.close()

    return jsonify([
        dict(notification)
        for notification in notifications
    ])


# ============================================================
# UPDATE PASSENGER
# ============================================================

@app.route(
    "/api/update_passenger",
    methods=["POST"]
)
def update_passenger():
    """
    Passenger changes language or zone.

    Zone changes simulate passenger movement through
    different terminal areas.
    """

    try:

        passenger_id = int(
            request.form["pid"]
        )

    except (KeyError, ValueError):

        return jsonify({
            "ok": False,
            "error": "Invalid passenger ID"
        }), 400

    language = request.form.get(
        "lang"
    )

    zone = request.form.get(
        "zone"
    )

    c = conn()

    # --------------------------------------------------------
    # Update language
    # --------------------------------------------------------

    if language:

        c.execute(
            """
            UPDATE passengers
            SET lang=?
            WHERE id=?
            """,
            (
                language,
                passenger_id
            )
        )

    # --------------------------------------------------------
    # Update zone
    # --------------------------------------------------------

    if zone:

        c.execute(
            """
            UPDATE passengers
            SET zone=?
            WHERE id=?
            """,
            (
                zone,
                passenger_id
            )
        )

    c.commit()

    c.close()

    return jsonify({
        "ok": True
    })


# ============================================================
# APPLICATION STARTUP
# ============================================================

# Ensure database and ML models exist when the application
# is imported by a production server such as Gunicorn.
ensure_ready()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    # Debug mode is disabled by default.
    debug = (
        os.environ.get(
            "FLASK_DEBUG",
            "0"
        ) == "1"
    )

    app.run(
        debug=debug,
        host="0.0.0.0",
        port=port
    )