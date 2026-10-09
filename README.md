# AI-Driven Flight Announcement and Passenger Notification System

A prototype that automates multi-lingual, zone-targeted passenger alerts for a
Nigerian airport (Benin Airport context), and uses machine learning to predict
and pre-empt missed boardings. Built with Python, Flask, SQLite and scikit-learn.

The system runs on realistic simulated flight and passenger data. It does not
connect to live airline or airport systems.

## What it does

1. Converts flight events (boarding open, final call, gate change, delay) into
   automated announcements and individual push notifications.
2. Delivers each notification in the passenger's chosen language: English,
   Nigerian Pidgin, Hausa, or Edo.
3. Targets notifications to the passengers and terminal zone a flight concerns,
   instead of broadcasting to the whole terminal.
4. Uses a trained machine learning model to score each passenger's risk of
   missing boarding, and a second model to predict flight delays, so staff can
   alert at-risk passengers before the boarding call is lost.

## Requirements

- Python 3.10 or newer
- The packages in requirements.txt

## Setup and run

```
pip install -r requirements.txt
python core/ml/generate_data.py      # generate the simulated datasets
python core/ml/train.py              # train the two models (prints metrics)
python core/db.py                    # create and seed the database
python app.py                        # start the server
```

Then open http://127.0.0.1:5000 in a browser.

- Passenger view: open directly, pick a passenger, choose language and zone.
- Staff console: sign in with  admin / admin  (administrator) or  staff / staff.

## Project layout

```
app.py                     Flask application (routes for console and passenger view)
core/db.py                 SQLite schema and seed data
core/ml/generate_data.py   Simulated dataset generator
core/ml/train.py           Model training and evaluation (writes metrics.json)
core/ml/engine.py          Runtime prediction, NLP announcement generation, translation
core/models/               Trained models (.pkl) and metrics.json
core/templates/            HTML templates
core/static/style.css      Styling
data/                      Datasets and the SQLite database
requirements.txt
```

## The machine learning

Two supervised models, trained on held-out splits:

- Missed-boarding risk (Random Forest). Features: minutes to departure, gate
  check-in status, walking distance from the passenger's zone to the gate, prior
  lateness, frequent-flyer status. This is the primary predictive feature.
- Flight-delay prediction (Random Forest). Features: scheduled departure hour,
  carrier, adverse weather flag, inbound-aircraft-late flag, weekend flag.

Running train.py prints accuracy, precision, recall, F1 and the confusion matrix
for each model, and saves them to core/models/metrics.json. These are the figures
reported in Chapter Four.

## Multi-lingual announcements

Announcement text is produced by a controlled template-and-fill method, so the
phrasing is always correct and never fabricated. Indigenous-language output uses
verified curated strings, which is the dependable choice given the current
maturity of machine translation for these languages. Spoken output in the
browser uses the Web Speech API, which can be swapped for a server-side
text-to-speech engine without changing the rest of the system.

## Note on storage

The prototype uses SQLite so it runs anywhere with no database server to install.
The data layer is written so it can be repointed at PostgreSQL for a production
deployment without changing the application logic.

## Deploying to Render (live hosting)

This package is ready to deploy to Render as a web service.

1. Push this folder to a GitHub repository.
2. On render.com, create a new Web Service and connect that repository.
3. Render reads render.yaml automatically. If asked, use:
   - Build command:  pip install -r requirements.txt && python core/ml/generate_data.py && python core/ml/train.py && python core/db.py
   - Start command:  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
4. Click Create Web Service. The first build takes a few minutes while it
   installs the libraries and trains the two models.
5. When it is live, open the URL Render gives you. The staff console logins
   are admin/admin and staff/staff.
