from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///detailedfighters.db"
db = SQLAlchemy(app)

class Fighter(db.Model):
    id = db.Column(db.Integer, primary_key=True, index=True)
    name = db.Column(db.String, nullable=False)
    record = db.Column(db.String)
    SLpM = db.Column(db.Float)
    Str_Acc = db.Column(db.String)
    SApM = db.Column(db.Float)
    Str_Def = db.Column(db.String)
    TD_Avg = db.Column(db.Float)
    TD_Acc = db.Column(db.String)
    TD_Def = db.Column(db.String)
    Sub_Avg = db.Column(db.Float)
    Height = db.Column(db.String)
    Weight = db.Column(db.String)
    Reach = db.Column(db.String)
    Stance = db.Column(db.String)
    DOB = db.Column(db.String)
    ufcstats_id = db.Column(db.String)  # names are not unique; see scrapers/fighter_ids.py

    fights = db.relationship("Fight", back_populates="fighter")

class Fight(db.Model):
    id = db.Column(db.Integer, primary_key=True, index=True)
    fighter_id = db.Column(db.Integer, db.ForeignKey('fighter.id'))
    result = db.Column(db.String)
    opponent = db.Column(db.String)
    fighterKD = db.Column(db.String)
    fighterSTR = db.Column(db.String)
    fighterTD = db.Column(db.String)
    fighterSUB = db.Column(db.String)
    opponentKD = db.Column(db.String)
    opponentSTR = db.Column(db.String)
    opponentTD = db.Column(db.String)
    opponentSUB = db.Column(db.String)
    titlefight = db.Column(db.Boolean)
    event = db.Column(db.String)
    date = db.Column(db.String)
    method = db.Column(db.String)
    round = db.Column(db.String)
    time = db.Column(db.String)

    fighter = db.relationship('Fighter', back_populates='fights')