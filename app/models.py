from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///accounts.db'
db = SQLAlchemy(app)

ATM_DB_PATH = 'accounts.db'

class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text)
    account_number = db.Column(db.Integer, unique=True, nullable=False)
    pin = db.Column(db.Integer, unique=False, nullable=False)
    balance = db.Column(db.Integer)

    def __repr__(self):
        return f"Name : {self.name}, Account_Number: {self.account_number}, PIN:{self.pin}, Balance: {self.balance}"