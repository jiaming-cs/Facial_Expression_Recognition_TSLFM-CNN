from flask import Flask, request, Response
from flask_sqlalchemy import SQLAlchemy
import sys

from sqlalchemy.orm import backref 
sys.path.append("..") 
from config.config import SQLALCHEMY_DATABASE_URI
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(25), nullable=False)
    
    
    
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    
    
    
class Expressions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    expression = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
    time_stamp = db.Column(db.Integer, nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    user_expression = db.relationship('User', backref='experssion', lazy='joined')
    
    video_expression = db.relationship('Video', backref='expression', lazy='joined')
    
    def __repr__(self):
        return f'<{self.id} {self.expression} {self.score} {self.time_stamp} {self.video_id}>'
    
db.create_all()  

def get_or_create(table, name):
    record = table.query.filter_by(name=name).first()
    if record is None:
        record = table(name=name)
        db.session.add(record)
        db.session.commit()
    return record.id    

@app.route('/exp', methods=['POST'])
def add_exp():
    data = request.get_json()
    exp = Expressions(expression = data['expression'], score= data['score'], time_stamp = data['time_stamp'], video_id=data['video_id'], user_id=data['user_id'])
    db.session.add(exp)
    db.session.commit()
    print('add new record')
    return Response("ok", 200)

@app.route('/user', methods=['POST'])
def add_user():
    data = request.get_json()
    id_ = get_or_create(User, data['name'])
    
    return Response(str(id_), 200)

@app.route('/video', methods=['POST'])
def add_video():
    data = request.get_json()
    id_ = get_or_create(Video, data['name'])
    
    return Response(str(id_), 200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    

