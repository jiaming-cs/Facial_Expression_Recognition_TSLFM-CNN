from flask import Flask, request, Response
from flask_sqlalchemy import SQLAlchemy
import sys 
sys.path.append("..") 
from config.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_DB_FILE
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)



class Experssions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    experssion = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
    time_stamp = db.Column(db.String(25), nullable=False)
    video_name = db.Column(db.String(30), nullable=False)
    
    def __repr__(self):
        return f'<{self.id} {self.experssion} {self.score} {self.time_stamp} {self.video_name}>'
    
db.create_all()  
    
@app.route('/exp', methods=['POST'])
def hello_world():
    data = request.get_json()
    exp = Experssions(experssion = data['expression'], score= data['score'], time_stamp = data['time_stamp'], video_name=data['video_name'])
    db.session.add(exp)
    db.session.commit()
    print('add new record')
    return Response("ok", 200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    

