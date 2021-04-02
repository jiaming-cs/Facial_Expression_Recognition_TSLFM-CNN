import matplotlib.pyplot as plt
from server import Experssions, db


expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
colors = ['red', 'gold', 'indigo', 'yellow', 'lightgrey', 'deeppink', 'cyan']

exps = Experssions.query.order_by(Experssions.id).all()

color_seq = []
score_seq = []


for exp in exps:
    
    color_seq.append(colors[exp.experssion])
    score_seq.append(exp.score)
    
plt.bar(range(len(score_seq)), score_seq, color=color_seq, alpha=0.6) 
plt.xlabel('Expression')
plt.ylabel('Score')
plt.title('Expression Changes')
plt.legend()
plt.show()

