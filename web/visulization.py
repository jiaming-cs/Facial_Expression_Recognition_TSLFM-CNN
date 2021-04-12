import matplotlib.pyplot as plt
import os
from server import Expressions, User, Video


expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
colors = ['red', 'gold', 'indigo', 'yellow', 'lightgrey', 'deeppink', 'cyan']

# exps = Expressions.query.order_by(Expressions.id).all()


users = User.query.order_by(User.id).all()
videos = Video.query.order_by(Video.id).all()

for user in users:
    for video in videos:
        color_seq = []
        score_seq = []
        time_seq  = []
        for i, exp in enumerate(Expressions.query.filter_by(user_id=user.id, video_id=video.id).order_by(Expressions.time_stamp).all()) :
            if i>0:
                t = exp.time_stamp//500
                if t == time_seq[-1]:
                    continue      
            color_seq.append(colors[exp.expression])
            score_seq.append(exp.score)
            time_seq.append(exp.time_stamp//500)

        if color_seq:
            plt.bar(time_seq, score_seq, color=color_seq, alpha=0.6) 
            plt.plot(time_seq, score_seq, '--')
            plt.ylim((0.5, 1.2))
            plt.xlabel('Time/s')
            plt.ylabel('Score')
            plt.title(f'Expression Changes {user.name} - Video:{video.id}')
            plt.show()

