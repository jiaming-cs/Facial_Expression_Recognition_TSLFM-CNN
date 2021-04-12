import os

expression_mapping = dict(zip(list(range(7)), ["AGR", "CTM", "DIS", "FER", "HAP", "SAD", "SUP"]))


SKIP_FRAMES = 5
MAP_LENGTH = 2


# CAMERA_TYPE = './data/test.mp4'
CAMERA_TYPE = 0
MODEL_PATH = './models/model_weights.h5'

BBOX_COLOR = (255, 0, 0)
BBOX_THINKNESS = 2
LANDMARKS_COLOR = (0, 255, 0)
LANDMARKS_RADIOUS = 1

FONT_SCALE = 1
FONT_THINKNESS = 2
FONT_COLOR = (0, 0, 255)

LEFT = 'Left'
RIGHT = 'Right'
UP = 'Up'
DOWN = 'Down'
DONE = 'Done'

LINE_HEIGHT = 15

EXPRESSION_CHECK_FREQUENCY = 5


POST_DATA_URL_EXP = "http://127.0.0.1:5000/exp"
POST_DATA_URL_USER = "http://127.0.0.1:5000/user"
POST_DATA_URL_VIDEO = "http://127.0.0.1:5000/video"
SQLALCHEMY_DATABASE_URI = r'sqlite:///{}'.format(os.path.join(os.getcwd(), "test.db"))
SQLALCHEMY_DB_FILE = os.path.join(os.getcwd(), "test.db")
