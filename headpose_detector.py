import warnings
import datetime

import numpy as np
import torch
import math
from torchvision import transforms
import cv2
from utilities.dectect import AntiSpoofPredict
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from utilities.pfld.pfld import PFLDInference
from config.config import *
from web.sender import send_data


warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet    = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) 
        fc_in_dim = list(resnet.children())[-1].in_features 
        self.fc = nn.Linear(fc_in_dim, num_classes) 
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out



class HeadPoseDetector():
    def __init__(self):
        self.model_path = "./models/pfld_weights.pth.tar"
        self.device_id = 0
        checkpoint = torch.load(self.model_path, map_location=device)
        plfd_backbone = PFLDInference().to(device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        plfd_backbone.eval()
        self.plfd_backbone = plfd_backbone.to(device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.videoCapture = cv2.VideoCapture(0)
        fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("fps:", fps, "size:", size)
        self.model_test = AntiSpoofPredict(self.device_id)
        
        self.yaw_high = None
        self.yaw_low = None
        self.pitch_high = None
        self.pitch_low = None
        
        self.tolerance_degree_yaw = 0
        self.tolerance_degree_pitch = 0
        self.tolerance_frames = 2
        
        self.bad_pose_frames = 0
        
        
        self.preprocess_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        model_save_path = "./models/wiki2020.pth" #mode path
        self.emotion = None
        self.res18 = Res18Feature(pretrained = False)
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
        self.res18.load_state_dict(checkpoint['model_state_dict'])
        self.res18.cpu()
        self.res18.eval()
        

    def get_num(self, point_dict, name, axis):
        num = point_dict.get(f'{name}')[axis]
        num = float(num)
        return num

    def cross_point(self, line1, line2):
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]

        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]

        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        if (x4 - x3) == 0:
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            b2 = y3 * 1.0 - x3 * k2 * 1.0
        if k2 == None:
            x = x3
        else:
            x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]

    def point_line(self, point, line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        x3 = point[0]
        y3 = point[1]

        k1 = (y2 - y1)*1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        k2 = -1.0/k1
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]

    def point_point(self, point_1, point_2):
        x1 = point_1[0]
        y1 = point_1[1]
        x2 = point_2[0]
        y2 = point_2[1]
        distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
        return distance
    
    def get_setting(self):
        return f"yaw_high:{self.yaw_high}\nyaw_low:{self.yaw_low}\npitch_high: {self.pitch_high}\npitch_low:{self.pitch_low}"


    def get_pose(self, img, frame_index=None, time_stamp=None, video_name=None):
        height, width = img.shape[:2]

        image_bbox = self.model_test.get_bbox(img)
        x1 = image_bbox[0]
        y1 = image_bbox[1]
        x2 = image_bbox[0] + image_bbox[2]
        y2 = image_bbox[1] + image_bbox[3]
        w = x2 - x1
        h = y2 - y1

        size = int(max([w, h]))
        cx = x1 + w/2
        cy = y1 + h/2
        x1 = cx - size/2
        x2 = x1 + size
        y1 = cy - size/2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[int(y1):int(y2), int(x1):int(x2)]
        
        
        if frame_index is not None and frame_index % EXPRESSION_CHECK_FREQUENCY == 0:
            face = cropped[:, :, ::-1]
            image_tensor = self.preprocess_transform(face)
            tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)
            tensor=tensor.cpu()
            
            _, outputs = self.res18(tensor)
            score, predicts = torch.max(outputs, 1)
            self.emotion = result[int(predicts.cpu().data)]
            
            ret_json = dict(expression=int(predicts.cpu().data), score=float(score.cpu().data), time_stamp=time_stamp, video_name=video_name)
            
            print(ret_json)
            send_data(ret_json)
            
        
        
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, int(
                edy), dx, edx, cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transform(input).unsqueeze(0).to(device)
        _, landmarks = self.plfd_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach(
        ).numpy().reshape(-1, 2) * [112, 112]
        point_dict = {}
        i = 0
        for (x, y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x, y]
            i += 1

        # yaw
        point1 = [self.get_num(point_dict, 1, 0), self.get_num(point_dict, 1, 1)]
        point31 = [self.get_num(point_dict, 31, 0), self.get_num(point_dict, 31, 1)]
        point51 = [self.get_num(point_dict, 51, 0), self.get_num(point_dict, 51, 1)]
        crossover51 = self.point_line(
            point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = self.point_point(point1, point31) / 2
        yaw_right = self.point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)

        # pitch
        pitch_dis = self.point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)

        # roll
        roll_tan = abs(self.get_num(point_dict, 60, 1) - self.get_num(point_dict, 72, 1)) / \
            abs(self.get_num(point_dict, 60, 0) - self.get_num(point_dict, 72, 0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if self.get_num(point_dict, 60, 1) > self.get_num(point_dict, 72, 1):
            roll = -roll
        roll = int(roll)

        return yaw, pitch, roll
    
    def check_setup_status(self):
        if not self.yaw_low:
            return LEFT #left
        elif not self.yaw_high:
            return RIGHT #right
        elif not self.pitch_low:
            return UP #up
        elif not self.pitch_high:
            return DOWN #down
        else:
            return DONE    
    
    def init_setup(self):
        self.yaw_low, self.yaw_high, self.pitch_high, self.pitch_low = None, None, None, None

    def setup(self, frame):    
        yaw, pitch, roll = self.get_pose(frame)
        
        # cv2.putText(frame, f"Head_Yaw(degree): {yaw}", (30, 50),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Head_Pitch(degree): {pitch}", (
        #     30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Head_Roll(degree): {roll}", (30, 150),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        # videoWriter.write(img)
        
        
        status = self.check_setup_status()
        

        if status == LEFT:
            self.yaw_low = yaw
            ret_status = RIGHT
            
        elif status == RIGHT:
            self.yaw_high = yaw
            ret_status = UP
            
        elif status == UP:
            self.pitch_low = pitch
            ret_status = DOWN
            
        elif status == DOWN:    
            self.pitch_high = pitch
            ret_status = DONE
        info = f"yaw_low = {self.yaw_low}\nyaw_high = {self.yaw_high}\npitch_high = {self.pitch_high}\npitch_low = {self.pitch_low}\n" + f"Please Look at {status} most side of your screen\nPress 'enter' to confirm\n'c' to clear the selection."
        
        
        return ret_status, info    
    
    def check_pose(self, yaw, pitch):
            if (yaw > self.yaw_low - self.tolerance_degree_yaw 
                and yaw < self.yaw_high + self.tolerance_degree_yaw
                and pitch > self.pitch_low - self.tolerance_degree_yaw
                and pitch < self.pitch_high + self.tolerance_degree_yaw):
                return True
            else:
                return False
            
    def detect(self, frame, frame_index, time_stamp, video_name):
        
        yaw, pitch, roll = self.get_pose(frame, frame_index, time_stamp, video_name)

        if not self.check_pose(yaw, pitch):
            self.bad_pose_frames += 1
        else:
            self.bad_pose_frames = 0
        if self.bad_pose_frames > self.tolerance_frames:
            return False, yaw, pitch, roll
        else:
            return True, yaw, pitch, roll
            
# if __name__ == "__main__":
#     headPostDetector = HeadPoseDetector()
#     headPostDetector.setup()
#     headPostDetector.detect()
                

            
        
