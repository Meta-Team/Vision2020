import numpy as np
import math
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)


class PID:
    def __init__(self, P=1.0, I=0.05, D=0.1, up_bound=2000):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.up_bound = up_bound
        self.clear()

    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.output = 0.0

    def update(self, error, dt):
        delta_error = error - self.last_error
        self.PTerm = self.Kp * error
        self.ITerm += error * dt
        self.ITerm = min(self.up_bound, max(-self.up_bound, self.ITerm))
        self.DTerm = delta_error / (abs(dt)+0.01)
        self.last_error = error
        self.output = self.PTerm + \
            (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
        print('PID:',self.PTerm,self.ITerm, self.DTerm)


def print_target(target_x, target_y, enemy_x, enemy_y):
    print('\033[0;31;42m', int(target_x), int(target_y),
          int(enemy_x), int(enemy_y), '\033[0m')


class PostProcess:

    image_width = 640
    image_height = 480
    angle_per_pixel_x = 90/640
    angle_per_pixel_y = 90/480

    max_dx = 100  # target change
    max_dy = 100

    delay_yaw = 0.05  # seconds
    delay_pitch = 0.05

    gain = 0.6

    def __init__(self):
        self.PID_yaw = PID()
        self.PID_pitch = PID(P=1.0, I=0.05, D=0.1, up_bound=2000)
        self.x_records = np.zeros(10)
        self.y_records = np.zeros(10)
        self.t_records = np.zeros(10)

        self.invalid_count = 0
        self.valid_count = 0
        self.t_prev = 0

    def ideal_target(self, enemy_height):
        # get distance form box_height and get one ideal target for shoot per distance;
        # only the enemy on this target position can be hit
        x = math.floor(300 + enemy_height*0.001)
        y = math.floor(200 + enemy_height*0.001)
        return x, y

    def poly2fit(self, x, y, x_new):
        coeffs = np.polyfit(x, y, 2)
        return coeffs[0]*x_new*x_new + coeffs[1]*x_new + coeffs[2]

    def predict(self, dx, dy, t):
        self.x_records[0:-1] = self.x_records[1:]
        self.y_records[0:-1] = self.y_records[1:]
        self.t_records[0:-1] = self.t_records[1:]
        self.x_records[-1] = dx
        self.y_records[-1] = dy
        self.t_records[-1] = t
        if self.valid_count < 5:
            return dx, dy

        tt = self.t_records[-8:]
        xx = self.x_records[-8:]
        yy = self.y_records[-8:]
        # return np.mean(xx), np.mean(yy)
        tx_new = t+self.delay_yaw
        ty_new = t+self.delay_pitch
        return self.poly2fit(tt, xx, tx_new),  self.poly2fit(tt, yy, ty_new)

    def post_process(self, bboxes, t):
        # return an array:
        # 1. x position of the enemy, [0~63], positive right;
        # 2. y position of the enemy, [0~31], positive down;
        # 3. delta_yaw from the target to the current, [-89 ~ 89], in degree
        # 4. delta_pitch from the target to the current, [-89 ~ 89], in degree
        t_start = time.time()
        scores = []
        for arr in bboxes:
            x_left, y_top, x_right, y_bottom, enemy_class, confidence = arr
            y_top = 1 - y_top
            y_bottom = 1 - y_bottom
            if enemy_class != 0 or confidence < 0.1:  # must be red
                continue
            enemy_width = (x_right-x_left)*self.image_width
            enemy_height = (y_top - y_bottom)*self.image_height
            enemy_x = (x_left+x_right)/2*self.image_width
            enemy_y = (y_bottom+y_top)/2*self.image_height
            # target position, near the center but may not be the center
            target_x, target_y = self.ideal_target(enemy_height)
            distance = abs(target_x-enemy_x) + abs(target_y-enemy_y)
            # distance_score = 1000  if abs(target_x+target_y-enemy_x-enemy_y) < 100 else 0
            # confidence_score = confidence*100000
            # height_width_ratio = enemy_height/enemy_width
            # angle_score = height_width_ratio*500 if height_width_ratio > 0.5 else -500
            scores.append([target_x, target_y, enemy_x, enemy_y,confidence])

        # when there is no input
        if len(scores) == 0:
            if self.invalid_count < 10:
                self.invalid_count += 1
            else:
                self.valid_count = 0
                self.PID_yaw.clear()
                self.PID_pitch.clear()
            # print("no enemy detect!")
            return 0, 0, 5000, 5000  # keep current position
        if self.valid_count < 10:
            self.valid_count += 1
        self.invalid_count = 0

        # select the maximum score
        temp = [i[0] for i in scores]
        idx = temp.index(max(temp))
        target_x, target_y, enemy_x, enemy_y, confidence = scores[idx]
        # print_target(target_x, target_y, enemy_x, enemy_y)
        yaw_diff = enemy_x - target_x
        pitch_diff = enemy_y - target_y

        if self.valid_count > 1 and (abs(yaw_diff-self.x_records[9]) > self.max_dx or abs(pitch_diff-self.y_records[9]) > self.max_dx):
            return 0, 0, 5000, 5000
            self.valid_count = 1

        yaw_diff_predicted, pitch_diff_predicted = self.predict(yaw_diff, pitch_diff, t)
        yaw_diff = yaw_diff * (1-self.gain) + yaw_diff_predicted * self.gain
        pitch_diff = pitch_diff_predicted * (1-self.gain) + pitch_diff_predicted * self.gain
        # yaw_diff = yaw_diff_predicted
        # pitch_diff = pitch_diff_predicted

        # print("after_predict: yaw_diff", yaw_diff, "pitch_diff", pitch_diff)

        dt = self.t_records[-1] - self.t_records[-2]
        self.PID_yaw.update(yaw_diff, dt)
        self.PID_pitch.update(pitch_diff, dt)

        yaw_diff = self.PID_yaw.output
        pitch_diff = self.PID_pitch.output

        # print("after PID: yaw_diff", yaw_diff, "pitch_diff", pitch_diff)

        yaw_diff *= self.angle_per_pixel_x
        pitch_diff *= self.angle_per_pixel_x

        # print('yaw_diff',yaw_diff,"pitch_diff", pitch_diff)

        # for robot
        # notice: 4970 < yaw_int < 5100 won't work; so as pitch_int;
        yaw_diff = arg_limit(-89,yaw_diff,89) 
        pitch_diff = arg_limit(-89,pitch_diff,89)
        yaw_int = math.floor((yaw_diff + 90)/180*10000)
        pitch_int = math.floor((pitch_diff + 90) / 180 * 10000)



        yaw_int = arg_limit(3000, yaw_int, 7000)
        pitch_int = arg_limit(4000, pitch_int, 6000)

        enemy_x = math.floor(enemy_x/self.image_width*63.99)
        enemy_y = math.floor(enemy_y/self.image_height*31.99)

        t_curr = time.time()
        print('fps=', 1/(t_curr-self.t_prev),'post precess delay=', 1000*(t_curr-t_start), 'ms')
        self.t_prev = t_curr
        return enemy_x, enemy_y, yaw_int, pitch_int


def arg_limit(a, x, b):
    return min(max(a, x), b)
