import numpy as np


def calculate_gap_y(angle_degrees, side_length):
    angle_radians = np.radians(angle_degrees)
    other_side_length = np.tan(angle_radians) * side_length
    return other_side_length


class BoxCfg:
    def __init__(self, scale, box_size_x=None, box_size_y=None, length_y=None, angle=None):
        # self.orientation = orientation
        # self.facing = facing
        # self.box_size_y = box_size_y if box_size_y else 600 / scale
        self.scale = scale
        self.angle = angle if angle else 15 / scale

    def create_box_right(self, width, height):
        x_mid = width / 2
        y_mid = height / 2
        box_size_x = width / 5
        box_size_y = height / self.scale

        length_x = width / 2.5
        length_y = height / 2.5

        gap_y = height / 10
        # gap_y = calculate_gap_y(self.angle, (x_mid - length_x))

        xf1 = x_mid - length_x
        xb2 = x_mid + length_x
        xf2 = xf1 + box_size_x
        xb1 = xb2 - box_size_x

        yf3 = y_mid + length_y
        yf4 = yf3 - gap_y
        yf1 = yf4 - box_size_y
        yf2 = yf3 - box_size_y

        yb1 = y_mid - length_y
        yb2 = yb1 + gap_y
        yb3 = yb2 + box_size_y
        yb4 = yb1 + box_size_y

        front_top_left = (xf1, yf1)
        front_top_right = (xf2, yf2)
        front_bot_right = (xf2, yf3)
        front_bot_left = (xf1, yf4)
        front_face = [front_top_left, front_top_right, front_bot_right, front_bot_left]
        front_face = [(int(x), int(y)) for x, y in front_face]

        back_top_left = (xb1, yb1)
        back_top_right = (xb2, yb2)
        back_bot_right = (xb2, yb3)
        back_bot_left = (xb1, yb4)
        back_face = [back_top_left, back_top_right, back_bot_right, back_bot_left]
        back_face = [(int(x), int(y)) for x, y in back_face]

        return front_face, back_face

    def create_box_left(self, width, height):
        x_mid = width / 2
        y_mid = height / 2
        length_x = width / 2.5
        length_y = 300
        gap_y = self.box_size_y * 2 / 3
        # gap_y = calculate_gap_y(self.angle, (x_mid - length_x))

        xf1 = x_mid - length_x
        xb2 = x_mid + length_x
        xf2 = xf1 + self.box_size_x
        xb1 = xb2 - self.box_size_x

        yf3 = y_mid + length_y
        yf4 = yf3 - gap_y
        yf1 = yf4 - self.box_size_y
        yf2 = yf3 - self.box_size_y

        yb1 = y_mid - length_y - gap_y
        yb2 = yb1 + gap_y
        yb3 = yb2 + self.box_size_y
        yb4 = yb1 + self.box_size_y

        front_top_left = (xf1, yf1)
        front_top_right = (xf2, yf2)
        front_bot_right = (xf2, yf3)
        front_bot_left = (xf1, yf4)
        front_face = [front_top_left, front_top_right, front_bot_right, front_bot_left]
        front_face = [(int(x), int(y)) for x, y in front_face]

        back_top_left = (xb1, yb1)
        back_top_right = (xb2, yb2)
        back_bot_right = (xb2, yb3)
        back_bot_left = (xb1, yb4)
        back_face = [back_top_left, back_top_right, back_bot_right, back_bot_left]
        back_face = [(int(x), int(y)) for x, y in back_face]

        return front_face, back_face
