from enum import Enum


class ACTION(Enum):
    IDLE = 0
    WALK = 1
    WAVE = 2
    BRUSH_HAIR = 3
    CATCH = 4
    CLAP = 5
    CLIMB_STAIRS = 6
    GOLF = 7
    JUMP = 8
    KICK_BALL = 9
    PICK = 10
    POUR = 11
    PULLUP = 12
    PUSH = 13
    RUN = 14
    SHOOT_BALL = 15
    SHOOT_BOW = 16
    SHOOT_GUN = 17
    SIT = 18
    STAND = 19
    SWING_BASEBALL = 20
    THROW = 21
    RANGE_OF_MOTION = 22
    LOOK_FOR_TRAFFIC = 23
    HITCHHIKE = 24
    TURN_AROUND = 25
    WORK = 26
    ARGUE = 27
    STUMBLE = 28
    OPEN_DOOR = 29
    FALL = 30
    STAND_UP = 31
    FIGHT = 32
    JOG = 33
    WAVE_CAR_OUT = 34


JHMDB_ACTIONS = [
    ACTION.BRUSH_HAIR,
    ACTION.CATCH,
    ACTION.CLAP,
    ACTION.CLIMB_STAIRS,
    ACTION.GOLF,
    ACTION.JUMP,
    ACTION.KICK_BALL,
    ACTION.PICK,
    ACTION.POUR,
    ACTION.PULLUP,
    ACTION.PUSH,
    ACTION.RUN,
    ACTION.SHOOT_BALL,
    ACTION.SHOOT_BOW,
    ACTION.SHOOT_GUN,
    ACTION.SIT,
    ACTION.STAND,
    ACTION.SWING_BASEBALL,
    ACTION.THROW,
    ACTION.WALK,
    ACTION.WAVE
]

sim_name_to_action_mapping = {
    "Walking": ACTION.WALK,
    "Move": ACTION.WALK,
    "Running": ACTION.RUN,
    "Stand": ACTION.STAND,
    "Standing": ACTION.STAND,
    "ROM": ACTION.RANGE_OF_MOTION,
    "Carefully look for traffic": ACTION.LOOK_FOR_TRAFFIC,
    "Look for traffic": ACTION.LOOK_FOR_TRAFFIC,
    "Fast look for traffic": ACTION.LOOK_FOR_TRAFFIC,
    "Hitchhike": ACTION.HITCHHIKE,
    "Turn around": ACTION.TURN_AROUND,
    "Work": ACTION.WORK,
    "Idle": ACTION.IDLE,
    "Wave": ACTION.WAVE,
    "Kick": ACTION.KICK_BALL,
    "Argue": ACTION.ARGUE,
    "Fight": ACTION.FIGHT,
    "Stumble": ACTION.STUMBLE,
    "Open a door": ACTION.OPEN_DOOR,
    "Fall": ACTION.FALL,
    "Stand up": ACTION.STAND_UP,
    "Jogging": ACTION.JOG,
    "Throw": ACTION.THROW,
    "Look around": ACTION.IDLE
}