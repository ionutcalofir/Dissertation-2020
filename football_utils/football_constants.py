from enum import IntEnum

class PlayerActions(IntEnum):
    e_FunctionType_None = 0
    e_FunctionType_Movement = 1
    e_FunctionType_BallControl = 2
    e_FunctionType_Trap = 3
    e_FunctionType_ShortPass = 4
    e_FunctionType_LongPass = 5
    e_FunctionType_HighPass = 6
    e_FunctionType_Header = 7
    e_FunctionType_Shot = 8
    e_FunctionType_Deflect = 9
    e_FunctionType_Catch = 10
    e_FunctionType_Interfere = 11
    e_FunctionType_Trip = 12
    e_FunctionType_Sliding = 13
    e_FunctionType_Special = 14

PASS_ACTIONS = [
                PlayerActions.e_FunctionType_ShortPass,
                PlayerActions.e_FunctionType_LongPass,
                PlayerActions.e_FunctionType_HighPass
               ]
SHOT_ACTIONS = [
                PlayerActions.e_FunctionType_Shot
               ]
STEPS_PER_FRAME = 10
DATASET_GENERATION_FRAMES_WINDOW = 30 * STEPS_PER_FRAME
SHORT_EDGE_SIZE = 256
