import nxbt 
from nxbt import Buttons
from nxbt import Sticks

def turn_right(nx, controller_idx):
    print("Turn right")
    nx.tilt_stick(controller_idx, Sticks.RIGHT_STICK, -100, 0)

def turn_left(nx, controller_idx):
    print("Turn left")
    nx.tilt_stick(controller_idx, Sticks.LEFT_STICK, -100, 0)

def speed_up(nx, controller_idx):
    print("Speed up")
    nx.press_buttons(controller_idx, [nxbt.Buttons.A], down=1.0)

#TODO fix these
def slow_down(nx, controller_idx):
    print("Slow down")

def power_up(nx, controller_idx):
    print("Power up!")