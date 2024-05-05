#controls and connecting to switch controller, send controls
#author @ taylor tam

import nxbt 
from nxbt import Buttons
from nxbt import Sticks
import time

##### START SEQUENCE #####
#starts from change grip/order --> Mario Kart
MACRO = """
LOOP 12
    B 0.1s
    0.1s
1s
DPAD_UP 0.25s
L_STICK@-100+000 0.2s
A 0.1s
0.2s
DPAD_UP 0.25s
0.1s
A 0.1s
"""

MACRO2 = """
ZL+ZR 0.25s
"""

##### CONNECT & START #####
#connects and runs a start sequence
def connect_controller():
    first_connect = False
    print("Connecting...")
    nx = nxbt.Nxbt()
    try:
        #find old controller
        controller_index = nx.create_controller(
        nxbt.PRO_CONTROLLER,
        reconnect_address=nx.get_switch_addresses())
    except:
        #if first time connecting to controller
        controller_index = nx.create_controller(nxbt.PRO_CONTROLLER)
        first_connect = True

    nx.wait_for_connection(controller_index)
    print("Initialized")

    if first_connect:
        #must be blocking 
        macro_id = nx.macro(controller_index, MACRO, block=True)
        time.sleep(3)
        print("Stopping Macro")
        nx.stop_macro(controller_index, macro_id)
        print("Stopped Macro")
    else:
        macro_id = nx.macro(controller_index, MACRO2, block=True)
        time.sleep(3)
        print("Stopping Macro")
        nx.stop_macro(controller_index, macro_id)
        print("Stopped Macro")

    print("Ready to play!")
    return nx, controller_index

##### CONTROLS #####
def turn_right(nx, controller_idx):
    print("Turn right")
    nx.tilt_stick(controller_idx, Sticks.LEFT_STICK, 100, 0, tilted=0.5)
    print("tilt")
    nx.press_buttons(controller_idx, [nxbt.Buttons.A], down=0.2)
    print("done turning right")

def turn_left(nx, controller_idx):
    print("Turn left")
    nx.tilt_stick(controller_idx, Sticks.LEFT_STICK, -100, 0, tilted=0.5)
    print("tilt")
    nx.press_buttons(controller_idx, [nxbt.Buttons.A], down=0.2)
    print("done turning left")

def speed_up(nx, controller_idx):
    print("Speed up")
    nx.press_buttons(controller_idx, [nxbt.Buttons.A], down=0.2)

#TODO fix these
def slow_down(nx, controller_idx):
    nx.press_buttons(controller_idx, [nxbt.Buttons.B], down=0.2)
    print("Slow down")

def power_up(nx, controller_idx):
    print("Power up!")