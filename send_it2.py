#controls and connecting to switch controller, send controls
#author @ taylor tam
import nxbt 
from pathlib import Path
from nxbt import Buttons
from nxbt import Sticks
import time
import threading

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

## attempting threading v2
def continuous_press(nx, controller_idx, button):
    while True:
        nx.press_buttons(controller_idx, [button], down=0.1)
        time.sleep(0.05)


##### CONNECT & START #####
#connects and runs a start sequence
def connect_controller():
    first_connect = False
    print("Connecting...")
    nx = nxbt.Nxbt(debug=False)
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
    print("going right")
    input_packet = nx.create_input_packet()
    input_packet["A"] = True
    input_packet["L_STICK"]["X_VALUE"] = 70
    input_packet["L_STICK"]["Y_VALUE"] = 0
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)

def turn_left(nx, controller_idx):
    input_packet = nx.create_input_packet()
    input_packet["A"] = True
    input_packet["L_STICK"]["X_VALUE"] = -70
    input_packet["L_STICK"]["Y_VALUE"] = 0
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)

def speed_up(nx, controller_idx, first_press):
    input_packet = nx.create_input_packet()
    input_packet["A"] = True
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)

def slow_down(nx, controller_idx):
    input_packet = nx.create_input_packet()
    input_packet["A"] = False
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)

def power_up(nx, controller_idx):
    input_packet = nx.create_input_packet()
    input_packet["ZL"] = True
    input_packet["A"] = True
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)
    print('power up!')

def drift(nx, controller_idx):
    input_packet = nx.create_input_packet()
    input_packet["ZR"] = True
    input_packet["A"] = True
    nx.set_controller_input(controller_idx, input_packet)
    time.sleep(0.1)
    print("drift")
