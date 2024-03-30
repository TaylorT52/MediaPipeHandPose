import time
import os
import time
import psutil
from collections import deque
import multiprocessing

# from blessed import Terminal
# from nxbt import Nxbt, PRO_CONTROLLER

############ LOADING STUFF ############ 
class LoadingSpinner():
    SPINNER_CHARS = ['■ □ □ □', '□ ■ □ □', '□ □ ■ □', '□ □ □ ■', '□ □ □ ■', '□ □ ■ □', '□ ■ □ □', '■ □ □ □']  # noqa
    def __init__(self):
        self.creation_time = time.perf_counter()
        self.last_update_time = self.creation_time
        self.current_char_index = 0

    def get_spinner_char(self):
        current_time = time.perf_counter()
        delta = current_time - self.last_update_time
        if delta > 0.07:
            self.last_update_time = current_time

            if self.current_char_index == 7:
                self.current_char_index = 0
            else:
                self.current_char_index += 1
        return self.SPINNER_CHARS[self.current_char_index]

############ GUI STUFF!!! ############ 
class ControllerTUI():
    CONTROLS = {
        "ZL": "◿□□□□",
        "L": "◿□□□□",
        "ZR": "□□□□◺",
        "R": "□□□□◺",
        "LS_UP": ".─.",
        "LS_LEFT": "(",
        "LS_RIGHT": ")",
        "LS_DOWN": "`─'",
        "RS_UP": ".─.",
        "RS_LEFT": "(",
        "RS_RIGHT": ")",
        "RS_DOWN": "`─'",
        "DPAD_UP": "△",
        "DPAD_LEFT": "◁",
        "DPAD_RIGHT": "▷",
        "DPAD_DOWN": "▽",
        "MINUS": "◎",
        "PLUS": "◎",
        "HOME": "□",
        "CAPTURE": "□",
        "A": "○",
        "B": "○",
        "X": "○",
        "Y": "○",
    }

    def __init__(self, term):
        self.term = term
        self.DEFAULT_CONTROLS = self.CONTROLS.copy()
        self.CONTROL_RELEASE_TIMERS = self.CONTROLS.copy()
        for control in self.CONTROL_RELEASE_TIMERS.keys():
            self.CONTROL_RELEASE_TIMERS[control] = False

        self.auto_keypress_deactivation = True
        self.remote_connection = False

    def toggle_auto_keypress_deactivation(self, toggle):
        self.auto_keypress_deactivation = toggle

    def set_remote_connection_status(self, status):
        self.remote_connection = status

    def activate_control(self, key, activated_text=None):
        if activated_text:
            self.CONTROLS[key] = activated_text
        else:
            self.CONTROLS[key] = self.term.bold_black_on_white(self.CONTROLS[key])

        if self.auto_keypress_deactivation:
            self.CONTROL_RELEASE_TIMERS[key] = time.perf_counter()

    def deactivate_control(self, key):
        self.CONTROLS[key] = self.DEFAULT_CONTROLS[key]

    def render_controller(self):
        if self.auto_keypress_deactivation:
            # Release any overdue timers
            for control in self.CONTROL_RELEASE_TIMERS.keys():
                pressed_time = self.CONTROL_RELEASE_TIMERS[control]
                current_time = time.perf_counter()
                if pressed_time is not False and current_time - pressed_time > 0.25:
                    self.deactivate_control(control)
        ZL = self.CONTROLS['ZL']
        L = self.CONTROLS['L']
        ZR = self.CONTROLS['ZR']
        R = self.CONTROLS['R']
        LU = self.CONTROLS['LS_UP']
        LL = self.CONTROLS['LS_LEFT']
        LR = self.CONTROLS['LS_RIGHT']
        LD = self.CONTROLS['LS_DOWN']
        RU = self.CONTROLS['RS_UP']
        RL = self.CONTROLS['RS_LEFT']
        RR = self.CONTROLS['RS_RIGHT']
        RD = self.CONTROLS['RS_DOWN']
        DU = self.CONTROLS['DPAD_UP']
        DL = self.CONTROLS['DPAD_LEFT']
        DR = self.CONTROLS['DPAD_RIGHT']
        DD = self.CONTROLS['DPAD_DOWN']
        MN = self.CONTROLS['MINUS']
        PL = self.CONTROLS['PLUS']
        HM = self.CONTROLS['HOME']
        CP = self.CONTROLS['CAPTURE']
        A = self.CONTROLS['A']
        B = self.CONTROLS['B']
        X = self.CONTROLS['X']
        Y = self.CONTROLS['Y']

        if self.remote_connection:
            lr_press = "L + R - - - - - - - - -▷ E"
        else:
            lr_press = "                          "

        print(self.term.home + self.term.move_y((self.term.height // 2) - 9))
        print(self.term.center(f"      {ZL}        {ZR}                                    "))
        print(self.term.center(f"    ─{L}──────────{R}─      ┌─────────────┬────────────┐"))
        print(self.term.center("  ╱                        ╲    │  Controls   │    Keys    │"))
        print(self.term.center(f" ╱   {LU}   {MN}       {PL}   {X}    ╲   └─────────────┴────────────┘"))  # noqa
        print(self.term.center(f"│   {LL}   {LR}    {CP}   {HM}   {Y}   {A}   │   Left Stick ─ ─ ─ ▷ W/A/S/D "))  # noqa
        print(self.term.center(f"│    {LD}               {B}     │   DPad ─ ─ ─ ─ ─ ─ ▷ G/V/B/N "))
        print(self.term.center(f"│        {DU}         {RU}       │   Capture/Home ─ ─ ─ ─ ▷ [/] "))
        print(self.term.center(f"│╲     {DL} □ {DR}      {RL}   {RR}     ╱│   +/- ─ ─ ─ ─ ─ ─ ─ ─ ─▷ 6/7 "))  # noqa
        print(self.term.center(f"│░░╲     {DD}         {RD}    ╱░░│   X/Y/B/A ─ ─ ─ ─ ─▷ J/I/K/L "))
        print(self.term.center("│░░░░╲ ──────────────── ╱░░░░│   L/ZL ─ ─ ─ ─ ─ ─ ─ ─ ▷ 1/2 "))
        print(self.term.center("│░░░░╱                  ╲░░░░│   R/ZR ─ ─ ─ ─ ─ ─ ─ ─ ▷ 8/9 "))
        print(self.term.center("│░░╱                      ╲░░│   Right Stick - - - ▷ Arrows "))
        print(self.term.center(f"│╱                          ╲│   {lr_press} "))

############ INPUT STUFF ############ 
class InputTUI():
    KEYMAP = {
        # Left Stick Mapping
        "w": {
            "control": "LS_UP",
            "stick_data": {
                "stick_name": "L_STICK",
                "x": "+000",
                "y": "+100"
            }
        },
        "a": {
            "control": "LS_LEFT",
            "stick_data": {
                "stick_name": "L_STICK",
                "x": "-100",
                "y": "+000"
            }
        },
        "d": {
            "control": "LS_RIGHT",
            "stick_data": {
                "stick_name": "L_STICK",
                "x": "+100",
                "y": "+000"
            }
        },
        "s": {
            "control": "LS_DOWN",
            "stick_data": {
                "stick_name": "L_STICK",
                "x": "+000",
                "y": "-100"
            }
        },

        # Right Stick Mapping
        "KEY_UP": {
            "control": "RS_UP",
            "stick_data": {
                "stick_name": "R_STICK",
                "x": "+000",
                "y": "+100"
            }
        },
        "KEY_LEFT": {
            "control": "RS_LEFT",
            "stick_data": {
                "stick_name": "R_STICK",
                "x": "-100",
                "y": "+000"
            }
        },
        "KEY_RIGHT": {
            "control": "RS_RIGHT",
            "stick_data": {
                "stick_name": "R_STICK",
                "x": "+100",
                "y": "+000"
            }
        },
        "KEY_DOWN": {
            "control": "RS_DOWN",
            "stick_data": {
                "stick_name": "R_STICK",
                "x": "+000",
                "y": "-100"
            }
        },

        # Dpad Mapping
        "g": "DPAD_UP",
        "v": "DPAD_LEFT",
        "n": "DPAD_RIGHT",
        "b": "DPAD_DOWN",

        # Button Mapping
        "6": "MINUS",
        "7": "PLUS",
        "[": "CAPTURE",
        "]": "HOME",
        "i": "X",
        "j": "Y",
        "l": "A",
        "k": "B",

        # Triggers
        "1": "L",
        "2": "ZL",
        "8": "R",
        "9": "ZR",
    }

    def __init__(self, reconnect_target=None, debug=False, logfile=False, force_remote=False):
        self.reconnect_target = reconnect_target
        self.term = Terminal()
        if force_remote:
            self.remote_connection = True
        else:
            self.remote_connection = self.detect_remote_connection()
        self.controller = ControllerTUI(self.term)

        # Check if direct connection will fail
        if not self.remote_connection:
            try:
                from pynput import keyboard
            except ImportError as e:
                print("Unable to import pynput for direct input.")
                print("If you're accessing NXBT over a remote shell, ", end="")
                print("please use the 'remote_tui' option instead of 'tui'.")
                print("The original pynput import is displayed below:\n")
                print(e)
                exit(1)
        self.debug = debug
        self.logfile = logfile

    def detect_remote_connection(self):
        remote_connection = False
        remote_process_names = ['sshd', 'mosh-server']
        ppid = os.getppid()
        while ppid > 0:
            process = psutil.Process(ppid)
            if process.name() in remote_process_names:
                remote_connection = True
                break
            ppid = process.ppid()

        return remote_connection

    