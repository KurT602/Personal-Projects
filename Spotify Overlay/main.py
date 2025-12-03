import win32gui, win32con, win32process, psutil, ctypes, random, os
from ctypes import wintypes, windll, byref, sizeof, c_int
import customtkinter as ctk
from time import sleep
from PIL import Image
from settings import *

#region SIMULATE KEYBOARD INPUT
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
MAPVK_VK_TO_VSC = 0
wintypes.ULONG_PTR = wintypes.WPARAM
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))
class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))
    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))
class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))
LPINPUT = ctypes.POINTER(INPUT)
#endregion

#region KEY-PRESS
def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, byref(x), sizeof(x))
def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, byref(x), sizeof(x))
def TapKey(code):
    PressKey(code)
    sleep(0.2)
    ReleaseKey(code)
#endregion

#region SPOTIFY WINDOW INFO
def GetPIDByName(procname:str):
    pid = []
    for proc in psutil.process_iter():
        if procname in proc.name():
            pid.append(proc.pid)
    if pid != None:
        return pid
def GetHwndByPID(pid):
    hwnds = []
    path = ""
    def callback(hwnd, hwnds):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

        if found_pid == pid:
            global path
            path = psutil.Process(found_pid).exe()
            hwnds.append(hwnd)
        return True
    
    win32gui.EnumWindows(callback, hwnds)
    return hwnds, path
def GetSpotifyHwnd():
    pids = GetPIDByName("Spotify.exe")
    for i in pids:
        h,_ = GetHwndByPID(i)
        for hwnd in h:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                return hwnd
def GetWindowTitle(procname):
    pids = GetPIDByName(procname)
    for i in pids:
        hwnds,_ = GetHwndByPID(i)
        for hwnd in hwnds:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                return win32gui.GetWindowText(hwnd)
def GetSongInfo():
    windowtitle = GetWindowTitle("Spotify.exe")
    
    if windowtitle: 
        if windowtitle != "Advertisement":
            if windowtitle != "Spotify Free":
                split = windowtitle.split(" - ", 1)
                artist = split[0]
                title =  split[1]
            else:
                pass
        else:
            return windowtitle, ""
    else:
        return "☺",""

    try:
        return title, artist
    except:
        return None, None
#endregion

#region FOCUS SPOTIFY WINDOW
def bring_window_to_foreground(HWND):
        win32gui.ShowWindow(HWND, win32con.SW_RESTORE)
        win32gui.SetWindowPos(HWND, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE + win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(HWND, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE + win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(HWND, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_SHOWWINDOW + win32con.SWP_NOMOVE + win32con.SWP_NOSIZE)
def ShowSpotify():
        try:
            bring_window_to_foreground(GetSpotifyHwnd())
            user32.ShowWindow(GetSpotifyHwnd(),3)
        except:
            print("no spo")
            # os.startfile()
#endregion

# MISC
def RandomColor():
    lst = []
    lst.append(RED)
    lst.append(ORANGE)
    lst.append(YELLOW)
    lst.append(GREEN)
    lst.append(GREEN_LIGHT)
    lst.append(BLUE)
    lst.append(BLUE_DARK)
    lst.append(PURPLE)
    lst.append(PINK)
    newcolor = random.choice(lst)
    return newcolor

# MAIN APP WINDOW
class App(ctk.CTk):
    def __init__(self):

        # window
        super().__init__(fg_color=FG_COLOR)
        self.title("")
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.95)
        self.attributes("-toolwindow", True)
        self.iconbitmap(ICO_EMPTY_LIGHT)
        self.geometry(f"300x100+{str(self.winfo_screenwidth()-10)}+{str(self.winfo_screenheight()-14)}")
        self.resizable(False, False)
        self.change_title_bar_color()

        # layout
        self.rowconfigure   (0,weight=1)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)

        # data
        self.title_str =  ctk.StringVar(value="Title")
        self.artist_str = ctk.StringVar(value="Artist")
        self.is_playing = ctk.BooleanVar(value=True)
        self.update_info()

        # widgets
        Caffeine(self)
        Dopamine(self, self.title_str, self.artist_str, self.is_playing)

        # main looping
        self.sched()
        self.mainloop()
    
    #region FUNCTIONS
    # song info listener and updater
    def update_info(self):
        songtitle, artist = GetSongInfo()
        focused_hwnd = win32gui.GetForegroundWindow()
        # print(f"SONG: {songtitle} BY ARTIST: {artist}")

        if focused_hwnd == GetSpotifyHwnd():
            self.withdraw()
        else:
            if self.state() == "withdrawn":
                self.deiconify()

        if songtitle != None:
            self.title_str.set (songtitle)
            self.artist_str.set(artist)
            self.is_playing.set(value=True)
        else:
            self.is_playing.set(value=False)
    def sched(self):
        # try to loop the listener
        try:
            self.update_info()
            self.after(3000, self.sched)
        except:
            pass
    def change_title_bar_color(self):
        try:
            HWND = user32.GetParent(self.winfo_id())
            DWMWA_ATTRIBUTE = 35
            COLOR = TITLE_HEX_COLOR
            windll.dwmapi.DwmSetWindowAttribute(HWND, DWMWA_ATTRIBUTE, byref(c_int(COLOR)), sizeof(c_int))
        except:
            pass
    #endregion

#region FRAMES
# Left
class Dopamine(ctk.CTkFrame):
    def __init__(self, parent, t_s, a_s, is_playing):
        super().__init__(parent, fg_color=TRANSPARENT)
        self.grid(column=0,row=0,sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure   (0, weight=10)
        self.rowconfigure   (1, weight=1)

        MainInputs  (self, is_playing)
        SongInfoText(self, t_s, a_s)

class Caffeine(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=TRANSPARENT, width=30)
        self.grid(column=1,row=0,sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure   (0, weight=1)
        self.rowconfigure   (1, weight=1)

        spotifybutton = ctk.CTkButton(
            master=self,
            text="olieyfgkuasfgkucyfekc",
            corner_radius=10,
            fg_color=GREEN,
            hover_color=FG_COLOR,
            text_color=WHITE,
            command=lambda:ShowSpotify()
        )
        spotifybutton.grid(column=0, row=0, sticky="nsew")
#endregion

#region WIDGETS
# song text
class SongInfoText(ctk.CTkFrame):
    def __init__(self, parent, title_str, artist_str):
        super().__init__(parent, fg_color=FG_COLOR_SECONDARY)
        self.grid(row=0, sticky="nsew", padx=4, pady=1)

        # layout
        self.rowconfigure   (0,     weight=2, uniform="c")
        self.rowconfigure   ((1,2), weight=1, uniform="c")
        self.columnconfigure(0,     weight=1, uniform="c")

        # text widgets
        title_text =  ctk.CTkLabel(
            master=self,
            corner_radius=5,
            text="songtitle",
            text_color=TEXT_COLOR,
            font=(FONT, SONG_TEXT_SIZE),
            textvariable=title_str,
            anchor="w"
            # fg_color=RandomColor()
        )
        artist_text = ctk.CTkLabel(
            master=self,
            corner_radius=5,
            text="artist",
            text_color=TEXT_COLOR_SECONDARY,
            font=(FONT, ARTIST_TEXT_SIZE),
            textvariable=artist_str,
            # fg_color=RandomColor()
        )
        title_text.grid (row=0, column=0, sticky="sw", padx=2, pady=2)
        artist_text.grid(row=1, column=0, sticky="nw", padx=2, pady=0)

# action bar
class MainInputs(ctk.CTkFrame):
    def __init__(self, parent, is_playing):
        super().__init__(parent, fg_color=FG_COLOR)
        self.grid(row=1, sticky="nsew", padx=2, pady=2)

        # button layout
        self.rowconfigure(0, weight=1, uniform="b")
        self.columnconfigure((0,1,2), weight=1, uniform="b")

        # icon images
        img_play  = ctk.CTkImage(Image.open(ICO_PLAY_DARK).resize((50,50)))
        img_pause = ctk.CTkImage(Image.open(ICO_PAUSE_DARK).resize((50,50)))
        img_next  = ctk.CTkImage(Image.open(ICO_NEXT_DARK).resize((50,50)))
        img_prev  = ctk.CTkImage(Image.open(ICO_PREV_DARK).resize((50,50)))
        
        # buttons
        pauseplaybutton = ctk.CTkButton(
            master=self, 
            text="",
            image=img_play,
            corner_radius=10,
            # width=10,
            # height=40,
            fg_color=TRANSPARENT,
            # fg_color=RandomColor(),
            hover_color=FG_COLOR,
            text_color=BLACK,
            command=lambda:[TapKey(CODE_PLAYPAUSE), change_button_icon()]
            )
        nextbutton = ctk.CTkButton(
            master=self, 
            text="",
            image=img_next,
            corner_radius=10,
            # width=10,
            # height=40,
            fg_color=TRANSPARENT,
            # fg_color=RandomColor(),
            hover_color=FG_COLOR,
            text_color=BLACK,
            command=lambda:TapKey(CODE_NEXT)
            )
        prevbutton = ctk.CTkButton(
            master=self, 
            text="",
            image=img_prev,
            corner_radius=10,
            # width=10,
            # height=40,
            fg_color=TRANSPARENT,
            # fg_color=RandomColor(),
            hover_color=FG_COLOR,
            text_color=BLACK,
            command=lambda:TapKey(CODE_PREV)
            )
        pauseplaybutton.grid(row=0, column=1, sticky="nsew")
        nextbutton.grid     (row=0, column=2, sticky="nsew")
        prevbutton.grid     (row=0, column=0, sticky="nsew")

        def change_button_icon(*args):
            if is_playing.get() != True:
                pauseplaybutton.configure(image=img_pause)
            else:
                pauseplaybutton.configure(image=img_play)
#endregion

if __name__ == "__main__":
    App()