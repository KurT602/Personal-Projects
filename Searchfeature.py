from pytubefix import YouTube, Search, Channel
import customtkinter as ctk
from customtkinter import filedialog
import urllib.request, io
from PIL import Image, ImageTk
from time import gmtime, strftime


def download_video(url, save_path, fileExt : str):
    try:
        yt = YouTube(url)
    except Exception as e:
        print(f"error: {e}")

    if fileExt.lower() == "mp3":
        ys = yt.streams.get_audio_only()
        ys.download(mp3=True,output_path=save_path)
    elif fileExt.lower() == "mp4":
        ys = yt.streams.get_highest_resolution()
        ys.download(output_path=save_path)
    else:
        raise ValueError("Invalid fileExt value. May only be 'mp4' or 'mp3'")

def length_format(seconds):
    if seconds > 3600:
        return strftime("%H:%M:%S", gmtime(seconds))
    else:
        return strftime("%M:%S", gmtime(seconds))

class Main(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("A window")
        self.geometry("250x250")
        # self.columnconfigure(0,weight=1)
        # self.rowconfigure((0,1),weight=2)
        self.all_videos = []

        button = ctk.CTkButton(self,fg_color="#665544",text="Open Search",command=lambda:self.open_search_window())
        button.pack()

        # download_button = ctk.CTkButton(self,fg_color="#665544",text="dolwoand",command=lambda:download_video())
        # button.pack()

        self.mainloop()

    def open_search_window(self):
        search_win = SearchWindow(self)

    # def download_list(self):
    #     save_path = filedialog
    #     for video in self.all_videos:
    #         download_video(video, )

class SearchWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("600x700")
        self.title("Search window")
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)

        frame = ctk.CTkFrame(self,fg_color="transparent")
        frame.grid(row=0,column=0,sticky="nsew")

        infowin = Results(frame)
        searchbar = Searchbar(frame, infowin, parent)      

class Searchbar(ctk.CTkFrame):
    def __init__(self, parent, infowin, main):
        super().__init__(parent,width=300,height=50,corner_radius=0,fg_color="transparent")
        # self.grid(row=0,column=0,ipadx=3,sticky="nsew")
        self.pack(fill="x",side="top",pady=(6,2))
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.search_prompt = ctk.StringVar()

        search_entry_frame = ctk.CTkFrame(self,fg_color="transparent")
        search_entry_frame.grid(row=0,column=0,ipadx=3)

        label = ctk.CTkLabel(search_entry_frame, text="Search: ")
        self.search_entry = ctk.CTkEntry(search_entry_frame, corner_radius=20,placeholder_text="Search video...")
        self.search_entry.grid(row=0,column=1)
        label.grid(row=0,column=0)
        search_button = ctk.CTkButton(search_entry_frame,text="🔍",width=30,command=lambda:[infowin.showstats(),self.get_search(infowin)])
        search_button.grid(row=0,column=2,sticky="w")

        finish_button = ctk.CTkButton(search_entry_frame,text="Finished",width=30,fg_color="#446644",command=lambda:infowin.finish_search(main))
        finish_button.grid(row=0,column=3,sticky="w")
        yah_button = ctk.CTkButton(search_entry_frame,text="Download video",width=30,command=lambda:infowin.download("mp4"))
        yah_button.grid(row=0,column=3,sticky="w")
        yeh_button = ctk.CTkButton(search_entry_frame,text="Download audio",width=30,command=lambda:infowin.download("mp3"))
        yeh_button.grid(row=0,column=4,sticky="w")

    def get_search(self,infowin):
        self.search_prompt = self.search_entry.get()
        infowin.search(prompt=self.search_prompt)

class Results(ctk.CTkFrame):
    def __init__(self,parent):
        super().__init__(parent, fg_color="transparent")
        # self.grid(row=1,column=0,sticky="nsew")
        self.pack(side="bottom",fill="both",expand=True)

        self.selected_vids = []
        self.loaded_vids = []

        self.xtraInfo = ctk.CTkFrame(self,fg_color="transparent",height=20)
        self.xtraText = ctk.CTkLabel(self.xtraInfo, text_color="#444444")
        self.xtraText.pack()

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both",expand=True,padx=10,pady=3)

    def finish_search(self,main):
        print("GYATT")
        for video in self.selected_vids:
            main.all_videos.append(video)
        print(main.all_videos)
        self.master.master.destroy()

    def load_ui(self):
        for video in self.results.videos[len(self.loaded_vids):][:5]:
            video_frame = ctk.CTkFrame(self.scroll_frame,corner_radius=10,fg_color="transparent")
            video_frame.pack(padx=5,pady=5,expand=True,fill="x")
            video_frame.bind("<Button-1>",self.on_frame_click)

            thumbnail = self.thumbnail(video)
            awd = ctk.CTkLabel(video_frame,text="", image=thumbnail,corner_radius=10)
            awd.pack(side="left",padx=3,pady=3)

            text_frame = ctk.CTkFrame(video_frame,fg_color="transparent")
            text_frame.pack(fill="x", expand=True, padx=4,side="top",anchor="n",pady=10)

            title_text = ctk.CTkLabel(text_frame,text=video.title,justify="left",anchor="w",fg_color="transparent",font=("Century Gothic Bold", 22))
            title_text.grid(row=0,column=0)

            channel_name = ctk.CTkLabel(text_frame,text=Channel(video.channel_url).channel_name,fg_color="transparent",text_color="#888888",justify="left",anchor="w",font=("Century Gothic Bold", 19))
            channel_name.grid(row=1,column=0,sticky="nsew")

            duration = ctk.CTkLabel(text_frame,text=length_format(video.length),fg_color="transparent",text_color="#444444",justify="left",anchor="w",font=("Century Gothic Bold", 16))
            duration.grid(row=2,column=0,sticky="nsew")

            self.loaded_vids.append(video)
        
        self.xtraText.configure(text=f"{len(self.results.videos)} results       {len(self.loaded_vids)} shown")
        
        if len(self.results.videos) == len(self.loaded_vids): self.results.get_next_results()

    def download(self, fileExt : str):
        path = filedialog.askdirectory()
        for video in self.selected_vids:
            download_video(video,path,fileExt)
            print(f"downloaded {YouTube(video).title} ({length_format(YouTube(video).length)})")
        print(f"finished downloading {len(self.selected_vids)} videos")

    def search(self,prompt : str):
        try:
            self.results = Search(prompt)
            print(f"{len(self.results.videos)} results found")
            self.xtraText.configure(text=f"{len(self.results.videos)} results       {len(self.loaded_vids)} shown")
        except Exception as e:
            print(e)
            return

        for w in self.scroll_frame.winfo_children():
            w.destroy()
        
        self.load_ui()
        self.load_button = ctk.CTkButton(self.scroll_frame,text="Load more",corner_radius=10,fg_color="#446644",command=lambda:[self.load_ui(),self.load_button_repos()])
        self.load_button.pack()

    def load_button_repos(self):
        self.load_button.destroy()
        self.load_button = ctk.CTkButton(self.scroll_frame,text="Load more",corner_radius=10,fg_color="#446644",command=lambda:[self.load_ui(),self.load_button_repos()])
        self.load_button.pack()
    
    def on_frame_click(self, event):
        frame = event.widget.master
        vid_link = self.loaded_vids[self.scroll_frame.winfo_children().index(frame)].watch_url
        if vid_link in self.selected_vids:
            print("is in already")
            frame.configure(fg_color="transparent")
            self.selected_vids.remove(vid_link)
        else:
            print(f"clickedd {frame}")
            frame.configure(fg_color="#222222")
            self.selected_vids.append(vid_link)
        print(self.selected_vids)

    def showstats(self):
        self.xtraInfo.pack(fill="x",padx=10,pady=3)
        self.scroll_frame.pack(after=self.xtraInfo)

    def thumbnail(self, video):
        try:
            with urllib.request.urlopen(video.thumbnail_url) as u:
                raw = u.read()
        except Exception as e:
            print(f"error: {e}")

        try:
            image = Image.open(io.BytesIO(raw)).resize((240,170))
            # self.photo = ImageTk.PhotoImage(image)
            return ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"error2: {e}")
        
if __name__ == "__main__":
    Main()