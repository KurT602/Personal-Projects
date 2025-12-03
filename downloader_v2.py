from pytubefix import YouTube#, Search
from pytubefix.cli import on_progress
import customtkinter as ctk
from customtkinter import filedialog
import os.path, threading, urllib.request, io
from PIL import Image, ImageTk
from datetime import datetime

class Main(ctk.CTk):
    def __init__(self, fg_color = None, **kwargs):
        super().__init__(fg_color, **kwargs)
        self.title("YouTube Video Downloader")
        self.geometry("700x600")

        self.video_links = []

        self.output_path = ""

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tabview.add("Log")
        self.tabview.add("Main")
        self.tabview.add("Search")

        self.main_tab = self.tabview.tab("Main")
        self.search_tab = self.tabview.tab("Search")
        self.log_tab = self.tabview.tab("Log")

        self.tabview.set("Main")

        self.create_main_window()
        self.create_log_window()
        self.create_search_window()

        self.mainloop()

    def create_main_window(self):
        main_frame = ctk.CTkFrame(self.main_tab)
        main_frame.pack(fill="both", padx=5, pady=(8,5))

        # Choose Path
        save_frame = ctk.CTkFrame(main_frame)
        save_frame.pack(fill="x",ipadx=10,ipady=10)

        self.path_label = ctk.CTkLabel(save_frame,text="Output Path:",anchor="w",height=10)
        self.path_label.pack(fill="x",padx=8,pady=(10,0))

        self.path_entry = ctk.CTkEntry(save_frame,placeholder_text="Enter export path or browse folder")
        self.path_entry.pack(fill="x",expand=True, side="left",padx=(10,0))

        browse_btn = ctk.CTkButton(save_frame,width=50,text="Browse", command=lambda: (self.browse(), self.path_label.configure(text_color="#ffffff", text="Output:")))
        browse_btn.pack(side="right", padx=(0,10))

        # Choose output type
        ext_frame = ctk.CTkFrame(main_frame)
        ext_frame.pack(fill="x",ipadx=10,ipady=10)

        ext_label = ctk.CTkLabel(ext_frame,text="Output Type:",height=10)
        ext_label.pack(padx=8, side="left", anchor="center")

        self.file_ext_dropdown = ctk.CTkComboBox(ext_frame,values=["Video","Audio"])
        self.file_ext_dropdown.pack(side="left")

        # Add Video Link
        url_frame = ctk.CTkFrame(main_frame)
        url_frame.pack(fill="x",ipadx=10,ipady=10)

        url_label = ctk.CTkLabel(url_frame,text="Video Link/URL:",anchor="w",height=10)
        url_label.pack(fill="x",padx=8,pady=(10,0))

        self.url_entry = ctk.CTkEntry(url_frame,placeholder_text="Enter YouTube video link or URL")
        self.url_entry.pack(fill="x",expand=True, side="left",padx=(10,0))

        add_btn = ctk.CTkButton(url_frame,width=50,text="Add link", command=lambda: self.add_video_link(self.url_entry.get()))
        add_btn.pack(side="right", padx=(0,10))

        # Download list of videos
        videos_frame = ctk.CTkFrame(main_frame)
        videos_frame.pack(fill="x",ipadx=10,ipady=10)

        videos_label = ctk.CTkLabel(videos_frame,text="Videos:",height=10)
        videos_label.pack(pady=8, fill="x", anchor="center")

        self.videos_container = ctk.CTkScrollableFrame(videos_frame)
        self.videos_container.pack(fill="x",ipadx=10,ipady=10)

        download_btn = ctk.CTkButton(videos_frame,width=50,text="Download all", command=self.download_all)
        download_btn.pack(pady=10)

    def create_log_window(self):
        main_frame = ctk.CTkFrame(self.log_tab)
        main_frame.pack(fill="both", padx=5, pady=(8,5))

        self.log_box = ctk.CTkTextbox(main_frame, width=480, height=200)
        self.log_box.pack(padx=10, pady=10)
        self.log_box.configure(state="disabled")  # Start as read-only

        self.progress_bar = ctk.CTkProgressBar(main_frame, indeterminate_speed=0.067, determinate_speed=0.067)
        self.progress_bar.pack(fill="x", padx=5)

    def create_search_window(self):
        main_frame = ctk.CTkFrame(self.search_tab)
        main_frame.pack(fill="both", padx=5, pady=(8,5))

        self.path_label = ctk.CTkLabel(main_frame,text="Doesn't work yet sorry :(",anchor="w",height=10)
        self.path_label.pack(fill="x",padx=8,pady=(10,0))

    def browse(self):
        path = filedialog.askdirectory()
        self.output_path = path

        self.path_entry.insert(0,path)

        print(self.output_path)

    def log_message(self,message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")  # HH:MM:SS format
        self.log_box.configure(state="normal")  # Enable editing
        self.log_box.insert("end", f"{timestamp} {message}\n")
        self.log_box.configure(state="disabled")  # Make read-only again
        self.log_box.see("end")  # Auto-scroll to latest message
        self.log_box.update_idletasks()

    def seconds_to_mm_ss(self,total_seconds):
        minutes, seconds = divmod(total_seconds, 60)
        return f"{int(minutes):02d}:{int(seconds):02d}"

    def add_video_link(self,url):
        if url == "":
            return
        
        try:
            yt = YouTube(url=url)
        except Exception as e:
            raise ValueError(f"error: {e}")

        new_widget = ctk.CTkFrame(self.videos_container, fg_color="#676767")
        new_widget.pack(padx=5,pady=5,fill="x")

        # Title
        video_title = ctk.CTkLabel(new_widget, text=yt.title, fg_color="transparent")
        video_title.pack(side="left", padx=10)
        
        # Remove button
        remove_btn = ctk.CTkButton(new_widget,text="X", fg_color="#aa0000",width=25, command=lambda: (self.video_links.remove(url),new_widget.destroy()))
        remove_btn.pack(side="right",padx=5,pady=5, before=video_title)

        # Video length
        video_length = ctk.CTkLabel(new_widget, text=self.seconds_to_mm_ss(yt.length), fg_color="transparent")
        video_length.pack(side="right", padx=10, before=video_title)

        self.video_links.append(url)
        self.url_entry.delete(0, ctk.END)

    def download_all(self):
        self.path_label.configure(text="Output:", text_color="#ffffff")
        def loop_through():
            for i,link in enumerate(self.video_links):
                self.tabview.set("Log")

                self.progress_bar.set(i/len(self.video_links))
                print((i/len(self.video_links)))
                self.progress_bar.start()

                print(link)
                self.log_message(f"Downloading {link}")
                self.download_video(link,self.path_entry.get(),self.file_ext_dropdown.get())
            
            self.progress_bar.set(1)
            self.progress_bar.stop()

        loopthread = threading.Thread(target=loop_through)
        loopthread.start()

    def download_video(self, url, save_path, fileExt):
        if url == "":
            self.log_message("No video link or url found")
            return
        try:
            yt = YouTube(url,on_progress_callback=on_progress)
        except Exception as e:
            self.progress_bar.stop()
            self.progress_bar.set(0)
            raise ValueError(f"error: {e}")
        
        if os.path.exists(save_path):
            pass
        else:
            self.log_message(f"Path {save_path} not found.")
            self.progress_bar.stop()
            self.progress_bar.set(0)
            self.tabview.set("Main")
            self.path_entry.focus()
            self.path_label.configure(text_color="#ff0000", text="Output*:")
            raise LookupError(f"Path {save_path} not found.")

        if fileExt == "Audio":
            ys = yt.streams.get_audio_only()
            ys.download(output_path=save_path)
            self.log_message(f"Downloaded '{yt.title}'")
        elif fileExt == "Video":
            ys = yt.streams.get_highest_resolution()
            ys.download(output_path=save_path)
            self.log_message(f"Downloaded '{yt.title}'")
        else:
            raise ValueError("Invalid fileExt value.")
        
    def thumbnail(self, video):
        try:
            with urllib.request.urlopen(video) as u:
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