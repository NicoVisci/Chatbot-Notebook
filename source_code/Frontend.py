from tkinter import Tk, Label, Text, Scrollbar, Entry, Button
from tkinter import DISABLED, VERTICAL, NORMAL, END

from events import Events

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"    # Dark blue
TEXT_COLOR = "#EAECEE"
INPUT_BG_COLOR = "#2C3E50"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"



class FrontendImplementation:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

        self.events = Events()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # Head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # Tiny divider
        # line = Label(self.window, width=450, bg=BG_GRAY)
        # line.place(relwidth=1, relheight=0.012, rely=0.07)

        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relwidth=1, relheight=0.745, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        self.text_widget.tag_configure('italic', font=(FONT, 10, 'italic'))

        # Scroll bar
        scrollbar = Scrollbar(self.text_widget, orient=VERTICAL)
        scrollbar.place(relheight=1, relx=1)
        # Command to let the scrollbar change the y position of the text widget
        scrollbar.configure(command=self.text_widget.yview)

        # Bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # Chatting box
        self.msg_entry = Entry(bottom_label, bg=INPUT_BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.75, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # Send button
        send_button = Button(bottom_label, text="Send", width=20, bg=BG_GRAY, fg=TEXT_COLOR, font=FONT_BOLD,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relwidth=0.22, relheight=0.06, relx=0.77, rely=0.008)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self.insert_message(msg, "You")

        self.events.on_data(msg)

    def insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        msg_displayed = f"{sender}: {msg}" + "\n"
        self.text_widget.configure(state=NORMAL)
        if sender.startswith('System'):
            self.text_widget.insert(END, msg_displayed, 'italic')
        else:
            self.text_widget.insert(END, msg_displayed)
        self.text_widget.configure(state=DISABLED)

    def message_handler(self, message):
        self.insert_message(message['message'], message['sender'])