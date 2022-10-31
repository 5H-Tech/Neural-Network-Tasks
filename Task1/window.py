import tkinter
from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
import pandas as pd

class Gui:
    def btn_clicked(self):
        global class1
        global class2
        global feature1
        global feature2
        global learing_rate
        global epochs
        global bias

        class1 = self.entry0.get()
        class2 = self.entry2.get()
        feature1 = self.entry1.get()
        feature2 = self.entry3.get()
        learing_rate = self.entry7.get()
        epochs = self.entry8.get()
        bias = var1.get()
        try:
            float(self.entry7.get()) or int(self.entry7.get())
            float(self.entry8.get()) or int(self.entry8.get())
        except ValueError:
            messagebox.showinfo("Error", "Please, Enter the valid number")
        print("Button Clicked")

    def __init__(self, master=None):
        global species
        global feature
        df = pd.read_csv("Datasets/penguins.csv")
        species = list(df["species"].unique())
        feature = list(df.columns[1:])
        self.window = Tk()
        self.window.geometry("900x600")
        self.window.configure(bg="#FFFFFF")
        self.canvas = Canvas(
            self.window,
            bg="#003151",
            height=600,
            width=900,
            bd=0,
            highlightthickness=0,
            relief="ridge")
        self.canvas.place(x=0, y=0)

        self.background_img = PhotoImage(file=f"background.png")
        self.background = self.canvas.create_image(
            450.0, 352.0,
            image=self.background_img)

        self.entry0 = ttk.Combobox(self.window, state="readonly", values=species)
        self.entry0.place(
            x=175, y=126,
            width=230,
            height=33)

        self.entry1 = ttk.Combobox(self.window, state="readonly", values=feature)
        self.entry1.place(
            x=175, y=237,
            width=230,
            height=33)

        self.entry2 = ttk.Combobox(self.window, state="readonly", values=species)
        self.entry2.place(
            x=514, y=126,
            width=230,
            height=33)
        self.entry2.bind()

        self.entry3 = ttk.Combobox(self.window, state="readonly", values=feature)
        self.entry3.place(
            x=514, y=237,
            width=230,
            height=33)

        scale_1 = IntVar(value=0)
        self.entry4 = ttk.Scale(self.window,variable=scale_1,to=100)
        self.entry4.place(
            x=175, y=348,
            width=190,
            height=28)

        scale_2 = IntVar(value=0)
        self.entry5 = ttk.Scale(self.window,variable=scale_2,to=10000)
        self.entry5.place(
            x=525, y=348,
            width=190,
            height=28)

        global var1
        var1= IntVar()
        self.entry6 = ttk.Checkbutton(self.window,variable=var1, onvalue=1, offvalue=0)
        self.entry6.configure(text='bias')
        self.entry6.place(
            x=465, y=465,
            width=65,
            height=28)

        # self.entry7_img = PhotoImage(file=f"img_textBox7.png")
        # self.entry7_bg = self.canvas.create_image(
        #     363.0, 362.0,
        #     image=self.entry7_img)

        self.entry7 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0,textvariable=scale_1)

        self.entry7.place(
            x=370, y=347,
            width=40,
            height=28)

        # self.entry8_img = PhotoImage(file=f"img_textBox8.png")
        # self.entry8_bg = self.canvas.create_image(
        #     703.0, 362.0,
        #     image=self.entry8_img)

        self.entry8 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0,textvariable=scale_2)

        self.entry8.place(
            x=720, y=348,
            width=40,
            height=28)

        self.img0 = PhotoImage(file=f"img0.png")
        self.b0 = Button(
            image=self.img0,
            borderwidth=0,
            highlightthickness=0,
            command=self.btn_clicked,
            relief="flat")

        self.b0.place(
            x=501, y=502,
            width=85,
            height=63)

        self.canvas.create_text(
            449.5, 33.0,
            text="Single Layer Perceptron Task",
            fill="#ffffff",
            font=("None", int(20.0)))

        self.canvas.create_text(
            449.5, 90.5,
            text="Choose the Classes",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            100.0, 143.5,
            text="Class 1",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            108.0, 254.5,
            text="Feature 1",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            105.5, 363.0,
            text="Learning\nRate",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            463.0, 143.5,
            text="Class 2",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            466.0, 254.5,
            text="Feature 2",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            470, 366.0,
            text="Number of\nEpocqs",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            450.0, 193.5,
            text="Choose The Features",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            448.5, 307.5,
            text="Parameters",
            fill="#ffffff",
            font=("None", int(16.0)))

    def run(self):
        self.window.resizable(False, False)
        self.window.mainloop()


if __name__ == "__main__":
    app = Gui()
    app.run()
