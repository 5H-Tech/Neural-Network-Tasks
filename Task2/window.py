from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import single_layer_perceptron as slp

class Gui:
    def btn_clicked(self):
        global class1
        global class2
        global feature1
        global feature2
        global learning_rate
        global is_bias
        global mse_threshold
        class1 = self.entry0.get()
        class2 = self.entry2.get()
        feature1 = self.entry1.get()
        feature2 = self.entry3.get()
        learning_rate = self.entry7.get()
        mse_threshold = self.entry8.get()

        is_bias = var1.get()
        try:
            mse_threshold = float(mse_threshold)
            learning_rate = float(learning_rate)
            model = slp.Slp(learning_rate=learning_rate, is_bias=is_bias, first_class=class1,
                            second_class=class2,
                            first_feature=feature1, second_feature=feature2,threshold=mse_threshold)
            model.preprocessing(df=pd.read_csv('Datasets/penguins.csv'))
            model.run_model()
            y_res = model.predict(model.x_test)
            model.evaluation(y_res)
            model.plot_confusion_matrix(model.y_test, y_res)
        except ValueError:
            messagebox.showinfo("Error", f'Please, Enter the valid number{ValueError}')
        # main.run_model(feature1, feature2, class1, class2, int(is_bias), int(float(epochs)), float(learning_rate))

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

        # learning rate scale
        scale_1 = IntVar(value=0)
        self.entry4 = ttk.Scale(self.window, variable=scale_1, to=1)
        self.entry4.place(
            x=175, y=348,
            width=190,
            height=28)

        # mse_threshold scale
        mse_scale = IntVar(value=0)
        self.entry5 = ttk.Scale(self.window, variable=mse_scale, to=1)
        self.entry5.place(
            x=525, y=348,
            width=190,
            height=28)

        global var1
        var1 = IntVar()
        self.entry6 = ttk.Checkbutton(self.window, variable=var1, onvalue=1, offvalue=0)
        self.entry6.configure(text='bias')
        self.entry6.place(
            x=410, y=430,
            width=65,
            height=28)

        # learning rate textbox
        self.entry7 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=scale_1)

        self.entry7.place(
            x=370, y=347,
            width=40,
            height=28)

        # textbox of mse_threshold
        self.entry8 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=mse_scale)
        self.entry8.place(
            x=720, y=348,
            width=40,
            height=28)

        # Go button
        self.img0 = PhotoImage(file=f"img0.png")
        self.b0 = Button(
            image=self.img0,
            borderwidth=0,
            highlightthickness=0,
            command=self.btn_clicked,
            relief="flat")

        self.b0.place(
            x=400, y=502,
            width=85,
            height=63)

        self.canvas.create_text(
            449.5, 33.0,
            text="Adaline Learning Algorithm Task",
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
            100.0, 363.0,
            text="Learning\nRate",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            175, 330,
            text="0",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            365, 330,
            text="1",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            470, 366.0,
            text="MSE\nThreshold",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            525, 330,
            text="0",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            715, 330,
            text="1",
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

    def evaluation(self, x_tran, y_tran, y_test, y_pred_test, weight1, weight2, bias):
        # print("Perceptron classification accuracy", accuracy(y_test, y_pred_test))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(x_tran[:, 0], x_tran[:, 1], marker="o", c=y_tran)

        x0_1 = np.amin(x_tran[:, 0])
        x0_2 = np.amax(x_tran[:, 0])

        x1_1 = (-weight1 * x0_1 - bias) / weight2
        x1_2 = (-weight1 * x0_2 - bias) / weight2

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(x_tran[:, 1])
        ymax = np.amax(x_tran[:, 1])
        ax.set_ylim([ymin - 3, ymax + 3])

        plt.show()


if __name__ == "__main__":
    app = Gui()
    app.run()
