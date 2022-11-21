from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multi_layer_perceptron as mlp


class Gui:
    def btn_clicked(self):
        global activation_function
        global neurons
        global learning_rate
        global epochs
        global is_bias
        global hidden_layers

        activation_function = self.entry0.get()
        hidden_layers = self.entry2.get()
        neurons = self.entry4.get()
        learning_rate = self.entry6.get()
        epochs = self.entry8.get()
        is_bias = var1.get()

        try:
            learning_rate = float(learning_rate)
            epochs = int(epochs)
            hidden_layers = int(hidden_layers)
            neurons = int(neurons)
            model = mlp.Mlp(learning_rate=learning_rate, epochs=epochs, is_bias=is_bias, hidden_layers=hidden_layers,
                            neurons=neurons, act_func_type=activation_function)
            model.preprocessing(df=pd.read_csv('Datasets/penguins.csv'))
            model.run_model()
            y_res = model.predict(model.x_test)
            model.evaluation(y_res)
            model.plot_confusion_matrix(model.y_test, y_res)
        except ValueError:
            messagebox.showinfo("Error", f'Please, Enter the valid number{ValueError}')
        # main.run_model(feature1, feature2, class1, class2, int(is_bias), int(float(epochs)), float(learning_rate))

    def __init__(self, master=None):
        # global species
        # global feature
        global activation_functions
        df = pd.read_csv("Datasets/penguins.csv")
        # species = list(df["species"].unique())
        # feature = list(df.columns[1:])
        activation_functions = ['Sigmoid', 'Hyperbolic Tangent Sigmoid']
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

        # Activation function combobox
        self.entry0 = ttk.Combobox(self.window, state="readonly", values=activation_functions)
        self.entry0.place(
            x=405, y=126,
            width=230,
            height=33)

        # Hidden layers scale
        hidden_layers_scale = IntVar(value=0)
        self.entry1 = ttk.Scale(self.window, variable=hidden_layers_scale, to=100)
        self.entry1.place(
            x=175, y=237,
            width=190,
            height=28)

        # Hidden layers textbox
        self.entry2 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=hidden_layers_scale)

        self.entry2.place(
            x=370, y=237,
            width=40,
            height=28)

        # Neurons scale
        neurons_scale = IntVar(value=0)
        self.entry3 = ttk.Scale(self.window, variable=neurons_scale, to=100)
        self.entry3.place(
            x=526, y=237,
            width=190,
            height=28)

        # Neurons textbox
        self.entry4 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=neurons_scale)

        self.entry4.place(
            x=720, y=237,
            width=40,
            height=28)

        # Learning rate scale
        lr_scale = IntVar(value=0)
        self.entry5 = ttk.Scale(self.window, variable=lr_scale, to=1)
        self.entry5.place(
            x=175, y=348,
            width=190,
            height=28)

        # Learning rate textbox
        self.entry6 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=lr_scale)

        self.entry6.place(
            x=370, y=347,
            width=40,
            height=28)
        # command = lambda: self.entry5.set('%d' % float(self.entry5))

        # Epochs scale
        epochs_scale = IntVar(value=0)
        self.entry7 = Scale(from_=0, to=1000, resolution=1,variable=epochs_scale,orient=HORIZONTAL)
        #self.entry7 = ttk.Scale(self.window, variable=epochs_scale, to=1000)
        self.entry7.place(
            x=525, y=348,
            width=190,
            height=50)

        # Epochs textbox
        self.entry8 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0, textvariable=epochs_scale)

        self.entry8.place(
            x=720, y=348,
            width=40,
            height=50)


        global var1
        var1 = IntVar()
        self.entry9 = ttk.Checkbutton(self.window, variable=var1, onvalue=1, offvalue=0)
        self.entry9.configure(text='bias')
        self.entry9.place(
            x=410, y=430,
            width=65,
            height=28)

        # self.entry7_img = PhotoImage(file=f"img_textBox7.png")
        # self.entry7_bg = self.canvas.create_image(
        #     363.0, 362.0,
        #     image=self.entry7_img)



        # self.entry8_img = PhotoImage(file=f"img_textBox8.png")
        # self.entry8_bg = self.canvas.create_image(
        #     703.0, 362.0,
        #     image=self.entry8_img)



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
            text="Multi Layer Perceptron Task",
            fill="#ffffff",
            font=("None", int(20.0)))

        self.canvas.create_text(
            449.5, 90.5,
            text="Choose The Activation Function",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            300.0, 143.5,
            text="Activation Function",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            108.0, 254.5,
            text="Number of\nHidden Layers",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            175, 226,
            text="0",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            365, 226,
            text="100",
            fill="#ffffff",
            font=("None", int(16.0)))
        self.canvas.create_text(
            105.5, 363.0,
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
            466.0, 254.5,
            text="Neurons /\nLayer",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            525, 226,
            text="0",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            715, 226,
            text="100",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            525, 330,
            text="0",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            715, 330,
            text="1000",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            470, 366.0,
            text="Number of\nEpochs",
            fill="#ffffff",
            font=("None", int(16.0)))

        self.canvas.create_text(
            450.0, 193.5,
            text="Adjust Depth Parameters",
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
