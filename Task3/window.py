from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
from preprocessing_utilts import preprocessing
from MultiLayerPerceptron import MultiLayerPerceptron


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
            tmp_tokens = neurons.split(',')
            num_neurons_par_hidden =[]
            for token in tmp_tokens:
                num_neurons_par_hidden.append(int(token))
            learning_rate = float(learning_rate)
            hidden_layers = int(hidden_layers)
            # neurons = int(neurons)
            epochs = int(epochs)
            x_train, x_test, y_train, y_test = preprocessing()
            clf = MultiLayerPerceptron(learning_rate, 100000, activation_function == 'Sigmoid')
            clf.add_output_layer(3)
            for i in range(hidden_layers):
                clf.add_hidden_layer(num_neurons_par_hidden[i])
            clf.fit(x_train, y_train)
            clf.predict_and_get_accuracy(x_test, y_test, "Test")
            clf.plot_accuracy_graph()
        except ValueError:
            messagebox.showinfo("Error", f'Please, Enter the valid number{ValueError}')

    def __init__(self, master=None):
        global activation_functions
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

        # # Hidden layers scale
        # hidden_layers_scale = IntVar(value=0)
        # self.entry1 = ttk.Scale(self.window, variable=hidden_layers_scale, to=100)
        # self.entry1.place(
        #     x=175, y=237,
        #     width=190,
        #     height=28)

        # Hidden layers textbox
        self.entry2 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0)

        self.entry2.place(
            x=225, y=237,
            width=100,
            height=28)

        # # Neurons scale
        # neurons_scale = IntVar(value=0)
        # self.entry3 = ttk.Scale(self.window, variable=neurons_scale, to=100)
        # self.entry3.place(
        #     x=526, y=237,
        #     width=190,
        #     height=28)

        # Neurons textbox
        self.entry4 = Entry(
            bd=0,
            bg="#d9d9d9",
            highlightthickness=0)

        self.entry4.place(
            x=520, y=237,
            width=250,
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

        # Epochs scale
        epochs_scale = IntVar(value=0)
        self.entry7 = Scale(from_=0, to=1000, resolution=1, variable=epochs_scale, orient=HORIZONTAL)
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


if __name__ == "__main__":
    app = Gui()
    app.run()
