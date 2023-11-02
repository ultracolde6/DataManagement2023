import tkinter as tk

class GagePreprocessGUI:
    def __init__(self, master):
        self.master = master
        master.title("Gage Preprocess GUI")

        # Create labels and entry boxes for each variable
        self.het_freq_label = tk.Label(master, text="Heterodyne Frequency (MHz):")
        self.het_freq_entry = tk.Entry(master)
        self.dds_freq_label = tk.Label(master, text="DDS Frequency (MHz):")
        self.dds_freq_entry = tk.Entry(master)
        self.samp_freq_label = tk.Label(master, text="Sampling Frequency (MHz):")
        self.samp_freq_entry = tk.Entry(master)
        self.step_time_label = tk.Label(master, text="Step Time (us):")
        self.step_time_entry = tk.Entry(master)
        self.filter_time_label = tk.Label(master, text="Filter Time (us):")
        self.filter_time_entry = tk.Entry(master)
        self.LO_power_label = tk.Label(master, text="LO Power (uW):")
        self.LO_power_entry = tk.Entry(master)

        # Set default values for each entry box
        self.het_freq_entry.insert(0, "20.000446")
        self.dds_freq_entry.insert(0, "10.000223")
        self.samp_freq_entry.insert(0, "200")
        self.step_time_entry.insert(0, "50")
        self.filter_time_entry.insert(0, "100")
        self.LO_power_entry.insert(0, "314")

        # Create a button to submit the input values
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)

        # Pack the labels, entry boxes, and button into the GUI
        self.het_freq_label.pack()
        self.het_freq_entry.pack()
        self.dds_freq_label.pack()
        self.dds_freq_entry.pack()
        self.samp_freq_label.pack()
        self.samp_freq_entry.pack()
        self.step_time_label.pack()
        self.step_time_entry.pack()
        self.filter_time_label.pack()
        self.filter_time_entry.pack()
        self.LO_power_label.pack()
        self.LO_power_entry.pack()
        self.submit_button.pack()

    def submit(self):
        # Get the values from the entry boxes and store them in variables
        het_freq = float(self.het_freq_entry.get())
        dds_freq = het_freq/2
        samp_freq = float(self.samp_freq_entry.get())
        step_time = float(self.step_time_entry.get())
        filter_time = float(self.filter_time_entry.get())
        LO_power = float(self.LO_power_entry.get())

        # Do something with the variables (e.g. print them to the console)
        print(f"Heterodyne Frequency: {het_freq}")
        print(f"DDS Frequency: {dds_freq}")
        print(f"Sampling Frequency: {samp_freq}")
        print(f"Step Time: {step_time}")
        print(f"Filter Time: {filter_time}")
        print(f"LO Power: {LO_power}")

root = tk.Tk()
my_gui = GagePreprocessGUI(root)
root.mainloop()