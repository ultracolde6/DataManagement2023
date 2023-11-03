import tkinter as tk
import GagePreprocess as gp
import numpy as np
import FolderOps as fo
import os
import json

class GagePreprocessGUI:
    def __init__(self, master):
        self.master = master
        master.title("Gagescope Preprocess Settings GUI")

        # Create labels and entry boxes for each variable
        self.het_freq_label = tk.Label(master, text="Heterodyne Frequency (MHz):")
        self.het_freq_entry = tk.Entry(master)
        self.samp_freq_label = tk.Label(master, text="Sampling Frequency (MHz):")
        self.samp_freq_entry = tk.Entry(master)
        self.step_time_label = tk.Label(master, text="Step Time (us):")
        self.step_time_entry = tk.Entry(master)
        self.filter_time_label = tk.Label(master, text="Filter Time (us):")
        self.filter_time_entry = tk.Entry(master)
        self.LO_power_label = tk.Label(master, text="LO Power (uW):")
        self.LO_power_entry = tk.Entry(master)
        self.kappa_label = tk.Label(master, text="Kappa (MHz (factor of 2pi)):")
        self.kappa_entry = tk.Entry(master)
        self.step_time = tk.Label(master, text="Step Time (us):")
        self.step_time_entry = tk.Entry(master)
        self.filter_time = tk.Label(master, text="Filter Time (us):")
        self.filter_time_entry = tk.Entry(master)
        self.plot_bool = tk.Label(master, text="Plot? (True/False):")
        self.plot_bool_entry = tk.Entry(master)
        self.backup_disk = tk.Label(master, text="Backup Disk (e.g. /media/backup/):")
        self.backup_disk_entry = tk.Entry(master)

        # Set default values for each entry box
        self.het_freq_entry.insert(0, "20")
        self.samp_freq_entry.insert(0, "200")
        self.step_time_entry.insert(0, "50")
        self.filter_time_entry.insert(0, "100")
        self.LO_power_entry.insert(0, "314")
        self.kappa_entry.insert(0, "1.1")
        self.plot_bool_entry.insert(0, "True")
        self.backup_disk_entry.insert(0, "/backup/")

        # Create a button to submit the input values
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)

        # Pack the labels, entry boxes, and button into the GUI
        self.het_freq_label.pack()
        self.het_freq_entry.pack()
        self.samp_freq_label.pack()
        self.samp_freq_entry.pack()
        self.step_time_label.pack()
        self.step_time_entry.pack()
        self.filter_time_label.pack()
        self.filter_time_entry.pack()
        self.LO_power_label.pack()
        self.LO_power_entry.pack()
        self.kappa_label.pack()
        self.kappa_entry.pack()
        self.plot_bool.pack()
        self.plot_bool_entry.pack()
        self.backup_disk.pack()
        self.backup_disk_entry.pack()
        self.submit_button.pack()


    def submit(self):
        # Get the values from the entry boxes and store them in variables
        het_freq = float(self.het_freq_entry.get())
        dds_freq = het_freq/2
        samp_freq = float(self.samp_freq_entry.get())
        step_time = float(self.step_time_entry.get())
        filter_time = float(self.filter_time_entry.get())
        LO_power = float(self.LO_power_entry.get())
        # convert string with pi in it to float
        kappa = float(self.kappa_entry.get()) * 2 * np.pi
        plot_bool = self.plot_bool_entry.get()
        backup_disk = self.backup_disk_entry.get()
        

        # Do something with the variables (e.g. print them to the console)
        print(f"Heterodyne Frequency: {het_freq}")
        print(f"DDS Frequency: {dds_freq}")
        print(f"Sampling Frequency: {samp_freq}")
        print(f"Step Time: {step_time}")
        print(f"Filter Time: {filter_time}")
        print(f"LO Power: {LO_power}")
        
        # write the settings to a json file
        settings = {"het_freq": het_freq,
                    "dds_freq": dds_freq,
                    "samp_freq": samp_freq,
                    "step_time": step_time,
                    "filter_time": filter_time,
                    "LO_power": LO_power,
                    "kappa": kappa,
                    "plot_bool": plot_bool}
        # Run the GagePreprocess function
        gage_preprocessor = gp.GagePreprocessor("run0",
                                                filter_time,
                                                step_time,
                                                True,
                                                plot_bool,
                                                het_freq,
                                                dds_freq,
                                                samp_freq,
                                                kappa,
                                                LO_power)
        dir = os.path.dirname(__file__)
        gage_folder = "DataPathGage"
        fo.make_tarfile(dir+backup_disk, gage_folder)
        fo.kill_processed_file(gage_folder)
        with open('log' + file_name + '.json', 'w') as fp:
            json.dump(settings, fp)
            
root = tk.Tk()
my_gui = GagePreprocessGUI(root)
root.mainloop()