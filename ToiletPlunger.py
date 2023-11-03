import os
from datetime import datetime
import tarfile


def unclog_system(directory, cutoff_time, output_location):
    # create a dictionary to store the files by year and month
    files_by_month = {}
    # iterate through all files and directories in the directory
    for root, dirs, files in os.walk(directory):
        # add a progress notifier
        print(f"Currently in {root}. Plunging in progress...")
        for filename in files:
            # if the file is a h5 file and contains the keyword
            if filename.endswith(".h5") and "gage" in filename:
                # convert date integer to ctime format
                date = int(datetime.strptime(cutoff_time, "%Y-%m").timestamp())
                # get the creation time of the file
                creation_time = os.path.getctime(os.path.join(root, filename))
                print(filename)
                print(creation_time)
                print(date)
                # if the file was created before the date
                if creation_time < date:
                    # get the year and month of creation
                    year_month = datetime.fromtimestamp(creation_time).strftime('%Y-%m')
                    # add the file to the dictionary
                    if year_month not in files_by_month:
                        files_by_month[year_month] = []
                    files_by_month[year_month].append(os.path.join(root, filename))
    # iterate through the dictionary and compress the files by year and month
    for year_month, files in files_by_month.items():
        # create the compressed file
        with tarfile.open(os.path.join(output_location, f"{year_month}_gage_raw.tar.bz2"), "w:bz2") as f:
            # iterate through the files and add them to the compressed file
            for file in files:
                f.add(file, arcname=os.path.basename(file))
                # delete the original file
                os.remove(file)
                print(f"Deleted {file}")

# inputs: the first argument is the file directory. I would recommend something very very broad like "C:\\Users\\MicroscopePC\\" for example so the program can crawl through everything
# the second argument is the date string. Everything before this date will be compressed and then deleted. The format is "YYYY-MM" so for November 2023 it is "2023-11"
# the third argument is the output location. This is where the compressed files will be stored. I would set this to a directory in the harddrive I just bought.
# here is an example
unclog_system("C:\\Users\\tyxia\\", "2023-12", "C:\\Users\\tyxia\\Documents\\UltracoldRepos\\DataManagement2023\\targettest")