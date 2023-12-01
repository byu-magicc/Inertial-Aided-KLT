import csv

folder = "flight_angle"

# get list of speeds we have data for
speeds = [x / 10.0 for x in range(35, 90, 5)]

# open the file and
no_imu = []
imu = []

with open(folder + "/combined_tracking_output.csv", 'w', newline='') as outputFile:
    outputWriter = csv.writer(outputFile)
    for speed in speeds:
        # open the non imu file and read in the data
        with open(folder + "/tracking_data_" + str(speed) + ".csv", 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                no_imu.append(row)

        # open the imu file and read in the data
        with open(folder + "/tracking_data_" + str(speed) + "_imu.csv", 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                imu.append(row)

        # figure out which has more data
        imu_longer = False
        shorter_length = 0
        if len(imu) > len(no_imu):
            shorter_length = len(no_imu)
            imu_longer = True
        else:
            shorter_length = len(imu)

        # write this speed to the output file
        # write while we have both columns
        for i in range(shorter_length):
            outputWriter.writerow([speed] + no_imu[i] + imu[i])

        # whichever is longer, write that one with the other blank
        if imu_longer:
            for i in range(shorter_length, len(imu)):
                outputWriter.writerow([speed] + [""] + imu[i])
        else:
            for i in range(shorter_length, len(no_imu)):
                outputWriter.writerow([speed] + no_imu[i] + [""])
