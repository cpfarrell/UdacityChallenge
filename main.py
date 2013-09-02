import csv
import numpy as np
import pylab as plt
import datetime
import time
import math

def readOutcome(outcome):

    if outcome == "":
        return 0

    else:
        return int(outcome)


def readTime(time):

    return datetime.datetime.strptime(time[:-1], "%Y-%m-%dT%H:%M:%S.%f")


def readcsvfile(file, Cast0 = str, Cast1 = str, Cast2 = str, Cast3 = str, Cast4 = str):

    lines = []

    with open(file, 'rb') as csvfile:
        next(csvfile)

        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            
            Fields = len(row)

            if Fields > 0:
                row[0] = Cast0(row[0])

            if Fields > 1:
                row[1] = Cast1(row[1])

            if Fields > 2:
                row[2] = Cast2(row[2])

            if Fields > 3:
                row[3] = Cast3(row[3])

            if Fields > 4:
                row[4] = Cast4(row[4])

            lines.append(row)

    return lines


#Investigate how much time generally elapses between visits
def DetermineSessionLength(accountVisits):

    deltat = []

    for acc in accountVisits:
        accountVisits[acc] = sorted(accountVisits[acc])
        
        for idx, visit in enumerate(accountVisits[acc]):
            if idx == 0:
                continue

            deltat.append((time.mktime(visit[0].timetuple()) - time.mktime(accountVisits[acc][idx-1][0].timetuple()))/60.)


    deltat = np.array(deltat)

    ax = plt.subplot(111)
    bins = [i*15 for i in range(1000)]
    plt.hist(np.clip(deltat, 0, 10001), bins, histtype='step', normed=1)
    ax.set_yscale('log')
    plt.xlim([0, 10000])
    ax.grid(True)
    plt.show()

    ax = plt.subplot(111)
    bins = [i for i in range(11000)]
    plt.hist(np.clip(deltat, 0, 10001), bins, histtype='step', normed=1, cumulative=True)
    plt.xlim([0, 10000])
    plt.ylim([0.9, 1.05])
    ax.grid(True)
    plt.show()


    ax = plt.subplot(111)
    bins = [i for i in range(11000)]
    plt.hist(np.clip(deltat, 0, 10001), bins, histtype='step', normed=1)
    ax.set_yscale('log')
    plt.xlim([0, 100])
    ax.grid(True)
    plt.show()

    ax = plt.subplot(111)
    bins = [i for i in range(123)]
    plt.hist(np.clip(deltat, 0, 121), bins, histtype='step', normed=1, cumulative=True)
    ax.set_yscale('linear')
    plt.xlim([0, 120])
    plt.ylim([0.8, 1.0])
    ax.grid(True)
    plt.show()


def main():

    accounts = readcsvfile('accounts.csv', str, readTime, readOutcome)

    nodevisits = readcsvfile('nodevisits.csv', str, str, readTime, str)

    submissions = readcsvfile('submissions.csv', str, str, readTime, str, str)

    accountVisits = {}

    for visit in nodevisits:

        acc = visit[1]

        if acc in accountVisits:
            accountVisits[acc].append((visit[2], visit[3]))
        
        else:
            accountVisits[acc] = [(visit[2], visit[3])]

    #DetermineSessionLength(accountVisits)

    for acc in accountVisits:
        accountVisits[acc] = sorted(accountVisits[acc])

        for idx, visit in enumerate(accountVisits[acc]):
            if idx == 0:
                continue

            #Time between visits in minutes
            deltatime = (time.mktime(visit[0].timetuple()) - time.mktime(accountVisits[acc][idx-1][0].timetuple()))/60.

            if deltatime > 20:
                print "New session"

if __name__ == '__main__':
    main()
