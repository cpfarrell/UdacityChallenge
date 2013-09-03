import csv
import numpy as np
import pylab as plt
import datetime
import time
import math
import itertools


def readOutcome(outcome):

    if outcome == "":
        return 0

    else:
        return int(outcome)


def readTime(time):

    return datetime.datetime.strptime(time[:-1], "%Y-%m-%dT%H:%M:%S.%f")


def readcsvfile(file, CastEvals = [str, str, str, str, str]):

    lines = []

    with open(file, 'rb') as csvfile:
        next(csvfile)

        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:

            for i, (value, CastEval) in enumerate(itertools.izip(row, CastEvals)):

                row[i] = CastEval(value)

            lines.append(row)

    return lines


def deltaMinutes(time1, time2):

    return (time.mktime(time1.timetuple()) - time.mktime(time2.timetuple()))/60.


def plotHist(Values, Xmin = 0, Xmax = 0, Ymin = 0, Ymax = 0, Bins = 100., histtype = 'step', normed = False, yscale = "linear", cumulative = False):

    ax = plt.subplot(111)
    bins = np.linspace(Xmin, Xmax, Bins)

    plt.hist(np.clip(Values, Xmin, Xmax), bins, histtype=histtype, normed=normed, cumulative = cumulative)

    plt.xlim([Xmin, Xmax])

    if Ymin != Ymax:
        plt.ylim([Ymin, Ymax])

    ax.set_yscale(yscale)
    ax.grid(True)
    plt.show()

#Investigate how much time generally elapses between visits
def determineSessionLength(accountVisits):

    deltat = []

    for acc in accountVisits:
        accountVisits[acc] = sorted(accountVisits[acc], key=lambda k: k['time'])

        for idx, visit in enumerate(accountVisits[acc]):
            if idx == 0:
                continue

            deltat.append(deltaMinutes(visit["time"], accountVisits[acc][idx-1]["time"]))

    deltat = np.array(deltat)

    plotHist(deltat, Xmin = 0, Xmax = 120, Bins = 120, normed = True, yscale = "log")
    plotHist(deltat, Xmin = 0, Xmax = 120, Ymin = 0.7, Ymax = 1.0, Bins = 120, normed = True, yscale = "linear", cumulative = True)

    plotHist(deltat, Xmin = 0, Xmax = 10000, Bins = 75, yscale = "log")

def sessionProperties(accountSessions):

    allSessions = []
    for acc in accountSessions:
        allSessions.extend(accountSessions[acc])


    sessionNodes = np.asarray([len(session) for session in allSessions])
    sessionTime = np.asarray([deltaMinutes(session[len(session)-1]["time"], session[0]["time"]) for session in allSessions])

    plotHist(sessionNodes, Xmin = 1, Xmax = 50, Bins = 50, normed = True, yscale = "log")

    plotHist(sessionTime, Xmin = 0, Xmax = 50, Bins = 50, normed = True, yscale = "log")

    plotHist(np.divide(sessionTime, sessionNodes), Xmin = 0, Xmax = 10, Bins = 50, normed = True, yscale = "log")

def main():

    accounts = readcsvfile('accounts.csv', CastEvals = [str, readTime, readOutcome])

    nodevisits = readcsvfile('nodevisits.csv', CastEvals = [str, str, readTime, str])

    submissions = readcsvfile('submissions.csv', CastEvals = [str, str, readTime, str, str])

    accountVisits = {}

    for visit in nodevisits:

        acc = visit[1]

        if acc in accountVisits:
            accountVisits[acc].append({"time":visit[2], "path":visit[3]})
        
        else:
            accountVisits[acc] = [{"time":visit[2], "path":visit[3]}]

    determineSessionLength(accountVisits)

    accountSessions = {}

    for acc in accountVisits:
        accountVisits[acc] = sorted(accountVisits[acc], key=lambda k: k['time'])
        accountSessions[acc] = []

        Session = []

        for idx, visit in enumerate(accountVisits[acc]):
            if idx == 0:
                session = [visit]

            else:
                #Time between visits in minutes
                deltaTime = deltaMinutes(visit["time"], accountVisits[acc][idx-1]["time"])

                #New session, defined as 20 minutes from above
                if deltaTime > 20:
                    accountSessions[acc].append(session)
                    session = [visit]

                else:
                    session.append(visit)

        accountSessions[acc].append(session)


    sessionProperties(accountSessions)

if __name__ == '__main__':
    main()
