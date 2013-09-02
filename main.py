import csv
import numpy as np
import pylab as plt
import dateutil.parser
import datetime
from datetime import timedelta

def readOutcome(outcome):

    if outcome == "":
        return 0

    else:
        return int(outcome)


def readcsvfile(file, Cast0 = str, Cast1 = str, Cast2 = str, Cast3 = str, Cast4 = str):

    print Cast0(4)

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

def main():

    accounts = readcsvfile('accounts.csv', str, dateutil.parser.parse, readOutcome)

    nodevisits = readcsvfile('nodevisits.csv', str, str, dateutil.parser.parse, str)

    submissions = readcsvfile('submissions.csv', str, str, dateutil.parser.parse, str, str)

    accountVisits = {}

    for visit in nodevisits:

        acc = visit[1]

        if acc in accountVisits:
            accountVisits[acc].append((visit[2], visit[3]))
        
        else:
            accountVisits[acc] = [(visit[2], visit[3])]


    i = 0

    for acc in accountVisits:
        accountVisits[acc] = sorted(accountVisits[acc])
        
        for visit in accountVisits[acc]:
            print visit

        print "\n"
        if i > 5:
            break

        i += 1

    progress = [int(row[2]) for row in accounts]

    ax = plt.subplot(111)

    plt.hist(np.array(progress), 4, histtype='step')

    ax.set_yscale('log')

    plt.show()

if __name__ == '__main__':
    main()
