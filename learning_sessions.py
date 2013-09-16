#Standard modules
import sys
import csv
import datetime
import time
import math
import itertools
from collections import defaultdict
import bisect
import heapq

#Thord party modules
import numpy as np
import pylab as plt
from sklearn import cluster
from sklearn.decomposition import SparsePCA


def main():
    accounts = csv_to_dict('accounts.csv', 0, cast_evals=[str, read_time, readOutcome], type="account")
    account_nodes = csv_to_dict('nodevisits.csv', 1, cast_evals=[str, str, read_time, str], type="node")
    account_submissions = csv_to_dict('submissions.csv', 1, cast_evals=[str, str, read_time, str, str], type="submission")

    account_visits = account_nodes
    for acc in account_visits:
        account_visits[acc].extend(account_submissions[acc])
        account_visits[acc] = sorted(account_visits[acc], key=lambda k: k['time'])
    session_length(account_visits)

    #Build sessions based on time scale determined from previous code as 15 minutes
    sessions = []
    for acc in account_visits:
        actions = []
        for idx, visit in enumerate(account_visits[acc]):
            if idx == 0:
                actions = {"node": [], "submission": [], "learning_outcome": accounts[acc][0]["learning_outcome"]}
                actions[account_visits[acc][idx]["type"]].append(visit)
            else:
                #Time between visits in minutes
                delta_time = delta_minutes(visit["time"], account_visits[acc][idx-1]["time"])
                #New session, defined as 15 minutes from above
                if delta_time > 15:
                    sessions.append(actions)
                    actions = {"node": [], "submission": [], "learning_outcome": accounts[acc][0]["learning_outcome"]}
                    actions[account_visits[acc][idx]["type"]].append(visit)

                else:
                    actions[account_visits[acc][idx]["type"]].append(visit)
        sessions.append(actions)

    for session in sessions:
        if len(session["node"]) > 0 and len(session["submission"]) > 0:
            session["start_time"] = min(session["node"][0]["time"], session["submission"][0]["time"])
            session["end_time"] = max(session["node"] [len(session["node"]) -1]["time"], session["submission"] [len(session["submission"]) -1]["time"])
        elif len(session["node"]) > 0:
            session["start_time"] = session["node"][0]["time"]
            session["end_time"] = session["node"] [len(session["submission"]) -1]["time"]
        else:
            session["start_time"] = session["submission"][0]["time"]
            session["end_time"] = session["submission"] [len(session["submission"]) -1]["time"]

    #Remove sessions without any time difference or no nodes visited
    sessions = [session for session in sessions if delta_minutes(session["end_time"], session["start_time"]) != 0]

    X = session_properties(sessions)
    X = standardize(X)
    pca = SparsePCA(n_components = 2)
    #Negative one just makes plot easier to look at, PCA is sign insensitive so no real effect
    X_r = -1 * pca.fit(X).transform(X)

    kmeans = cluster.KMeans(n_clusters=4)
    group = kmeans.fit_predict(X_r)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.rc('font', family='serif', size=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.scatter(X_r[:,0], X_r[:,1],s=20,marker = 'o', c=group)
    plt.show()

    outcomes = np.asarray([session["learning_outcome"] for session in sessions])
    session_by_outcome = []
    tags = []
    labels = get_labels(X_r, group, 4)
    for result in range(0, 4):
        session_by_outcome.append(group[outcomes == result])
        if result == 0:
            tags.append("No certificate achieved")
        else:
            tags.append("Mastery Level = " + str(result))

    plot_hist(session_by_outcome, x_min = 0, x_max = 4, y_min = 0, y_max = 1, bins = 4, tags = tags, y_label = "Fraction of sessions", labels=labels)


#Rearrange random numbering of groups to get same plot every time and attach labels to them
def get_labels(X_r, group, n_clusters):
    cluster_averages = []
    temp_group = np.copy(group)
    for i in range(n_clusters):
        X_i = X_r[temp_group==i, :]
        cluster_averages.append((np.average(X_i[:, 0]), np.average(X_i[:, 1])))

    labels = [None]*4
    for idx, cluster_average in enumerate(cluster_averages):
        if cluster_average[0] < 0 and cluster_average[1] < 0:
            labels[0] = "Low accuracy/\nLow review"
            group[temp_group==idx] = 0
        elif cluster_average[0] < 0 and cluster_average[1] > 0:
            labels[1] = "Low accuracy/\nHigh review"
            group[temp_group==idx] = 1
        elif cluster_average[0] > 0 and cluster_average[1] < 0:
            labels[2] = "High accuracy/\nLow review"
            group[temp_group==idx] = 2
        if cluster_average[0] > 0 and cluster_average[1] > 0:
            labels[3] = "High accuracy/\nHigh review"
            group[temp_group==idx] = 3

    return labels
            
def readOutcome(outcome):
    if outcome == "":
        return 0
    else:
        return int(outcome)


def read_time(time):
    return datetime.datetime.strptime(time[:-1], "%Y-%m-%dT%H:%M:%S.%f")


def csv_to_dict(file, key_column, cast_evals=[str, str, str, str, str], type=""):
    return_dict = defaultdict(list)
    with open(file, 'rb') as csvfile:
        header = next(csvfile).rstrip("\n").split(",")
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            base_dict = {"type":type}
            key = ""
            for i, (value, cast_eval) in enumerate(itertools.izip(row, cast_evals)):
                if i == key_column:
                    key = cast_eval(value)
                else:
                    base_dict[header[i]] = cast_eval(value)
            return_dict[key].append(base_dict)
    return return_dict


def delta_minutes(time1, time2):
    return (time.mktime(time1.timetuple()) - time.mktime(time2.timetuple()))/60.


def plot_hist(
    graphs, x_min=0, x_max=0, y_min=0, y_max=0, bins=1, histtype='step', normed=False, yscale="linear", 
    tags=[], x_label="", y_label="", save_path="", cumulative=False, labels=[]):

    plt.rc('font', family='serif', size=20) 
    ax = plt.subplot(111)

    histo_bins = np.linspace(x_min, x_max, bins + 1)
    if x_min != x_max:
        graphs = [np.clip(graph, x_min, x_max) for graph in graphs]
    plt.hist(graphs, histo_bins, normed = True, histtype=histtype, label=tags, linewidth=2.0, cumulative = cumulative)

    if x_min != x_max:
        ax.set_xlim(x_min, x_max)
    if y_min != y_max:
        ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)

    if labels:
        ticks = [x_min + (i+0.5) * (x_max - x_min)/bins for i in range(0, int(x_max - x_min))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,rotation=30, rotation_mode="anchor", ha="right")

    plt.subplots_adjust( hspace=0.4)
    plt.grid()
    if len(tags) > 1:
        plt.legend()
    ax.set_yscale(yscale)
    if save_path != "":
        plt.savefig('Results/' + save_path + '.png', bbox_inches=0) 
    plt.show()


#Investigate how much time generally elapses between visits
def session_length(account_visits):
    delta_t = []
    for acc in account_visits:
        for idx, visit in enumerate(account_visits[acc]):
            if idx == 0:
                continue
            delta_t.append(delta_minutes(visit["time"], account_visits[acc][idx-1]["time"]))

    delta_t = np.array(delta_t)
    plot_hist(
        delta_t, x_min=0, x_max=120, bins=120, normed=True, 
        yscale="log", x_label="Minutes between student activities", y_label="Fraction of splits")
    plot_hist(
        delta_t, x_min=0, x_max=120, y_min=0.5, y_max=1.0, bins=120, normed=True, 
        yscale="linear", cumulative=True, x_label="Minutes between student activities", y_label="Cumulative fraction of splits")
    plot_hist(delta_t / (60 * 24), x_min=0, x_max=7, bins=75, yscale="log", x_label="Days between student activities", y_label="Fraction of splits")


def findNext(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    else:
        return -1


def exercise(path):
    return path[:path.find("/m-")]


def session_properties(sessions, show_hists=False):
    #Different variables about sessions to keep track of
    session_nodes = np.asarray([len(session["node"]) for session in sessions])
    session_time = np.asarray([delta_minutes(session["end_time"], session["start_time"]) for session in sessions])
    repeat_nodes = np.zeros(len(sessions))
    quiz_right = np.zeros(len(sessions))
    quiz_answers = np.zeros(len(sessions))
    fraction_right = np.zeros(len(sessions))
    fraction_test = np.zeros(len(sessions))
    average_tests = np.zeros(len(sessions))
    average_first_tests = np.zeros(len(sessions))
    fraction_end_right = np.zeros(len(sessions))
    aver_nodes_bef_first_submit = np.zeros(len(sessions))
    aver_exer_nodes_bef_first_submit = np.zeros(len(sessions))
    aver_nodes_bef_final_submit = np.zeros(len(sessions))
    aver_exer_nodes_bef_final_submit = np.zeros(len(sessions))

    for i, session in enumerate(sessions):                
        paths_visited = defaultdict(list)

        for node in session["node"]:
            paths_visited[node["content_path"]].append(node["time"])

        repeat_nodes[i] = len(session["node"]) - len(paths_visited)


        right_quiz = 0.
        wrong_quiz = 0.
        quiz_visited = defaultdict(list)

        for submission in session["submission"]:
            quiz_visited[submission["content_path"]].append(submission)

            if submission["evaluation"] == "True":
                right_quiz += 1

            if submission["evaluation"] == "False":
                wrong_quiz += 1

        if (right_quiz + wrong_quiz) > 0:
            fraction_right[i] = right_quiz / (right_quiz + wrong_quiz)

        if len(session["submission"]) > 0:
            fraction_test[i] = (len(session["submission"]) - right_quiz - wrong_quiz) / len(session["submission"])


        for quiz in quiz_visited:
            n_tests = 0
            answered = False
            quiz_subissions = quiz_visited[quiz]

            found_correct = False
            first_submit = -1
            last_submit = -1

            for quiz_subission in quiz_subissions:
                #Just tested not scored
                if quiz_subission["evaluation"] == "":
                    n_tests += 1
                else:
                    last_submit = quiz_subission["time"]
                    #First time this quiz has been scored
                    if not answered:
                        average_first_tests[i] += n_tests
                        first_submit = quiz_subission["time"]
                        quiz_answers[i] += 1
                    answered = True
                    
                if quiz_subission["evaluation"] == "True":
                    found_correct = True

            average_tests[i] += n_tests

            if found_correct:
                quiz_right[i] += 1
                fraction_end_right[i] += 1

            #Investigate number of node visits user makes between seeing quiz and scoring
            #Quiz was never scored
            if first_submit == -1:
                continue

            firstSeen = quiz_subissions[0]["time"]
            #Should be true as must see quiz to submit but possible they have been split into different sessions
            if len(paths_visited[quiz]) > 0:
                firstSeen = min(firstSeen, paths_visited[quiz][0])


            node_times = [node["time"] for node in session["node"]]
            first_node = findNext(node_times, firstSeen)
        
            #No nodes were viewed after being presented with the quiz
            if first_node == -1:
                continue

            while first_node < len(node_times) and node_times[first_node] < first_submit:
                node_path = session["node"][first_node]["content_path"]
                #Just reviewing quiz not going back for more info
                if node_path == quiz:
                    first_node += 1
                    continue
                aver_nodes_bef_first_submit[i] += 1                
                aver_nodes_bef_final_submit[i] += 1
                if exercise(node_path) == exercise(quiz):
                    aver_exer_nodes_bef_first_submit[i] += 1
                    aver_exer_nodes_bef_final_submit[i] += 1
                first_node += 1

            while first_node < len(node_times) and node_times[first_node] < last_submit:
                node_path = session["node"][first_node]["content_path"]
                #Just reviewing quiz not going back for more info
                if node_path == quiz:
                    first_node += 1
                    continue
                aver_nodes_bef_final_submit[i] += 1                
                if exercise(node_path) == exercise(quiz):
                    aver_exer_nodes_bef_final_submit[i] += 1
                first_node += 1

        if len(quiz_visited) > 0:
            average_tests[i] /= len(quiz_visited)
            average_first_tests[i] /= len(quiz_visited)
            fraction_end_right[i] /= len(quiz_visited)
            aver_nodes_bef_first_submit[i] /= len(quiz_visited)
            aver_exer_nodes_bef_first_submit[i] /= len(quiz_visited)
            aver_nodes_bef_final_submit[i] /= len(quiz_visited)
            aver_exer_nodes_bef_final_submit[i] /= len(quiz_visited)

    if show_hists:
        plot_hist(session_nodes, x_min=1, x_max=150, bins=50, normed=True, yscale="log", x_label="Nodes")
        plot_hist(session_time, x_min=0, x_max=150, bins=50, normed=True, yscale="log", x_label="Time")
        plot_hist(np.divide(session_nodes, session_time), x_min=0, x_max=10, bins=50, normed=False, yscale="log", x_label="Nodes per minute")
        plot_hist(repeat_nodes, x_min=0, x_max=50, bins=50, normed=True, yscale="log", x_label="Repeated nodes")
        plot_hist(fraction_right, x_min=0, x_max=1, bins=50, normed=True, yscale="symlog", x_label="Fraction right")
        plot_hist(fraction_test, x_min=0, x_max=1, bins=50, normed=True, yscale="symlog", x_label="Fraction test")
        plot_hist(average_tests, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Average tests")
        plot_hist(average_first_tests, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Average tests before submit")
        plot_hist(fraction_end_right, x_min=0, x_max=1, bins=50, normed=True, yscale="log", x_label="Fraction eventually right")  
        plot_hist(aver_nodes_bef_first_submit, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Nodes between quiz view and first submit")
        plot_hist(aver_nodes_bef_final_submit, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Nodes between quiz view and final submit")
        plot_hist(aver_exer_nodes_bef_first_submit, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Exer. nodes between quiz view and first submit")
        plot_hist(aver_exer_nodes_bef_final_submit, x_min=0, x_max=10, bins=50, normed=True, yscale="log", x_label="Exer. nodes between quiz view and first submit")
        plot_hist(quiz_answers, x_min=0, x_max=50, bins=50, normed=True, yscale="log", x_label="Quizzes answered")
        plot_hist(np.divide(quiz_answers, session_time), x_min=0, x_max=6, bins=50, normed=False, yscale="log", x_label="Quizzes answered per minute")
        plot_hist(quiz_right, x_min=0, x_max=50, bins=50, normed=True, yscale="log", x_label="Total quizzes correct")
        plot_hist(np.divide(quiz_right, session_time), x_min=0, x_max=6, bins=50, normed=False, yscale="log", x_label="Correct quizzes per minute")

    #Add features to training matrix
    #Some of the features are (dramatically) skewed so cap (and possibly take log of) the distribution to put on similar scales
    percent_cut = 0.2
    X = np.zeros(shape=(len(sessions), 17))
    X[:,0] = largest_percent(np.log2(session_nodes + 0.5), percent_cut)
    X[:,1] = largest_percent(np.log2(session_time), percent_cut)
    X[:,2] = largest_percent(np.divide(session_nodes, session_time), percent_cut)
    X[:,3] = largest_percent(np.log2(repeat_nodes + 0.5), percent_cut)
    X[:,4] = fraction_right
    X[:,5] = fraction_test
    X[:,6] = largest_percent(average_tests, percent_cut)
    X[:,7] = largest_percent(average_first_tests, percent_cut)
    X[:,8] = fraction_end_right
    X[:,9] = largest_percent(aver_nodes_bef_first_submit, percent_cut)
    X[:,10] = largest_percent(aver_exer_nodes_bef_first_submit, percent_cut)
    X[:,11] = largest_percent(aver_nodes_bef_final_submit, percent_cut)
    X[:,12] = largest_percent(aver_exer_nodes_bef_final_submit, percent_cut)
    X[:,13] = largest_percent(np.log2(quiz_answers + 0.5), percent_cut)
    X[:,14] = largest_percent(np.divide(quiz_answers, session_time), percent_cut)
    X[:,15] = largest_percent(np.log2(quiz_right + 0.5), percent_cut)
    X[:,16] = largest_percent(np.divide(quiz_right, session_time), percent_cut)

    return X


#Get the value containing all but the largest p percent
def largest_percent(data, percent):

    n = percent * len(data)
    large_values = []
    top_value = -1 * sys.maxint
    for i in range(len(data)):
        heapq.heappush(large_values, data[i])
        if (len(large_values)>n):
            top_value = max(top_value, heapq.heappop(large_values))

    return np.clip(data, 0, top_value)

#Standardize data by clipping outliers and then setting mean to zero and unit variance
def standardize(X):
    columns = X.shape[1]
    for column in range(columns):
        data = X[:, column]
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        X[:, column] = data
    return X
    

if __name__ == '__main__':
    main()
