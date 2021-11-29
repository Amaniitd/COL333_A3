import matplotlib.pyplot as plt


# temporary variables

passenger_loaction = (0, 0)
destination_loaction = (4, 4)

# All possible states - 1250
# a state -> taxi_loaction, passenger is picked up or not, passenger location
States = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            for l in range(5):
                States.append(((i, j), False, (k, l)))
                States.append(((i, j), True, (k, l)))

# All possible actions: 4 directons, P - pickup, D - drop
actions = ['N', 'S', 'E', 'W', 'P', 'D']
Directions = ['N', 'S', 'E', 'W']

Colors = {'R': (0, 4), 'G': (4, 4), 'B': (3, 0), 'Y': (0, 0)}
ColorsList = [(0, 4), (4, 4), (3, 0), (0, 0)]

# Function for checking if there is a wall or not


def checkWall(xt, yt, d):
    rightWall = [(1, 1), (1, 0), (3, 0), (3, 1), (2, 3),
                 (2, 4)]  # wall encountered in moving right
    leftWall = [(0, 1), (0, 0), (2, 0), (2, 1), (1, 3),
                (1, 4)]  # wall encountered in moving left
    if d == 2:
        if ((xt, yt) in rightWall):
            return False
    elif d == 3:
        if ((xt, yt) in leftWall):
            return False
    return True

# Transition Function
# Right now only use for directional actions (pickup and drop are hard coded)


def Transition(s, a, s_):
    xt = s[0][0]
    yt = s[0][1]
    x2 = s_[0][0]
    y2 = s_[0][1]
    tmp_dict = {}
    tmp_dict[(xt, yt)] = 0
    if yt < 4 and checkWall(xt, yt, 0):
        tmp_dict[(xt, yt + 1)] = 0.05
    else:
        tmp_dict[(xt, yt)] += 0.05
    if yt > 0 and checkWall(xt, yt, 1):
        tmp_dict[(xt, yt - 1)] = 0.05
    else:
        tmp_dict[(xt, yt)] += 0.05
    if xt > 0 and checkWall(xt, yt, 2):
        tmp_dict[(xt - 1, yt)] = 0.05
    else:
        tmp_dict[(xt, yt)] += 0.05
    if xt < 4 and checkWall(xt, yt, 3):
        tmp_dict[(xt + 1, yt)] = 0.05
    else:
        tmp_dict[(xt, yt)] += 0.05
    if a == 'N':
        if yt < 4 and checkWall(xt, yt, 0):
            tmp_dict[(xt, yt + 1)] += 0.80
        else:
            tmp_dict[(xt, yt)] += 0.80
    elif a == 'S':
        if yt > 0 and checkWall(xt, yt, 1):
            tmp_dict[(xt, yt - 1)] += 0.80
        else:
            tmp_dict[(xt, yt)] += 0.80
    elif a == 'W':
        if xt > 0 and checkWall(xt, yt, 2):
            tmp_dict[(xt - 1, yt)] += 0.80
        else:
            tmp_dict[(xt, yt)] += 0.80
    elif a == 'E':
        if xt < 4 and checkWall(xt, yt, 3):
            tmp_dict[(xt + 1, yt)] += 0.80
        else:
            tmp_dict[(xt, yt)] += 0.80

    if (x2, y2) in tmp_dict:
        return tmp_dict[(x2, y2)]
    return 0

# Tester for Transition Function
# s = ((1, 4), False, (0, 0))
# s_ = ((1, 4), False, (0, 0))
# a = 'E'
# print(Transition(s, a, s_))


def ValueIteration(GAMMA, EPSILON=0.05):
    # Initial Values
    V = {}
    for s in States:
        V[s] = 0
    max_diffList = []
    iterations = 0
    while True:
        iterations += 1
        max_diff = 0
        for s in States:
            V_old = V[s]
            V_new = 0
            for a in actions:
                # Hardcoded for pickup actions
                if a == 'P':
                    V_tmp = GAMMA*V_old
                    if (s[0] == s[2]):
                        V_tmp -= 1
                    else:
                        V_tmp -= 10
                    V_new = max(V_new, V_tmp)

                elif a == 'D':
                    # Hardcoded for putdown actions
                    V_tmp = GAMMA*V_old
                    if (s[0] == destination_loaction):
                        if s[1]:
                            V_tmp += 20
                        else:
                            V_tmp -= 1
                    else:
                        V_tmp -= 10
                    V_new = max(V_new, V_tmp)
                else:
                    V_tmp = 0
                    xt = s[0][0]
                    yt = s[0][1]
                    # s_dash contains possible states having transition value non-zero
                    s_dash = [((xt, yt), s[1], s[2])]
                    if xt > 0:
                        s_dash.append(((xt - 1, yt), s[1], s[2]))
                    if yt > 0:
                        s_dash.append(((xt, yt - 1), s[1], s[2]))
                    if yt < 4:
                        s_dash.append(((xt, yt + 1), s[1], s[2]))
                    if xt < 4:
                        s_dash.append(((xt + 1, yt), s[1], s[2]))

                    for s_ in s_dash:
                        V_tmp += (GAMMA*V[s_] - 1)*Transition(s, a, s_)

                    V_new = max(V_new, V_tmp)
            V[s] = V_new
            max_diff = max(max_diff, abs(V_old - V[s]))
        max_diffList.append(max_diff)
        if max_diff < EPSILON:
            break
    iterationsList = [i for i in range(1, len(max_diffList) + 1)]
    plt.plot(iterationsList, max_diffList)
    plt.xlabel('Iterations')
    plt.ylabel('Max Norm Distance')
    title = 'Discount Factor ' + str(GAMMA)
    plt.title(title)
    filename = 'Plots/_' + str(GAMMA) + '.png'
    plt.savefig(filename)
    plt.clf()
    return (V, iterations)


GAMMAS = [0.01, 0.1, 0.5, 0.8, 0.99]

for g in GAMMAS:
    ValueIteration(g)
