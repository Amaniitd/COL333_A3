# imports
import random

# temporary variables
EPSILON = 0.05
GAMMA = 0.9
passenger_loaction = (0, 0)
destination_loaction = (1, 1)

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
directon_dict = {'N': (0, 1), 'S': (0, -1), 'W': (+1, 0), 'E': (-1, 0)}


# Function for checking if there is a wall or not
def checkWall(xt, yt, d):
    rightWall = [(1, 1), (1, 0), (3, 0), (3, 1), (2, 3),
                 (2, 4)]  # wall encountered in moving left
    leftWall = [(0, 1), (0, 0), (2, 0), (2, 1), (1, 3),
                (1, 4)]  # wall encountered in moving right
    if d == 2:
        if ((xt, yt) in rightWall):
            return False
    elif d == 3:
        if ((xt, yt) in leftWall):
            return False
    return True


def nextStateUpdateHelper(xt, yt, a):
    tmpx = xt
    tmpy = yt
    if a == 'N':
        if yt < 4 and checkWall(xt, yt, 0):
            tmpy += 1
    elif a == 'S':
        if yt > 0 and checkWall(xt, yt, 1):
            tmpy -= 1
    elif a == 'W':
        if xt > 0 and checkWall(xt, yt, 2):
            tmpx -= 1
    elif a == 'E':
        if xt < 4 and checkWall(xt, yt, 3):
            tmpx += 1
    return (tmpx, tmpy)


def simulator(s, a):
    xt = s[0][0]
    yt = s[0][1]
    status = s[1]
    xp = s[2][0]
    yp = s[2][1]
    reward = 0
    next_state = ()
    if a == 'P':
        if s[0] == s[2]:
            reward = - 1
            next_state = (s[0], True, s[2])
        else:
            reward = -10
            next_state = s
    elif a == 'D':
        if status:
            reward = 20
            next_state = (s[0], False, s[2])
        else:
            reward = -10
            next_state = s
    elif a in Directions:
        reward = -1
        tmp_directions = ['N', 'S', 'E', 'W']
        tmp_directions.remove(a)
        tmp_int = random.randint(1, 20)
        if (tmp_int < 18):
            next_state = (nextStateUpdateHelper(xt, yt, a), s[1], s[2])
        elif tmp_int == 18:
            next_state = (nextStateUpdateHelper(
                xt, yt, tmp_directions[0]), s[1], s[2])
        elif tmp_int == 19:
            next_state = (nextStateUpdateHelper(
                xt, yt, tmp_directions[1]), s[1], s[2])
        else:
            next_state = (nextStateUpdateHelper(
                xt, yt, tmp_directions[2]), s[1], s[2])
    return (next_state, reward)


# Simulator Tester
# s = ((1, 4), False, (0, 0))
# print(simulator(s, 'N'))


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


# Initial Values
V = {}
for s in States:
    V[s] = 0


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
                if (s[0] == s[2] and s[1] == False):
                    V_tmp -= 1
                else:
                    V_tmp -= 10
                V_new = max(V_new, V_tmp)

            elif a == 'D':
                # Hardcoded for putdown actions
                V_tmp = GAMMA*V_old
                if (s[0] == s[2] and s[1]):
                    V_tmp += 20
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

    if max_diff < EPSILON:
        break

print(iterations)
