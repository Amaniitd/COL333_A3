# imports
import random

# temporary variables


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
Colors = {'R': (0, 4), 'G': (4, 4), 'B': (3, 0), 'Y': (0, 0)}
ColorsList = [(0, 4), (4, 4), (3, 0), (0, 0)]

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


def NextState(s, a, destination_location):
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
        if s[0] == destination_location:
            if status:
                reward = 20
                next_state = (s[0], False, s[0])
            else:
                reward = -1
                next_state = s
        else:
            reward = -10
            next_state = (s[0], False, s[2])
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

def validPosition(x, y):
    if x < 0:
        return False
    if x > 4:
        return False
    if y < 0:
        return False
    if y > 4:
        return False
    return True

# Transition Function


def Transition(s, a, s_):
    xt = s[0][0]
    yt = s[0][1]
    x2 = s_[0][0]
    y2 = s_[0][1]
    if validPosition(xt, yt) == False:
        return 0
    if validPosition(x2, y2) == False:
        return 0
    if a not in actions:
        return 0
    if a in Directions:
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
    elif a == 'P' or a == 'D':
        if NextState(s, a)[0] == s_:
            return 1

    return 0

# Tester for Transition Function
# s = ((1, 4), False, (0, 0))
# s_ = ((1, 4), False, (0, 0))
# a = 'E'
# print(Transition(s, a, s_))


def ValueIteration(destination_location, GAMMA=0.9, EPSILON=0.05):
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
            V_new = float('-inf')
            for a in actions:
                # Hardcoded for pickup actions
                if a == 'P':
                    if (s[0] == s[2]):
                        V_tmp = GAMMA*V[(s[0], True, s[2])]
                        V_tmp -= 1
                    else:
                        V_tmp = GAMMA*V_old
                        V_tmp -= 10

                elif a == 'D':
                    # Hardcoded for putdown actions
                    V_tmp = GAMMA*V_old
                    if (s[0] == destination_location):
                        V_tmp = GAMMA*V[(s[0], False, s[2])]
                        if s[1]:
                            V_tmp += 20
                        else:
                            V_tmp -= 1
                    else:
                        V_tmp -= 10
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

    return (V, iterations)


def OneStepLookAhead(V, destination_location, GAMMA=0.9):
    Policies = {}
    for s in States:
        V_old = V[s]
        V_new = float('-inf')
        best_a = 'N'
        for a in actions:
            if a == 'P':
                if (s[0] == s[2]):
                    V_tmp = GAMMA*V[(s[0], True, s[2])]
                    V_tmp -= 1
                else:
                    V_tmp = GAMMA*V_old
                    V_tmp -= 10

            elif a == 'D':
                V_tmp = GAMMA*V_old
                if (s[0] == destination_location):
                    V_tmp = GAMMA*V[(s[0], False, s[2])]
                    if s[1]:
                        V_tmp += 20
                    else:
                        V_tmp -= 1
                else:
                    V_tmp -= 10

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

            if (V_tmp > V_new):
                V_new = V_tmp
                best_a = a
        Policies[s] = best_a
    return Policies


def PolicyEvaluation(policies, destination_location, GAMMA=0.9, EPSILON=0.05):
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
            a = policies(s)
            # Hardcoded for pickup actions
            if a == 'P':
                V_tmp = GAMMA*V_old
                if (s[0] == s[2]):
                    V_tmp = GAMMA*V[(s[0], True, s[2])]
                    V_tmp -= 1
                else:
                    V_tmp -= 10

            elif a == 'D':
                # Hardcoded for putdown actions
                V_tmp = GAMMA*V_old
                if (s[0] == destination_location):
                    V_tmp = GAMMA*V[(s[0], False, s[2])]
                    if s[1]:
                        V_tmp += 20
                    else:
                        V_tmp -= 1
                else:
                    V_tmp -= 10
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

            V[s] = V_tmp
            max_diff = max(max_diff, abs(V_old - V[s]))

        if max_diff < EPSILON:
            break
    return V



def PolicyEvaluationLinearAlg():
    return


def PolicyIteration(GAMMA, EPSILON, destinationLocation):
    policies = {}
    for s in States:
        policies[s] = random.choice(actions)
    new_policies = {}
    while (policies == new_poicies):
        improved_V = PolicyEvaluation(new_policies, destinationLocation, GAMMA, EPSILON)
        policies = new_policies
        new_poicies = OneStepLookAhead(improved_V)
    return new_policies


def printState(s):
    if s[1]:
        print("Taxi location -", s[0], "\tPassenger in taxi")
    else:
        print("Taxi location -",
              s[0], "\tPassenger not in taxi", "\tpassenger location -", s[2])


def First20States(PassengerLocation, TaxiLocation, DestinationLocation, GAMMA=0.9):
    V = ValueIteration(DestinationLocation, GAMMA)[0]
    policies = OneStepLookAhead(V, DestinationLocation, GAMMA)
    s = (TaxiLocation, False, PassengerLocation)
    printState(s)
    for i in range(20):
        print("step: " + str(i + 1))
        (next_s, reward) = NextState(s, policies[s], DestinationLocation)

        if policies[s] == 'D':
            if s[0] == DestinationLocation and s[1]:
                print("Passenger is deposited at the destination location")
                return
        s = next_s
        printState(s)


First20States((0, 0), (0, 3), (4, 4))
