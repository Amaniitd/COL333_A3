# imports
import random
import matplotlib.pyplot as plt
import numpy as np

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

# Function for calculating next state and reward - given current state and action taken


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

# Helper function for checking valid coordinates


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


def Transition(s, a, s_, destination_location):
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
        if NextState(s, a, destination_loaction)[0] == s_:
            return 1

    return 0


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
                    V_tmp += (GAMMA*V[s_] - 1)*Transition(s,
                                                          a, s_, destination_location)

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
            a = policies[s]
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
        if iterations > 100:
            break
        if max_diff < EPSILON:
            break
    return V


def PolicyEvaluationLinearAlg():
    return


def PolicyIteration(GAMMA, EPSILON, destinationLocation):
    policies = {}
    for s in States:
        policies[s] = 'N'
    old_policies = {}
    while (policies != old_policies):
        improved_V = PolicyEvaluation(
            policies, destinationLocation, GAMMA, EPSILON)
        old_policies = policies
        policies = OneStepLookAhead(improved_V, destinationLocation, GAMMA)
    return policies


def PolicyLoss(v1, v2):
    max_diff = 0
    for key in v1.keys():
        max_diff = max(max_diff, abs(v1[key] - v2[key]))
    return max_diff


def plotGeneratorForPartA2c(EPSILON, destinationLocation):
    GAMMAS = [0.01, 0.1, 0.5, 0.8, 0.99]
    for GAMMA in GAMMAS:
        print("Plotting for Discount Factor: " + str(GAMMA))
        policies = {}
        for s in States:
            policies[s] = 'N'
        old_policies = {}
        opt_v = {}
        i = 0
        while (policies != old_policies):
            i += 1
            if i > 30:
                break
            improved_V = PolicyEvaluation(
                policies, destinationLocation, GAMMA, EPSILON)
            old_policies = policies
            policies = OneStepLookAhead(
                improved_V, destinationLocation, GAMMA)
            opt_v = improved_V
        policies = {}
        for s in States:
            policies[s] = 'N'
        old_policies = {}
        iterations = 0
        PolicyLossList = []
        while (policies != old_policies):
            improved_V = PolicyEvaluation(
                policies, destinationLocation, GAMMA, EPSILON)
            old_policies = policies
            policies = OneStepLookAhead(improved_V, destinationLocation, GAMMA)
            iterations += 1
            if (iterations == 20):
                break
            pl = PolicyLoss(opt_v, improved_V)
            PolicyLossList.append(pl)
            if (pl < 0.05):
                break
        iterationsList = [i for i in range(1, len(PolicyLossList) + 1)]
        plt.plot(iterationsList, PolicyLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Policy Loss')
        title = 'Discount Factor ' + str(GAMMA)
        plt.title(title)
        filename = 'A_PolicyIteration_' + str(GAMMA) + '.png'
        plt.savefig(filename)
        plt.clf()


def printState(s):
    if s[1]:
        print("Taxi location -", s[0], "\tPassenger in taxi")
    else:
        print("Taxi location -",
              s[0], "\tPassenger not in taxi", "\tpassenger location -", s[2])


def First20StatesUsingPolicyEvaluation(PassengerLocation, TaxiLocation, DestinationLocation, GAMMA=0.9):
    policies = PolicyIteration(GAMMA, 0.05, DestinationLocation)
    s = (TaxiLocation, False, PassengerLocation)
    total_rewards = 0
    printState(s)
    for i in range(20):
        print("Step", i + 1, "Next action:", policies[s])
        (next_s, reward) = NextState(s, policies[s], DestinationLocation)
        total_rewards += reward
        if policies[s] == 'D':
            if s[0] == DestinationLocation and s[1]:
                print("Passenger is deposited at the destination location")
                print("Reward:", reward, "   Total reward:", total_rewards)
                return
        s = next_s
        printState(s)
        print("Reward:", reward, "   Total reward:", total_rewards)


# plotGeneratorForPartA2c(0.05, (4, 4))

# PolicyIteration(0.99, 0.05, (4, 4))


def RewardFunction(s, a, s_, destination_location):
    if a in Directions:
        return -1
    if a == 'P':
        if (s[0] == s[2]):
            return - 1
        else:
            return -10
    if a == 'D':
        if (s[0] == destination_location):
            if s[1]:
                return 20
            else:
                return -1
        else:
            return -10
    return 0


def PolicyEvaluationLinearAlg(policies, destination_location, GAMMA=0.9):
    l = len(States)
    tmp_v = np.zeros(l)
    T = np.zeros((l, l))
    R = np.zeros(l)
    for j in range(l):
        for k in range(l):
            T[j][k] = Transition(States[j], policies[States[j]],
                                 States[k], destination_location)
            R[j] += T[j][k] * \
                RewardFunction(States[j], policies[States[j]],
                               States[k], destination_location)
    tmp_v = np.linalg.solve(np.identity(l) - GAMMA*T, R)
    V = {}
    for i in range(l):
        V[States[i]] = tmp_v[i]
    return V


def PolicyIterationLinearAlg(GAMMA, destinationLocation):
    policies = {}
    for s in States:
        policies[s] = 'N'
    old_policies = {}
    while (policies != old_policies):
        improved_V = PolicyEvaluationLinearAlg(
            policies, destinationLocation, GAMMA)
        old_policies = policies
        policies = OneStepLookAhead(improved_V, destinationLocation, GAMMA)
    return policies


print(PolicyIterationLinearAlg(0.9, (4, 4)))
