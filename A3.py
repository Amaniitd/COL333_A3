# six actions indexed as 0 - north, 1 - east, 2-south, 3-west, 4 - pickup, 5 - putdown

import random
import math
import matplotlib.pyplot as plt
import numpy as np


# The code takes input for the question number. It also generate plots asked in sub-questions.
# The code will first ask for inputs related to the part that has to be run and then it will ask for required data as input.


"""
_________________________________   PART A   _______________________________________
"""
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


# Helper function
# for checking wall while travelling East and west
def checkWall(xt, yt, d):
    Wall_in_moving_west = [(1, 1), (1, 0), (3, 0), (3, 1), (2, 3),
                           (2, 4)]  # wall encountered in moving west
    Wall_in_moving_east = [(0, 1), (0, 0), (2, 0), (2, 1), (1, 3),
                           (1, 4)]  # wall encountered in moving east
    if d == 2:  # d is a variable for direction
        if ((xt, yt) in Wall_in_moving_west):
            return False
    elif d == 3:
        if ((xt, yt) in Wall_in_moving_east):
            return False
    return True


# Helper function
# for considering boundries and wall encounter
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
        if NextState(s, a, destination_location)[0] == s_:
            return 1

    return 0


# Value iteration function - default discount rate - 0.9 (GAMMA) and default EPSILON = 0.05
def ValueIteration(GAMMA=0.9, EPSILON=0.05, destination_location=(4, 4)):
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
                        V_tmp += (GAMMA*V[s_] - 1)*Transition(s,
                                                              a, s_, destination_location)

                V_new = max(V_new, V_tmp)
            V[s] = V_new
            max_diff = max(max_diff, abs(V_old - V[s]))

        if max_diff < EPSILON:
            break

    return (V, iterations)


def plotGeneratorForPartA2b(destination_location=(4, 4), EPSILON=0.05):
    GAMMAS = [0.01, 0.1, 0.5, 0.8, 0.99]
    for GAMMA in GAMMAS:
        print("Plotting for Discount Factor: " + str(GAMMA))
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
                            V_tmp += (GAMMA*V[s_] - 1)*Transition(s,
                                                                  a, s_, destination_location)

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
        filename = 'A_ValueIteration_' + str(GAMMA) + '.png'
        plt.savefig(filename)
        plt.clf()


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
                    V_tmp += (GAMMA*V[s_] - 1)*Transition(s,
                                                          a, s_, destination_location)

            V[s] = V_tmp
            max_diff = max(max_diff, abs(V_old - V[s]))

        if max_diff < EPSILON:
            break
    return V


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


def printState(s):
    if s[1]:
        print("Taxi location -", s[0], "\tPassenger in taxi")
    else:
        print("Taxi location -",
              s[0], "\tPassenger not in taxi", "\tpassenger location -", s[2])


def First20States(PassengerLocation, TaxiLocation, DestinationLocation, GAMMA=0.9):
    V = ValueIteration(GAMMA, 0.05, DestinationLocation)[0]
    policies = OneStepLookAhead(V, DestinationLocation, GAMMA)
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


def Simulator(PassengerLocation, TaxiLocation, DestinationLocation, GAMMA=0.9):
    V = ValueIteration(GAMMA, 0.05, DestinationLocation)[0]
    policies = OneStepLookAhead(V, DestinationLocation, GAMMA)
    s = (TaxiLocation, False, PassengerLocation)
    total_rewards = 0
    printState(s)
    for i in range(100):
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


"""
__________________________________   PART B   _______________________________________

"""
Q = {}  # Quality is initialised as global variable so that all functions can access it
for x in range(5):
    for y in range(5):
        for px in range(5):
            for py in range(5):
                Q[(x, y, px, py, False)] = [0, 0, 0, 0, 0, 0]
                Q[(x, y, px, py, True)] = [0, 0, 0, 0, 0, 0]

random.seed(1)


def nextState(state, action):  # returns the deterministic nextState on navigation actions
    next_state = list(state)
    x = state[0]
    y = state[1]

    if(action == 0):  # north
        if(y != 4):
            next_state[1] += 1
    elif(action == 1):  # east
        if not (x == 4 or (y < 2 and (x == 0 or x == 2)) or (y > 2 and x == 1)):
            next_state[0] += 1
    elif(action == 2):  # south
        if (y != 0):
            next_state[1] -= 1
    elif (action == 3):  # west
        if not (x == 0 or (y < 2 and (x == 1 or x == 3)) or (y > 2 and x == 2)):
            next_state[0] -= 1
    if(state[4]):
        next_state[2] = next_state[0]
        next_state[3] = next_state[1]

    return tuple(next_state)


def nextStateReward(state, action, dest):  # state = list [x,y], action = 0 - 5
    # pickup, putdown are deterministic
    if(action == 4):  # pickup
        if (state[0] == state[2] and state[1] == state[3]):  # pickup
            next_state = list(state)
            next_state[4] = True
            return (tuple(next_state), -1)
        else:  # wrong position pick
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -10)

    if(action == 5):  # put down
        # desination reached, passenger dropped
        if (state[0] == dest[0] and state[1] == dest[1] and state[4]):
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), 20)
        # destination is reached but passenger not in taxi
        elif (state[0] == state[2] and state[1] == state[3]):
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -1)
        else:  # not the destination, penalize the drop
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -10)
    randm = random.random()
    # change the actual navigation with given probability
    if(randm <= 0.85):
        action = action
    elif(randm <= 0.9):
        action = (action+1) % 4
    elif(randm <= 0.95):
        action = (action+2) % 4
    elif(randm <= 1):
        action = (action+3) % 4

    next_state = nextState(state, action)
    return (next_state, -1)


def argmax(l):
    m = max(l)
    return l.index(m)


# calculates the discounted reward sum from start to dest
def rewardSum(state, discount, gamma, dest):
    q = Q[state]
    if(discount < gamma**30):  # if recursion depth 30 then return/ should perform within it
        return 0
    else:
        best_action = argmax(q)

        (next_state, reward) = nextStateReward(state, best_action, dest)
        if (reward == 20):  # if dest reached stop
            return discount*20

        next = rewardSum(next_state, discount*gamma, gamma, dest)
        return discount*reward + next


DEPOS = [[0, 0], [0, 4], [3, 0], [4, 4]]


# Fills the global Q
def qLearning(alpha=0.25, gamma=0.99, epsilon=0.1, dest=[3, 0]):
    print('Q learning')
    random.seed(1)
    rewardUpdate = []
    reward_sum = 0  # temp variable for storing the rewards after each step
    reached = 0  # no of times correctly dropped
    picks = 0  # no of times picked
    drops = 0  # no of times dropped
    for i in range(4000):  # episodes
        passenger = DEPOS[random.randint(0, 3)]
        start = (random.randint(0, 4), random.randint(
            0, 4), passenger[0], passenger[1], False)
        state = start
        for j in range(500):  # steps
            if (random.random() <= epsilon):
                action = random.randint(0, 5)
            else:
                action = argmax(Q[state])

            (next_state, reward) = nextStateReward(state, action, dest)
            # QLearning update equation
            Q[state][action] = (1-alpha)*Q[state][action] + \
                alpha*(reward + gamma*max(Q[next_state]))

            if(state[4] == False and next_state[4]):
                picks += 1
            if(state[4] and not next_state[4]):
                drops += 1
            if(reward == 20):  # dest reached
                reached += 1
                break
            state = next_state

        reward_sum += rewardSum(start, 1, gamma, dest)
        if(i % 10 == 9):
            # append the average over 10 episodes into a list
            rewardUpdate.append(reward_sum/10)
            reward_sum = 0
    print('picks = ', picks, 'drops = ', drops, 'correct drops = ', reached)
    print('max discounted reward sum = ', max(rewardUpdate))
    return rewardUpdate  # list of dscuonted reward sum


def decayQLearning(alpha, gamma, epsilon, dest):
    print('Q learning with decaying epsilon')
    random.seed(1)
    rewardUpdate = []
    reward_sum = 0
    reached = 0
    picks = 0
    drops = 0
    Epsilon = epsilon
    iters = 0
    for i in range(4000):  # episodes
        passenger = DEPOS[random.randint(0, 3)]
        start = (random.randint(0, 4), random.randint(
            0, 4), passenger[0], passenger[1], False)
        state = start
        for j in range(500):  # steps
            # exponential decay of epsilon
            epsilon = Epsilon*math.exp(-0.00001*iters)
            iters += 1
            if (random.random() <= epsilon):
                action = random.randint(0, 5)
            else:
                action = argmax(Q[state])

            (next_state, reward) = nextStateReward(state, action, dest)

            Q[state][action] = (1-alpha)*Q[state][action] + \
                alpha*(reward + gamma*max(Q[next_state]))

            if(state[4] == False and next_state[4]):
                picks += 1
            if(state[4] and not next_state[4]):
                drops += 1
            if(reward == 20):  # dest reached
                reached += 1
                break
            state = next_state

        reward_sum += rewardSum(start, 1, gamma, dest)
        if(i % 10 == 9):
            rewardUpdate.append(reward_sum/10)
            reward_sum = 0
    print('picks = ', picks, 'drops = ', drops, 'correct drops = ', reached)
    print('max discounted reward sum = ', max(rewardUpdate))
    return rewardUpdate


def Sarsa(alpha=0.25, gamma=0.99, epsilon=0.1, dest=[3, 0]):
    print('SARSA learning')
    random.seed(1)
    rewardUpdate = []
    reward_sum = 0
    reached = 0
    picks = 0
    drops = 0
    for i in range(4000):  # episodes
        # choose a random location of passenger
        passenger = DEPOS[random.randint(0, 3)]
        start = (random.randint(0, 4), random.randint(0, 4),
                 passenger[0], passenger[1], False)  # random start state
        state = start

        for j in range(500):  # steps
            if (random.random() <= epsilon):
                action = random.randint(0, 5)
            else:
                action = argmax(Q[state])

            (next_state, reward) = nextStateReward(state, action, dest)

            if(random.random() <= epsilon):  # selecting the action in next state
                next_action = random.randint(0, 5)
            else:
                next_action = argmax(Q[next_state])

            Q[state][action] = (1-alpha)*Q[state][action] + \
                alpha*(reward + gamma*Q[next_state][next_action])

            if(state[4] == False and next_state[4]):
                picks += 1
            if(state[4] and not next_state[4]):
                drops += 1
            if(reward == 20):  # dest reached
                reached += 1
                break
            state = next_state

        # calculate the discounted reward sum from the start
        reward_sum += rewardSum(start, 1, gamma, dest)
        if(i % 10 == 9):
            rewardUpdate.append(reward_sum/10)
            reward_sum = 0
    print('picks = ', picks, 'drops = ', drops, 'correct drops = ', reached)
    print('max discounted reward sum = ', max(rewardUpdate))
    return rewardUpdate


def decaySarsa(alpha=0.25, gamma=0.99, epsilon=0.1, dest=[3, 0]):
    print('SARSA learning with decaying epsilon')
    random.seed(1)
    rewardUpdate = []
    reward_sum = 0
    reached = 0
    picks = 0
    drops = 0
    Epsilon = epsilon
    iters = 0
    for i in range(4000):  # episodes
        passenger = DEPOS[random.randint(0, 3)]
        start = (random.randint(0, 4), random.randint(
            0, 4), passenger[0], passenger[1], False)
        state = start

        for j in range(500):  # steps
            epsilon = Epsilon*(1 - math.exp(-0.00001*iters))
            iters += 1
            if (random.random() <= epsilon):
                action = random.randint(0, 5)
            else:
                action = argmax(Q[state])

            (next_state, reward) = nextStateReward(state, action, dest)

            if(random.random() <= epsilon):
                next_action = random.randint(0, 5)
            else:
                next_action = argmax(Q[next_state])

            Q[state][action] = (1-alpha)*Q[state][action] + \
                alpha*(reward + gamma*Q[next_state][next_action])

            if(state[4] == False and next_state[4]):
                picks += 1
            if(state[4] and not next_state[4]):
                drops += 1
            if(reward == 20):  # dest reached
                reached += 1
                break
            state = next_state

        reward_sum += rewardSum(start, 1, gamma, dest)
        if(i % 10 == 9):
            rewardUpdate.append(reward_sum/10)
            reward_sum = 0
    print('picks = ', picks, 'drops = ', drops, 'correct drops = ', reached)
    print('max discounted reward sum = ', max(rewardUpdate))
    return rewardUpdate


def plot(X, Y, title='', xlabel='', ylabel=''):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(X, Y, c='firebrick', s=0.4)
    plt.plot(X, Y, c='navy', linewidth=0.3)
    plt.show()


dest = [3, 0]  # destination used for this experiment


def varyMethod(method='Q'):
    if(method == 'Q'):
        Y = qLearning(0.25, 0.99, 0.1, [3, 0])
    elif(method == 'decayQ'):
        Y = decayQLearning(0.25, 0.99, 0.1, [3, 0])
    elif(method == 'Sarsa'):
        Y = Sarsa(0.25, 0.99, 0.1, [3, 0])
    elif(method == 'decaySarsa'):
        Y = decaySarsa(0.25, 0.99, 0.1, [3, 0])
    X = [10*i for i in range(1, len(Y)+1)]
    plot(X, Y, method + ' learning', 'episodes', 'Discounted reward sum')


# executes the policy with following locations and gives the discounted sum reward, steps taken to reah [3,0]
def executePolicy():
    decayQLearning(0.25, 0.99, 0.1, [3, 0])
    start = [[4, 4], [2, 2], [0, 0], [1, 4], [3, 1]]
    passenger = [[0, 0], [0, 4], [4, 4], [4, 4], [0, 4]]

    for i in range(len(start)):
        state = (start[i][0], start[i][1], passenger[i]
                 [0], passenger[i][1], False)
        steps = 0
        rewardSum = 0
        picks = 0
        drops = 0
        while(30):
            qs = Q[state]
            action = qs.index(max(qs))
            if(action == 4):
                picks += 1
            elif(action == 5):
                drops += 1
            (state, reward) = nextStateReward(state, action, [3, 0])
            rewardSum += (0.99**steps)*reward
            steps += 1
            if(reward == 20):
                break
        print('picks = ', picks, 'drops = ', drops,
              'discounted reward sum = ', rewardSum, 'steps = ', steps)


def varyEpsilon():
    Epsilon = [0, 0.05, 0.1, 0.5, 0.9]
    for e in Epsilon:
        Q.clear()
        for x in range(5):
            for y in range(5):
                for px in range(5):
                    for py in range(5):
                        Q[(x, y, px, py, False)] = [0, 0, 0, 0, 0, 0]
                        Q[(x, y, px, py, True)] = [0, 0, 0, 0, 0, 0]

        Y = qLearning(0.1, 0.99, e, dest)
        X = [10*i for i in range(1, len(Y)+1)]
        plot(X, Y, f'Epsilon = {e} in Q learning',
             'episodes', 'Discounted rewards sum')


def varyAlpha():
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5]

    for a in alpha:
        Q.clear()
        for x in range(5):
            for y in range(5):
                for px in range(5):
                    for py in range(5):
                        Q[(x, y, px, py, False)] = [0, 0, 0, 0, 0, 0]
                        Q[(x, y, px, py, True)] = [0, 0, 0, 0, 0, 0]
        Y = qLearning(a, 0.99, 0.1, dest)
        X = [10*i for i in range(1, len(Y)+1)]
        plot(X, Y, f'alpha = {a} in Q learning',
             'episodes', 'Discounted rewards sum')


# varyMethod()
# executePolicy()
# varyEpsilon()
# varyAlpha()
########################### part B 5#######################
Q10 = {}


random.seed(1)

left_walls = [(3, 6), (3, 7), (3, 8), (3, 9)]
left_walls.extend([(4, 0), (4, 1), (4, 2), (4, 3)])
left_walls.extend([(6, 4), (6, 5), (6, 6), (6, 7)])
left_walls.extend([(1, 0), (1, 1), (1, 2), (1, 3)])
left_walls.extend([(8, 0), (8, 1), (8, 2), (8, 3)])
left_walls.extend([(8, 6), (8, 7), (8, 8), (8, 9)])

right_walls = [(0, 0), (0, 1), (0, 2), (0, 3)]
right_walls.extend([(3, 0), (3, 1), (3, 2), (3, 3)])
right_walls.extend([(7, 0), (7, 1), (7, 2), (7, 3)])
right_walls.extend([(2, 6), (2, 7), (2, 8), (2, 9)])
right_walls.extend([(7, 6), (7, 7), (7, 8), (7, 9)])
right_walls.extend([(5, 4), (5, 5), (5, 6), (5, 7)])


def nextState10(state, action):  # returns the deterministic nextState
    next_state = list(state)
    x = state[0]
    y = state[1]

    if(action == 0):  # north
        if(y != 9):
            next_state[1] += 1
    elif(action == 1):  # east
        if not (x == 9 or ((x, y) in right_walls)):
            next_state[0] += 1
    elif(action == 2):  # south
        if (y != 0):
            next_state[1] -= 1
    elif (action == 3):  # west
        if not (x == 0 or ((x, y) in left_walls)):
            next_state[0] -= 1
    if(state[4]):  # the passenger is picked so the location changes correspondingly
        next_state[2] = next_state[0]
        next_state[3] = next_state[1]

    return tuple(next_state)


# state = list [x,y], action = 0 - 5
def nextStateReward10(state, action, dest):
    # pickup, putdown will remain as it is, since they are deterministic
    if(action == 4):
        if (state[0] == state[2] and state[1] == state[3]):  # pickup
            next_state = list(state)
            next_state[4] = True
            return (tuple(next_state), -1)
        else:  # wrong position pick
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -10)

    if(action == 5):  # put down
        if (state[0] == dest[0] and state[1] == dest[1] and state[4]):
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), 20)
        elif (state[0] == state[2] and state[1] == state[3]):
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -1)
        else:
            next_state = list(state)
            next_state[4] = False
            return (tuple(next_state), -10)
    randm = random.random()
    # change the actual navigation with given probability
    if(randm <= 0.85):
        action = action
    elif(randm <= 0.9):
        action = (action+1) % 4
    elif(randm <= 0.95):
        action = (action+2) % 4
    elif(randm <= 1):
        action = (action+3) % 4

    next_state = nextState10(state, action)
    return (next_state, -1)


def argmax10(l):
    m = max(l)
    return l.index(m)


def rewardSum10(state, discount, gamma, dest):
    q = Q10[state]
    if(discount < gamma**100):
        return 0
    else:
        best_action = argmax10(q)

        (next_state, reward) = nextStateReward10(state, best_action, dest)
        if (reward == 20):
            return discount*20
        next = rewardSum10(next_state, discount*gamma, gamma, dest)

        return discount*reward + next


DEPOS10 = [[0, 1], [0, 9], [3, 6], [4, 0], [5, 9], [6, 5], [8, 9], [9, 0]]

# we will be using Q learning for 10x10 problem because the convergence is faster (then SARSA) and consistent (than decay epsilon)


def qLearning10(alpha, gamma, epsilon, dest):
    print('Q learning for 10x10')
    random.seed(1)
    for x in range(10):
        for y in range(10):
            for px in range(10):
                for py in range(10):
                    Q10[(x, y, px, py, False)] = [0, 0, 0, 0, 0, 0]
                    Q10[(x, y, px, py, True)] = [0, 0, 0, 0, 0, 0]
    rewardUpdate = []
    reward_sum = 0
    for i in range(10000):  # episodes
        passenger = DEPOS10[random.randint(0, 7)]
        start = (random.randint(0, 9), random.randint(
            0, 9), passenger[0], passenger[1], False)
        state = start
        for j in range(1000):  # steps
            if (random.random() <= epsilon):
                action = random.randint(0, 5)
            else:
                action = argmax10(Q10[state])

            (next_state, reward) = nextStateReward10(state, action, dest)

            Q10[state][action] = (1-alpha)*Q10[state][action] + \
                alpha*(reward + gamma*max(Q10[next_state]))

            # if(i==4000):
            #    print(state, action, next_state)
            if(reward == 20):
                break
            state = next_state

        reward_sum += rewardSum10(start, 1, gamma, dest)
        if(i % 10 == 9):
            rewardUpdate.append(reward_sum/10)
            reward_sum = 0
    # print(max(rewardUpdate))
    return rewardUpdate


def plot10x10():
    Y = qLearning10(0.3, 0.99, 0.1, [9, 0])
    X = [10*i for i in range(1, len(Y)+1)]
    plot(X, Y, f' Q learning', 'episodes', 'Discounted rewards sum')


def execute10():
    start = [[1, 4], [2, 3], [9, 9], [4, 4], [0, 0]]
    passenger = [[3, 6], [6, 5], [0, 1], [0, 9], [5, 9]]
    dest = [[9, 0], [0, 1], [8, 9], [6, 5], [3, 6]]

    for i in range(len(start)):
        qLearning10(0.3, 0.99, 0.1, dest[i])
        state = (start[i][0], start[i][1], passenger[i]
                 [0], passenger[i][1], False)
        steps = 0
        rewardSum = 0
        picks = 0
        drops = 0
        while(steps < 100):
            qs = Q10[state]
            action = qs.index(max(qs))
            if(action == 4):
                picks += 1
            elif(action == 5):
                drops += 1
            (state, reward) = nextStateReward10(state, action, dest[i])
            rewardSum += (0.99**steps)*reward
            steps += 1
            if(reward == 20):
                break
        print('picks = ', picks, 'drops = ', drops,
              'discounted reward sum = ', rewardSum, 'steps = ', steps)


def Run():
    part = input('Type the part you want to run: ')
    if part == 'A':
        PROBLEM = input('\nType the part to execute (1, 2, 3) = ')
        if (PROBLEM == "1"):
            pass
        elif PROBLEM == "2":
            SUBPROBLEM = input('\nType the part to execute (a, b, c) = ')
            if SUBPROBLEM == 'b':
                E = input('\nType epsilon value (dafault 0.05): ')
                if E == "":
                    plotGeneratorForPartA2b()
                else:
                    plotGeneratorForPartA2b(E)
            elif SUBPROBLEM == 'c':
                pX = int(input(
                    '\n Enter Passenger location\'s X coordinate : '))
                pY = int(
                    input('\n Enter Passenger location\'s Y coordinate : '))
                destX = int(input(
                    '\n Enter Passenger destination\'s X coordinate : '))
                destY = int(input(
                    '\n Enter Passenger destination\'s Y coordinate : '))
                taxiX = int(input('\n Enter taxi location\'s X coordinate : '))
                taxiY = int(input('\n Enter taxi location\'s Y coordinate : '))
                G = float(input('\nEnter Discount Factor : '))
                First20States((pX, pY), (taxiX, taxiY), (destX, destY), G)

            elif SUBPROBLEM == 'a':
                print("To run the value iteration function:")
                destX = int(input(
                    '\n Enter Passenger destination\'s X coordinate : '))
                destY = int(input(
                    '\n Enter Passenger destination\'s Y coordinate : '))
                G = float(input('\nEnter Discount Factor : '))
                E = float(input('\n Enter maxmimum error allowed : '))
                (pol, NoOfIter) = ValueIteration(G, E, (destX, destY))
                print(pol)
                print("\nNumber of iterations:", NoOfIter)
        elif PROBLEM == '3':
            plotGeneratorForPartA2c(0.05, (4, 4))

    elif part == 'B':
        PROBLEM = input('\nType the part to execute (2,3,4,5) = ')
        if(PROBLEM == '2'):
            METHOD = input(
                '\nType the method to use (Q, decayQ, Sarsa, decaySarsa) = ')
            varyMethod(METHOD)
        elif(PROBLEM == '3'):
            executePolicy()
        elif(PROBLEM == '4'):
            p2 = input(
                "\nTo vary epsilon press 'e', to vary alpha press 'a' = ")
            if(p2 == 'e'):
                varyEpsilon()
            elif(p2 == 'a'):
                varyAlpha()
        if(PROBLEM == '5'):
            p2 = input(
                "\nTo obtain the learning plot press 'p', to execute the policy press 'e' = ")
            if(p2 == 'p'):
                plot10x10()
            elif(p2 == 'e'):
                execute10()


Run()
