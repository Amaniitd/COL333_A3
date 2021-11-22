import random
Directions = ['N', 'S', 'E', 'W']


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
