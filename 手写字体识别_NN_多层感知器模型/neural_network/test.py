'''
拿石子游戏：
    假设有n个石子，A、B两人比赛，每人每次最少拿 min_take 个石子，最多拿 max_take 个石子
    A 先手，最后拿走石子的人获胜，问 A 在有 n 个石子时，能否获胜
'''

# 最少拿石数、最大拿石数
min_take = 1
max_take = 3


def fun(n, min_take, max_take):
    # 初始化字典，记录剩下多少石子时先手必赢
    #   eg. win[1]=True表示剩下1个石子时，先手必赢；win[4]=False表示剩下4个石子时先手必输
    win = dict()
    for i in range(1, n + 1):
        if min_take <= i <= max_take:
            win[i] = True
        elif i == min_take + max_take:
            win[i] = False
        else:
            win[i] = None

    def iswin(n):
        for i in range(min_take, max_take + 1):
            if win[n - i] == None:
                iswin(n - i)
            if win[n - i] == False:
                win[n] = True
                break
            else:
                win[n] = False
        return
    iswin(n)

    return win[n]


print(f'当有{10}个石子，且A先手时，A是否会赢：', fun(10, min_take, max_take))

