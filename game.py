import random
import curses
import time

# 初始化游戏窗口
screen = curses.initscr()
curses.curs_set(0)  # 隐藏光标
sh, sw = screen.getmaxyx()  # 窗口高、宽
w = curses.newwin(sh, sw, 0, 0)
w.keypad(1)  # 开启键盘监听
w.timeout(100)  # 控制游戏速度（数值越小越快）

# 蛇的初始状态
snake_x = sw // 4
snake_y = sh // 2
snake = [
    [snake_y, snake_x],
    [snake_y, snake_x - 1],
    [snake_y, snake_x - 2]
]

# 食物初始位置
food = [sh // 2, sw // 2]
w.addch(int(food[0]), int(food[1]), curses.ACS_PI)  # 画食物

# 初始方向
key = curses.KEY_RIGHT

# 游戏主循环
while True:
    next_key = w.getch()
    key = key if next_key == -1 else next_key

    # 游戏结束判断：撞墙 / 撞到自己
    head = snake[0]
    if (head[0] in [0, sh-1] or head[1] in [0, sw-1] or
            head in snake[1:]):
        time.sleep(1)
        curses.endwin()
        print(f"游戏结束！最终得分：{len(snake) - 3}")
        quit()

    # 新头部位置
    new_head = [snake[0][0], snake[0][1]]

    # 方向控制
    if key == curses.KEY_DOWN:
        new_head[0] += 1
    if key == curses.KEY_UP:
        new_head[0] -= 1
    if key == curses.KEY_LEFT:
        new_head[1] -= 1
    if key == curses.KEY_RIGHT:
        new_head[1] += 1

    snake.insert(0, new_head)

    # 吃到食物
    if snake[0] == food:
        food = None
        while food is None:
            nf = [
                random.randint(1, sh - 2),
                random.randint(1, sw - 2)
            ]
            food = nf if nf not in snake else None
        w.addch(food[0], food[1], curses.ACS_PI)
    else:
        # 没吃到，删除尾部
        tail = snake.pop()
        w.addch(int(tail[0]), int(tail[1]), ' ')

    # 画蛇
    w.addch(int(snake[0][0]), int(snake[0][1]), curses.ACS_CKBOARD)

    # 显示分数
    w.addstr(0, 2, f'得分: {len(snake) - 3} ')