min_lr = 1
max_lr = 2
lr_counter = 0
step = 7
lr_inc = 1
lr_delta = (max_lr-min_lr) / step
prev_lr = min_lr - (lr_delta * step * 100)

def getNextlr():
    global lr_inc 
    global lr_delta 
    global min_lr
    global max_lr
    global lr_counter
    global prev_lr

    prev_lr = prev_lr + ( lr_delta * lr_inc)
    if prev_lr >= max_lr:
        lr_inc = -1
        prev_lr = max_lr
    elif prev_lr < min_lr:
        lr_inc = 1

    return prev_lr




def main():
    print(lr_delta)
    for i in range(100):
        print(i,'\t',getNextlr())



if __name__=="__main__":
    main()

