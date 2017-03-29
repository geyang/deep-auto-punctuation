from termcolor import cprint, colored as c


def inc(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1


def precision_recall(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))

    keys = []
    true_p = {}
    p = {}
    all_p = {}
    for i in range(len(output)):

        inc(all_p, target[i])
        inc(p, output[i])
        if target[i] == output[i]:
            inc(true_p, output[i])

    precision = {k: (true_p[k] if k in true_p else 0) / all_p[k] for k in all_p.keys()}

    recall = {k: (true_p[k] if k in true_p else 0) / p[k] for k in p.keys()}

    return precision, recall, {"true_p": true_p, "p": p, "all_p": all_p}


def F_score(p, r):
    f_scores = {
        k: None if p[k] == 0 else (0 if r[k] == 0 else 2 / (1 / p[k] + 1 / r[k]))
        for k in p
    }
    return f_scores


def print_pc(o, target):
    """returns: 
        p<recision>, 
        r<ecall>, 
        f<-score>, 
        {"true_p", "p", "all_p"} """
    p, r, _ = precision_recall(o, target)
    f = F_score(p, r)

    for k in p.keys():
        cprint("Key: " + c(("  " + k)[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[k] * 100)[-5:], 'green') + '%' +
               "\tRecall: " + c("  {:.1f}".format((r[k] if k in r else 0) * 100)[-5:], 'green') + "%" +
               "\tF-Score: " + ("  N/A" if f[k] is None else (c("  {:.1f}".format(f[k] * 100)[-5:], "green") + "%"))
               )
    return p, r, f, _


if __name__ == "__main__" or True:
    test_o = ['<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<cap>', '<cap>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<cap>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', "'", '<nop>', '<nop>', '<cap>', '<nop>', '<nop>', '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', "'", '<nop>',
              '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>']
    test_target = ['<nop>', '<nop>', ',', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<cap>', '<cap>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '.', '<nop>',
                   '<cap>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', "'", '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', ',', '<nop>', '<nop>',
                   '<nop>', "'", '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', "'", '<nop>', '<nop>',
                   '<nop>', '<nop>', '<nop>', '<nop>', '<nop>', '<nop>']

    p, r, f, _ = print_pc(test_o, test_target)

    print('\n')
    for k in _:
        print(k + ':\t' + str(_[k]))
