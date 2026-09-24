from calc_fbol import do_fbol


def main():
    starnames = ['HD27371','HD27697','HD28305','HD28307']

    for starname in starnames:
        print('Plotting Fbol for {}'.format(starname))
        do_fbol(starname)


if __name__ == '__main__':
    main()
