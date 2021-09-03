import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import argparse
import os
from tqdm import tqdm


MINUTES_AHEAD = 5
SRATE = 100
LABELDICT = {'none': 0, 'event': 1, 'semi': 2}


def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_segments(vals):
    # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
    i = 0

    idxdict = {
        'none': [],     # non_event (negative)
        'event': [],    # event (positive)
        'semi': []      # semi-event (semi-positve)
    }

    # 20 sec values (2000)
    xs = []

    # 1: if all of 2sec moving average values from +5 to +6 min are lower than 65 mmHg
    # 2: if any of 2sec moving average values from  0 to +6 min are lower than 65 mmHg
    ys = []

    skip_time = 0

    while i < len(vals) - SRATE * (20 + (1 + MINUTES_AHEAD) * 60):
        segx = vals[i: i + SRATE * 20]
        segy = vals[i + SRATE * (20 + MINUTES_AHEAD * 60): i + SRATE * (20 + (1 + MINUTES_AHEAD) * 60)]
        segz = vals[i: i + SRATE * (20 + MINUTES_AHEAD * 60)]

        # 결측값 10% 이상이면
        if np.mean(np.isnan(segx)) > 0.0 or \
                np.mean(np.isnan(segy)) > 0.0 or \
                np.max(segx) > 200 or np.min(segx) < 20 or \
                np.max(segy) > 200 or np.min(segy) < 20 or \
                np.max(segx) - np.min(segx) < 30 or \
                np.max(segy) - np.min(segy) < 30 or \
                (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
            i += SRATE  # 1 sec 씩 전진
            skip_time += 1
            continue

        # 출력 변수
        segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
        segz = moving_average(segz, 2 * SRATE)

        event = None      # ignore
        if np.nanmax(segy) < 65:
            event = 'event'     # positive
        elif np.nanmin(segy) < 65 or np.nanmin(segz) < 65:
            event = 'semi'      # semi-positive
        elif np.nanmin(segy) > 65:
            event = 'none'      # negative

        if event:
            xs.append(segx)
            ys.append(LABELDICT[event])
            idxdict[event].append(i)

        i += 20 * SRATE  # 30sec

    return xs, ys, idxdict, skip_time


def read_index(filename):
    keys = ['event', 'semi', 'none']

    idxdict = {}
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            idxdict[keys[i]] = [int(v) for v in line.strip().split(',') if v != '']

    return idxdict


def analyze_and_save_stats(listfile, outfile):
    df = pd.read_csv(listfile)
    print(len(df), 'data is listed.')

    with open(outfile, 'wt') as fp:
        fp.write('caseid,age,sex,weight,height,bmi,xmean,nsample,nevent,nsemi\n')

        for i, row in df.iterrows():
            print(f'{i}, ', end='')
            print(', '.join([f'{j}={v}' for j, v in row.iteritems()]), end=' ... ')

            id = row['caseid']
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            height = row['height']

            bmi = weight / (height / 100) ** 2

            vals = pd.read_csv(f'./data/train_data/{id}.csv', header=None).values.flatten()
            xs, _, idxdict, nskips = get_segments(vals)

            nsample = len(xs)
            nevent = len(idxdict['event'])
            nsemi = len(idxdict['semi'])
            nnone = len(idxdict['none'])

            if nsample > 0:
                xs = np.asarray(xs)
                x_mean = np.mean(xs)
            else:
                x_mean = -1

            print(f'num of samples: {nsample}, event: {nevent}, semi: {nsemi}, neg: {nnone}, skips: {nskips}, mean: {x_mean:.2f}',
                  end=' ... ')

            resstr = f'{id},{age},{sex},{weight},{height},{bmi:.2f},{x_mean:.2f},{nsample},{nevent},{nsemi}\n'
            fp.write(resstr)

            print('saving', end=' ... ')

            idxfile = f'./data/train_data_seg_index/{id}.csv'
            with open(idxfile, 'wt') as fpp:
                for key in ['event', 'semi', 'none']:
                    fpp.write(','.join([str(v) for v in idxdict[key]]))
                    fpp.write('\n')

            print('done')

    print('finished saving stats and index files.')


def save_dataset(listfile, savedir, istestdata):
    df = pd.read_csv(listfile)
    print(len(df), 'data is loaded.')

    if istestdata:
        x_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            height = row['height']
            bmi = weight / (height / 100) ** 2

            header = np.asarray([age, sex == 'F', sex == 'M', weight, height, bmi], dtype=np.float)
            vals = row[4:]

            x = np.hstack([header, vals])
            x_list.append(x)

        x_test = np.asarray(x_list, dtype=np.float32)
        np.savez_compressed(os.path.join(savedir, 'x_test.npz'), x_test)

    else:
        x_list = []
        y_list = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            id = row['caseid']
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            height = row['height']
            bmi = row['bmi']

            header = np.asarray([age, sex == 'F', sex == 'M', weight, height, bmi], dtype=np.float)

            vals = pd.read_csv(f'./data/train_data/{id}.csv', header=None).values.flatten()

            idx_dict = read_index(f'./data/train_data_seg_index/{id}.csv')

            event_exists = len(idx_dict['event']) > 0

            keys = ['event', 'semi', 'none']

            for key in keys:
                for j in idx_dict[key]:
                    x = vals[j: j + SRATE * 20]
                    x = np.hstack([header, x])
                    x_list.append(x)

                    y = np.array([int(event_exists), LABELDICT[key]], dtype=np.int)
                    y_list.append(y)

        x_train = np.asarray(x_list, dtype=np.float32)
        np.savez_compressed(os.path.join(savedir, 'x_train.npz'), x_train)
        x_train = None

        y_train = np.asarray(y_list, dtype=np.int)
        np.savez_compressed(os.path.join(savedir, 'y_train.npz'), y_train)
        y_train = None

    print('done')


def show(id, listfile=None):
    if listfile is not None:
        # load meta file first
        df = pd.read_csv(listfile, index_col='caseid')
        print('id:', id)
        print(df.loc[id])

    # load data file
    vals = pd.read_csv('./data/train_data/{}.csv'.format(id), header=None).values.flatten()
    print(len(vals))

    # 2 sec. moving average
    mvals = moving_average(vals, n=200)

    # plot
    plt.figure(figsize=(15, 5))
    plt.plot(vals, color='gray', linewidth=0.5, alpha=0.5)
    plt.plot(mvals, color='orange')
    plt.axhline(65, color='red', linestyle=':', linewidth=2)
    plt.ylim(0, 300)
    plt.xlim(0)

    # grid
    intervals = 5 * 60 * 100
    loc = plticker.MultipleLocator(base=intervals)
    ax = plt.gca()
    ax.xaxis.set_major_locator(loc)
    ax.grid(which='major', axis='x', linestyle='-')
    plt.show()


# new
def generate_dataset(listfile, testdata=False):
    df = pd.read_csv(listfile)
    print(len(df), 'data is listed.', flush=True)

    if testdata:
        x_test = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # sex: female --> 1,  male --> 0
            row[1] = float(row[1] == 'F')

            x = np.asarray(row, dtype=np.float32)
            x_test.append(x)

        x_test = np.asarray(x_test, dtype=np.float32)
        np.savez_compressed('./data/x_test.npz', x_test)

    else:
        x_train = []  # arterial waveform
        y_train = []  # hypotension (+1, 2, 3, 4, 5 min)

        n_neg, n_pos, n_mid = 0, 0, 0

        for i, row in df.iterrows():
            id = row['caseid']
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            height = row['height']

            header = np.array([age, sex == 'F', weight, height], dtype=np.float32)
            x = pd.read_csv(f'./data/train_data/{id}.csv', header=None).values.flatten()

            count = 0
            skip = 0
            j = 0
            while j < len(x) - SRATE * (MINUTES_AHEAD + 1) * 60:
                seg = x[j: j + SRATE * (20 + (MINUTES_AHEAD + 1) * 60)]

                # skip sequence
                #  1) check max - min
                if np.isnan(seg).all() or np.nanmax(seg) - np.nanmin(seg) < 30:
                    j += len(seg)
                    skip += 1
                    continue

                #  2) check nan
                k = -1
                if np.mean(np.isnan(seg)) > 0.1:
                    k = np.where(np.isnan(seg))[0][-int(len(seg)*0.1)+1] + 1

                # interpolate
                mask = np.isnan(seg)
                if mask.any():
                    seg[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), seg[~mask])

                #  3) check min, max
                if np.nanmax(seg) > 200:
                    k = max(k, np.where(seg > 200)[0][-1] + 1)
                if np.nanmin(seg) < 20:
                    k = max(k, np.where(seg < 20)[0][-1] + 1)

                #  4) check slope
                abs_diff = np.abs(np.diff(seg))
                if np.nanmax(abs_diff) > 30:
                    k = max(k, np.where(abs_diff > 30)[0][-1] + 1)

                if k > 0:
                    j += k
                    skip += 1
                    continue
                # end skip sequence

                # build data sequence
                segx = seg[:SRATE*20]
                segy = moving_average(seg, SRATE*2)
                segy = segy[-SRATE*(MINUTES_AHEAD+1)*60:].reshape((-1, SRATE*60))
                segy = np.mean(segy < 65, axis=1)

                x_train.append(np.hstack([header, segx]))
                y_train.append(segy)

                count += 1

                # move forward
                if np.sum(segy) == 0.0:
                    n_neg += 1
                    j += 60 * SRATE
                elif segy[-1] == 1.0:
                    n_pos += 1
                    j += 30 * SRATE
                else:
                    n_mid += 1
                    j += 30 * SRATE

            print(f'{i}, id: {id}, samples: {count}, skips: {skip}')

        print(f'{len(x_train)} samples are collected in total.')
        print(f'{n_pos} ({100*n_pos/len(x_train):.2f}%) positive, {n_neg} negative, {n_mid} semi-positive samples.')

        print('saving...', flush=True, end='')
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        np.savez_compressed('./data/x_train.npz', x_train)
        np.savez_compressed('./data/y_train.npz', y_train)
        print('done', flush=True)


# old
def build_dataset_old(listfile, testdata=False):
    df = pd.read_csv(listfile)

    if testdata:
        x_test = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # age = row['age']
            # sex = row['sex']
            # weight = row['weight']
            # height = row['height']
            # header = np.asarray([age, sex == 'F', weight, height], dtype=np.float32)

            # sex: female --> 1,  male --> 0
            row[1] = float(row[1] == 'F')

            x = np.asarray(row, dtype=np.float32)
            x_test.append(x)

        x_test = np.asarray(x_test, dtype=np.float32)
        np.savez_compressed('./data/x_test.npz', x_test)

    else:
        x_train = []  # arterial waveform
        y_train = []  # hypotension

        prev_non_event = False

        for _, row in df.iterrows():
            caseid = row['caseid']
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            height = row['height']

            header = np.array([age, sex == 'F', weight, height], dtype=np.float32)
            vals = pd.read_csv('./data/train_data/{}.csv'.format(caseid), header=None).values.flatten()

            # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
            i = 0
            event_idx = []
            non_event_idx = []
            while i < len(vals) - SRATE * (20 + (1 + MINUTES_AHEAD) * 60):
                segx = vals[i:i + SRATE * 20]
                segy = vals[i + SRATE * (20 + MINUTES_AHEAD * 60):i + SRATE * (20 + (1 + MINUTES_AHEAD) * 60)]

                # 결측값 10% 이상이면
                if np.mean(np.isnan(segx)) > 0.0 or \
                        np.mean(np.isnan(segy)) > 0.0 or \
                        np.max(segx) > 200 or np.min(segx) < 20 or \
                        np.max(segy) > 200 or np.min(segy) < 20 or \
                        np.max(segx) - np.min(segx) < 30 or \
                        np.max(segy) - np.min(segy) < 30 or \
                        (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                        (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
                    i += SRATE  # 1 sec 씩 전진
                    continue

                # 출력 변수
                segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
                event = 1 if np.nanmax(segy) < 65 else 0

                if event:  # event
                    event_idx.append(i)
                    x_train.append(np.hstack([header, segx]))
                    y_train.append(event)
                    prev_non_event = False

                elif np.nanmin(segy) > 65:  # non event
                    if prev_non_event:
                        prev_non_event = False
                        i += 40 * SRATE
                        continue

                    non_event_idx.append(i)
                    x_train.append(np.hstack([header, segx]))
                    y_train.append(event)
                    prev_non_event = True

                else:
                    prev_non_event = False

                if event:
                    i += 5 * SRATE  # 5sec
                else:
                    i += 20 * SRATE  # 20sec

            nsamp = len(event_idx) + len(non_event_idx)
            if nsamp > 0:
                print('{}: {} ({:.1f}%)'.format(caseid, nsamp, len(event_idx) * 100 / nsamp))

        print('saving...', flush=True, end='')
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=bool)
        np.savez_compressed('./data/x_train.npz', x_train)
        np.savez_compressed('./data/y_train.npz', y_train)
        print('done', flush=True)


def analyze(args):
    if args.generate_dataset:
        generate_dataset(args.list_file, args.test_data)
    elif args.dataset_old:
        build_dataset_old(args.list_file, args.test_data)

    # if args.output_file:
    #     analyze_and_save_stats(listfile, args.output_file)
    # elif args.id:
    #     show(args.id, listfile)
    # elif args.save_dir:
    #     save_dataset(listfile, args.save_dir, args.test_data)

    else:
        print('operation not found.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # ap.add_argument('-l', '--list_file', type=str, required=False, help='patient list')
    # ap.add_argument('-o', '--output_file', type=str, required=False, help='output statistics file')
    # ap.add_argument('-i', '--id', type=int, required=False, help='patient id')
    # ap.add_argument('-s', '--save_dir', type=str, required=False, help='dir to save dataset files')

    ap.add_argument('-g', '--generate_dataset', action='store_true', default=False, required=False)
    ap.add_argument('-d', '--dataset_old', action='store_true', default=False, required=False)
    ap.add_argument('-l', '--list_file', type=str, required=False, help='patient list')
    ap.add_argument('-t', '--test_data', action='store_true', default=False, required=False)
    args = ap.parse_args()

    analyze(args)

