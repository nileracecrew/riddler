import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.colors import LinearSegmentedColormap
bpr = LinearSegmentedColormap.from_list('bpr', [(1,0,0),(0.5,0.0,0.5),(0,0,1)])

nx = 14
ny = 10
n_district = 7
district_size = 20


def pick_move(d, district_pair):
    move_list = []

    for y in range(ny):
        for x in range(nx):
            if d[x, y] in district_pair:
                if x < nx-1 and d[x+1, y] in district_pair and d[x, y] != d[x+1, y]:
                        if d[x, y] == district_pair[0]:
                            move_list.append(((x, y), (x+1, y)))
                        else:
                            move_list.append(((x+1, y), (x, y)))
                if y < ny-1 and d[x, y+1] in district_pair and d[x, y] != d[x, y+1]:
                        if d[x, y] == district_pair[0]:
                            move_list.append(((x, y), (x, y+1)))
                        else:
                            move_list.append(((x, y+1), (x, y)))

    if len(move_list) <= 1:
        return None

    move_pairs = list(itertools.combinations(move_list, 2))
    random.shuffle(move_pairs)
    for moves in move_pairs:
        if moves_ok(d, *moves):
            return moves

    return None


def moves_ok(d, move1, move2):
    old_d1 = d[move1[0]]
    old_d2 = d[move2[1]]
    d[move1[0]] = d[move1[1]]
    d[move2[1]] = d[move2[0]]
    ok = contiguous_district_size(d, *move1[0]) == district_size and contiguous_district_size(d, *move2[1]) == district_size 
    d[move1[0]] = old_d1
    d[move2[1]] = old_d2
    return ok


def contiguous_district_size(d, x0, y0):
    visited = np.zeros_like(d)
    n = d[x0, y0]
    def dfs(x, y):
        visited[x, y] = 1
        if x > 0 and d[x-1, y] == n and not visited[x-1, y]:
            dfs(x-1, y)
        if x < nx-1 and d[x+1, y] == n and not visited[x+1, y]:
            dfs(x+1, y)
        if y > 0 and d[x, y-1] == n and not visited[x, y-1]:
            dfs(x, y-1)
        if y < ny-1 and d[x, y+1] == n and not visited[x, y+1]:
            dfs(x, y+1)

    dfs(x0, y0)
    return visited.sum() 


blue = np.array([
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]).T

district = np.array([
    [1, 1, 1, 2, 2, 2, 4, 4, 6, 6, 6, 6, 7, 7],
    [1, 1, 1, 2, 2, 2, 4, 4, 6, 6, 6, 6, 7, 7],
    [1, 1, 1, 2, 2, 2, 4, 4, 6, 6, 6, 6, 7, 7],
    [1, 1, 1, 2, 2, 2, 4, 4, 6, 6, 6, 6, 7, 7],
    [1, 1, 1, 2, 2, 2, 4, 4, 6, 6, 6, 6, 7, 7],
    [1, 1, 1, 2, 2, 2, 4, 4, 5, 5, 5, 5, 7, 7],
    [1, 1, 3, 2, 2, 3, 4, 4, 5, 5, 5, 5, 7, 7],
    [3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 7, 7],
    [3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 7, 7],
    [3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 7, 7],
]).T - 1

district = np.array([
    [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7],
    [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7],
    [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7],
    [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7],
    [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7],
    [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7],
    [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7],
    [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7],
    [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7],
    [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7],
]).T - 1


def score(district, party):
    result = np.empty(n_district)
    for d in range(n_district):
        result[d] = party[district == d].sum() 
    return result

def district_counts(district):
    result = np.empty(n_district)
    for d in range(n_district):
        result[d] = (district == d).sum() 
    return result

def is_better(new_score, orig_score, i):
    if (new_score >= district_size / 2).sum() > (orig_score >= district_size / 2).sum():
        return True
    elif (new_score >= district_size / 2).sum() < (orig_score >= district_size / 2).sum():
        return False
    else:
        new_losers = np.sort(new_score[new_score < district_size / 2])
        orig_losers = np.sort(orig_score[orig_score < district_size / 2])
        p_new = (new_score[new_score > district_size / 2] - district_size / 2).sum() \
            - (new_losers[-1] - district_size / 2).sum() \
            + (new_losers[:-1]).sum()
        p_orig = (orig_score[orig_score > district_size / 2] - district_size / 2).sum() \
            - (orig_losers[-1] - district_size / 2).sum() \
            + (orig_losers[:-1]).sum()

        return p_new <= p_orig


def plot_result(d, party):
    result = d.astype(float).copy()
    centroid = np.empty([n_district,2])
    for n in range(n_district):
        result[d == n] = party[d == n].sum()
        loc = (d == n).nonzero()
        centroid[n, 0] = loc[0].min()
        centroid[n, 1] = loc[1][loc[0].argmin()]

    centroid[:, 0] += 0.5
    centroid[:, 1] = ny - centroid[:, 1] - 0.5

    plt.pcolor(result[:,::-1].T, vmin=0.0, vmax=district_size, cmap=bpr)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    for y in range(ny):
        for x in range(nx):
            if x < nx-1 and d[x, y] != d[x+1, y]:
                plt.plot([x+1, x+1], [ny - y - 1, ny - y], color='k', linewidth=2)
            if y < ny-1 and d[x, y] != d[x, y+1]:
                plt.plot([x, x+1], [ny - y - 1, ny - y - 1], color='k', linewidth=2)

    for n in range(n_district):
        plt.text(centroid[n, 0], centroid[n, 1], "%i" % (result[d == n][0]), va='center', ha='center', fontsize=16, fontweight='bold')
                        
    
party = blue
max_wins = party.sum() / (district_size / 2)
if max_wins > n_district:
    max_wins = n_district
orig_score = score(district, party) 

np.random.seed(123122)
random.seed(112)

f = plt.figure(figsize=(19.2/2, 10.8/2))
plot_result(district, party)
move_count = -1
for n in range(30):
    move_count += 1
    plt.savefig("%05i.png" % (move_count), dpi=200)

for i in xrange(5000):
    if i % 1000 == 0:
        print "%5i" % i
    district_pair = np.random.choice(n_district, 2, replace=False)
    moves = pick_move(district, district_pair)
    if moves:
        old_d1 = district[moves[0][0]]
        old_d2 = district[moves[1][1]]
        district[moves[0][0]] = district[moves[0][1]]
        district[moves[1][1]] = district[moves[1][0]]
        new_score = score(district, party) 

        if is_better(new_score, orig_score, i):
            move_count += 1
            if move_count % 1 == 0:
                plt.clf()
                plot_result(district, party)
                plt.savefig("%05i.png" % move_count, dpi=200)
            #print "%5i: better" % i
            #print new_score
            orig_score = new_score
            if (orig_score >= district_size / 2).sum() == max_wins:
                print "%5i: found optimal solution" % i
                break

        else:
            #print "%5i: worse" % i
            district[moves[0][0]] = old_d1
            district[moves[1][1]] = old_d2

for n in range(30):
    plt.savefig("%05i.png" % (move_count + n + 1), dpi=200)

print score(district, party)

