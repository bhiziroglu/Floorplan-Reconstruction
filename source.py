#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import itertools
import json
import config as cfg



def euclidian_distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))


def show(im):
    cv2.namedWindow('TEST WINDOW', cv2.WINDOW_NORMAL)
    cv2.imshow("TEST WINDOW", im)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def corner_reducer(cornerList,THRESHOLD):
    reduced_Corners = []
    seen = []
    for p1 in cornerList:
        tmp = []  # Temporary stack to store points that are close to p1
        if p1 in seen:  # Already calculated !
            continue
        for p2 in cornerList:
            if (p1[0] == p2[0] and p1[1] == p2[1]):  # SAME POINT !
                continue
            dist = euclidian_distance(p1, p2)
            if (dist < THRESHOLD):
                tmp.append(p2)
                seen.append(p2)

        if (len(tmp) < 1):
            reduced_Corners.append(p1)
        else:
            avX = 0.0
            avY = 0.0
            for p in tmp:
                avX += p[0]
                avY += p[1]
            avX = avX / len(tmp)
            avY = avY / len(tmp)
            reduced_Corners.append((avX, avY))

    return reduced_Corners


def snap_to_grid(cornerList,GRID_SIZE): #Takes bunch of coordinates, changes their position to fit on a grid

    snapped = []

    for coord in cornerList:
        X = coord[0]
        Y = coord[1]

        X_val = X % GRID_SIZE
        Y_val = Y % GRID_SIZE

        if X_val < GRID_SIZE/2:
            X = X - X_val
        else:
            X = X + GRID_SIZE - X_val

        if Y_val < GRID_SIZE/2:
            Y = Y - Y_val
        else:
            Y = Y + GRID_SIZE - Y_val

        snapped.append((X,Y))

    return snapped


def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def bline(x1,y1,x2,y2): # Draws a line between (X1,X2) and (X2,Y2) in R2 cartesian plane

    res = []
    xinc = 0
    yinc = 0
    x = 0
    y = 0
    dx = 0
    dy= 0
    e = 0

    dx = math.fabs(x2 - x1)
    dy = math.fabs(y2 - y1)

    if (x1 < x2):
        xinc = 1
    else:
        xinc = -1

    if (y1 < y2):
        yinc = 1
    else:
        yinc = -1
    x = x1;
    y = y1;

    res.append((x,y))
    if (dx >= dy):
        e = (2 * dy) - dx
        while(x!=x2):
            if(e<0):
                e+=(2*dy)
            else:
                e+=(2*(dy-dx))
                y+=yinc
            x+=xinc
            res.append((x,y))
    else:
        e = (2 * dx) - dy
        while(y!=y2):
            if(e<0):
                e+=(2*dx)
            else:
                e+=(2*(dx-dy))
                x+=xinc
            y+=yinc
            res.append((x,y))
    return res

def thick_line_fixed_y(x1,y1,x2,y2):
    wy = 10
    res = []
    res += bline(x1,y1,x2,y2)

    for i in range(int(wy) + 1):
        res += bline(x1, y1 - i, x2, y2 - i)
        res += bline(x1, y1 + i, x2, y2 + i)

    return res

def thick_line_fixed_x(x1,y1,x2,y2):
    wx = 10
    res = []
    res += bline(x1,y1,x2,y2)

    for i in range(int(wx) + 1):
        res += bline(x1 - i, y1, x2 - i, y2)
        res += bline(x1 + i, y1, x2 + i, y2)

    return res

def thick_line(x1,y1,x2,y2):

    thickness = 5
    res = []
    res += bline(x1,y1,x2,y2)

    if (y2-y1)/(x2-x1)<1:
        wy=(thickness-1)*math.sqrt(pow((x2-x1),2)+pow((y2-y1),2))/(2*math.fabs(x2-x1));
        for i in range(int(wy)+1):
            res += bline(x1,y1-i,x2,y2-i);
            res += bline(x1,y1+i,x2,y2+i);
    else:
        wx=(thickness-1)*math.sqrt(pow((x2-x1),2)+pow((y2-y1),2))/(2*math.fabs(y2-y1));
        for i in range(int(wx)+1):
            res += bline(x1-i,y1,x2-i,y2)
            res += bline(x1+i,y1,x2+i,y2)
    return res


def add_border(src):

    top = int(0.04 * src.shape[0])*0+50
    bottom = int(0.04 * src.shape[0])*0+50
    left = int(0.04 * src.shape[1])*0+50
    right = int(0.04 * src.shape[1])*0+50
    res = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    return res

def add_border_binary(src):
    top = int(0.04 * src.shape[0])*0+50
    bottom = int(0.04 * src.shape[0])*0+50
    left = int(0.04 * src.shape[1])*0+50
    right = int(0.04 * src.shape[1])*0+50
    res = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)
    return res


def proximity_matrix(snapped):
    width = len(snapped)
    height = len(snapped)

    m = np.zeros((width,height)) #M[i,j] -> Proximity order of Vertex j wrt Vertex i

    for ind1, p1 in enumerate(snapped):
        tmp = []
        for ind2, p2 in enumerate(snapped):
            dist = euclidian_distance(p1,p2)
            tmp.append((p2,dist))

            tmp = sorted(tmp, key=lambda tup: tup[1]) # SOrt by distance

        for p,_ in tmp:
            ind2 = snapped.index(p)
            m[ind1,ind2] = tmp.index((p,_))

    return m




img = cv2.imread(cfg.IMAGE_NAME, cv2.IMREAD_COLOR)
img2 = add_border(img)

kernel = np.ones(cfg.CLOSE_KERNEL_SIZE, np.uint8)
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
gray = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)


T = cv2.imread(cfg.TARGET_NAME,cv2.IMREAD_GRAYSCALE)
T = add_border_binary(T)

kernel2 = np.ones(cfg.DILATE_KERNEL_SIZE, np.uint8)
T = cv2.morphologyEx(T, cv2.MORPH_DILATE, kernel2,iterations=2) # For ground truth checking

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,cfg.HARRIS_BLOCK_SIZE,cfg.HARRIS_K_SIZE,cfg.HARRIS_K)
dst = cv2.dilate(dst,None)

n = np.zeros_like(img2)
n[dst>0.06*dst.max()] = [255,255,255]

'''Get the corner coordinates'''
m = n[:,:,0] # Manual RGB to Binary
cornerList = []
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        if(m[i][j]==255): # CORNER FOUND !
            cornerList.append((i,j))


reducedCorners = corner_reducer(cornerList,cfg.REDUCTION_DISTANCE_1) # 20 is the Treshold for the max distance between any two vertices to treat them as one.

snapped = snap_to_grid(reducedCorners,cfg.GRID_SIZE)

snapped = corner_reducer(snapped,cfg.REDUCTION_DISTANCE_2) # Reduce again to dilate grouped up points



proximity = proximity_matrix(snapped)

if cfg.PIXEL_SHIFT>0:
    snapped_updated = []
    for p in snapped:
        snapped_updated.append((p[0],p[1]-5)) #Shift every corner 5 pixels up !
    snapped = snapped_updated


corners = {}
hash_ = lambda V : V[0]*V[1]+V[0]+V[1]


S = findsubsets(snapped,2) # N * (N-1) / 2 items

walls = []
seen = []
TOTAL = len(S)
skipCounter =0
iteration = 0
for pair in iter(S):
    iteration += 1
    if(iteration%50==0):
        print("Completed %"+str(iteration*100.0/TOTAL))

    p1, p2 = pair

    if proximity[snapped.index(p1),snapped.index(p2)] == 0: #Skip if same digit
        continue

    if (p1,p2) in seen or (p2,p1) in seen: # Do not add bi-directional walls
        continue

    ends = np.array([[ p1[0], p1[1]],
                 [ p2[0], p2[1]]])


    if int(ends[0][0]) - int(ends[1][0]) == 0: #or ends[0][1]-ends[1][1] == 0:

        points = thick_line_fixed_x(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))


    elif int(ends[0][1]) - int(ends[1][1]) == 0:

        points = thick_line_fixed_y(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    else:

        points = thick_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    cntr = 0


    for p in points:

        if(p[0]>=T.shape[0]):
            skipCounter +=1
            continue
        if(p[1]>=T.shape[1]):
            skipCounter +=1
            continue

        if T[int(p[0]),int(p[1])]==255:
            cntr+=1

    alpha = cfg.ALPHA
    beta = cfg.BETA

    rank = alpha*(cntr/len(points)) + beta * (1/proximity[snapped.index(p1),snapped.index(p2)])

    if(rank > cfg.RANK_THRESHOLD):

        walls.append({
            'corner1': str(int(hash_(p1))),
            'corner2': str(int(hash_(p2))),
            'frontTexture': {},
            'backTexture': {}
        })
        seen.append((p1,p2))


print("Completed %100")
avX = sum([pair[0] for pair in snapped]) / len(snapped)
avY = sum([pair[1] for pair in snapped]) /len(snapped)

for V in snapped:
    corners[str(int(hash_(V)))] = {
        'x': ((img2.shape[0]-V[0])-1*avX)*cfg.SCALE,
        'y': (V[1]-1*avY)*cfg.SCALE
    }


reduced_walls = []
seen = []
for wall in walls:

    if (wall['corner1'],wall['corner2']) in seen or (wall['corner2'],wall['corner1']) in seen:
        continue

    seen.append((wall['corner1'],wall['corner2']))

    reduced_walls.append({
        'corner1': wall['corner1'],
        'corner2': wall['corner2'],
        'frontTexture': {},
        'backTexture': {}
    })


if cfg.REMOVE_COLLISION==True:
    reduced_walls_2 = []
    skipList = []
    willRemove = []
    willAdd = []
    seen = []
    for i in range(2):
        for j in range(2):

            for wall in reduced_walls:

                corner1 = corners[wall['corner1']]
                corner2 = corners[wall['corner2']]

                if i == 0: # To generate permutations for corner1,corner2 pairs for each edge.

                    corner1_X = corner1['x']
                    corner1_Y = corner1['y']

                    corner2_X = corner2['x']
                    corner2_Y = corner2['y']

                elif i == 1:

                    corner1_X = corner2['x']
                    corner1_Y = corner2['y']

                    corner2_X = corner1['x']
                    corner2_Y = corner1['y']

                for wall2 in reduced_walls:

                    if wall2 in seen:
                        continue

                    #if wall['corner1'] == wall2['corner1'] and wall['corner2'] == wall2['corner2']: # Skip if it is the same wall
                    #    continue

                    corner3 = corners[wall2['corner1']]
                    corner4 = corners[wall2['corner2']]


                    if j == 0:

                        corner3_X = corner3['x']
                        corner3_Y = corner3['y']

                        corner4_X = corner4['x']
                        corner4_Y = corner4['y']

                    elif j == 1:

                        corner3_X = corner4['x']
                        corner3_Y = corner4['y']

                        corner4_X = corner3['x']
                        corner4_Y = corner3['y']


                    # Possible collision scenarios
                    # Vertical ->
                    #             X values are same for either corner1 or corner2
                    # Horizontal ->
                    #             Y values are same for either corner1 or corner2

                    if corner1_X == corner3_X: # All point should have same X value

                        # Two collision scenarios ->
                        # 1)
                        # <------>
                        #      <-------->
                        # 2)
                        # <------------------------>
                        #       <-------->

                        if corner2_Y > corner3_Y and corner1_Y < corner3_Y: # Scenario 1

                            seen.append(wall)
                            seen.append(wall2)

                            willRemove.append(wall)
                            willRemove.append(wall2)

                            willAdd.append({
                                'corner1': wall['corner1'],
                                'corner2': wall2['corner2'],
                                'frontTexture': {},
                                'backTexture': {}
                            })

                            continue

                        if corner1_Y > corner3_Y and corner2_Y < corner4_Y: #Scenario 2

                            willRemove.append(wall2)
                            seen.append(wall2)
                            continue


                        if corner2_Y == corner3_Y :

                            seen.append(wall)
                            seen.append(wall2)

                            willRemove.append(wall)
                            willRemove.append(wall2)

                            willAdd.append({
                                'corner1': wall['corner1'],
                                'corner2': wall2['corner2'],
                                'frontTexture': {},
                                'backTexture': {}
                            })
                            continue

                    if corner1_Y == corner3_Y :

                        if corner2_X > corner3_X and corner1_X < corner3_X:

                            seen.append(wall)
                            seen.append(wall2)

                            willRemove.append(wall)
                            willRemove.append(wall2)

                            willAdd.append({
                                'corner1': wall['corner1'],
                                'corner2': wall2['corner2'],
                                'frontTexture': {},
                                'backTexture': {}
                            })
                            continue

                        if corner3_X < corner1_X and corner2_X < corner4_X:

                            willRemove.append(wall)
                            seen.append(wall)
                            continue

                        if corner2_X == corner3_X :

                            seen.append(wall)
                            seen.append(wall2)

                            willRemove.append(wall)
                            willRemove.append(wall2)

                            willAdd.append({
                                'corner1': wall['corner1'],
                                'corner2': wall2['corner2'],
                                'frontTexture': {},
                                'backTexture': {}
                            })
                            continue


    '''Removing duplicates in '''
    willRemoveAgain = []
    for i in range(0, len(willRemove)):
        if willRemove[i] not in willRemove[i+1:]:
            willRemoveAgain.append(willRemove[i])


    for rem in willRemoveAgain:
        reduced_walls.remove(rem)


    for ad in willAdd:
        reduced_walls.append(ad)



print("Number of walls: "+str(len(walls)))
print("Number of walls after reducing: "+str(len(reduced_walls)))


res = {
    'floorplan': {},
    'items': []
}

res['floorplan'] = {
    'corners': corners,
    'floorTextures': {},
    'newFloorTextures': {},
    'wallTextures': [],
    'walls': reduced_walls
}

with open('data.blueprint3d', 'w') as fp:
    json.dump(res, fp)


if __name__=="__main__":
    print()

