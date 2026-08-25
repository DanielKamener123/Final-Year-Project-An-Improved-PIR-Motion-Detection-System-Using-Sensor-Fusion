from interface import getFrame as getRawFrame
import numpy as np

ALPHA = 0.99
THRESHOLD = 0.6
CONNECTIVITY_RULE = 8

def getFrame():
    rawFrame = np.array(getRawFrame()).reshape(24,32)

    #return rawFrame
    return getRawFrame()

def updateBackground(background, current, sigma, mask):
    # Update Running Background Average
    newBackground = []
    for i in range(0, 24):
        newRow = []
        for j in range(0,32):
            # Only updates for background - doesnt update if human pixel
            if mask[i][j] == 0:
                newRow.append(((1 - ALPHA) * background[i][j]) + (ALPHA * current[i][j]))
            else:
                newRow.append(background[i][j])
        newBackground.append(newRow)

    # Update Variance
    newSigma = []
    for i in range(0, 24):
        newRow = []
        for j in range(0,32):
            #
            if mask[i][j] == 0:
                newRow.append(((1-ALPHA) * sigma[i][j]) + (ALPHA * np.square(background[i][j] - current[i][j])))
            else:
                newRow.append(sigma[i][j])
        newSigma.append(newRow)

    return newBackground, newSigma

def foregroundExtraction(background, current, sigma):
    # How many multiples of this pixels normal variation is present
    Z = []
    for i in range(0, 24):
        item = []
        for j in range(0,32):
            item.append(np.abs(background[i][j] - current[i][j]) / sigma[i][j])
        Z.append(item)
    
    return Z

def spatialFilter(Z):
    Z  = np.array(Z)

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype = np.float64) / 16.0

    return np.convolve(Z, kernel, mode='reflect')

def thresholding(filteredZ):
    mask = []
    for i in range(0,24):
        row = []
        for j in range(0,32):
            if filteredZ[i][j] > THRESHOLD:
                row.append(1)
            else:
                row.append(0)
        mask.append(row)

    return mask

# Uses flood fill
def blobExtraction(mask):
    visited = [[False] * 24 for i in range(32)]
    labels = [[0] * 24 for i in range(32)]
    nextLabel = 1

    neighbours = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1,0], [1,1]
    ]

    for i in mask:
        for j in i:
            # Make sure pixel is human and not already grouped
            if mask[i][j] == 0:
                continue
            if visited[i][j]:
                continue

            # Make new blob
            queue = [[i, j]]
            visited[i][j] = True

            while len(queue) != 0:
                r, c = queue[0]
                queue.pop(0)

                labels[r][c] = nextLabel

                for dr, dc in neighbours:
                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < 24 and 0 <= nc < 32:
                        if mask[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append([nr, nc])

            nextLabel += 1

    # Determine pixel count and centroid of blobs
    pixelCount = []
    for i in range(0, nextLabel):
        count = 0
        for row in labels:
            for pixel in row:
                count += 1

        pixelCount.append(count)

    
                    
    return

def processingPipeline():
    rawFrame = getFrame()
    return

print(getFrame())