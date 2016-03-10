import numpy as np
import random
from matplotlib import pyplot as plt

class RANSAC(object):
    def __init__(self, subsets=2, model='line', err_func='lse'):
        self.subsets = subsets
        self.model = model
        self.err_func = err_func

    # points was stored in format [x(row), y(column)]
    # line: y=mx+c => ax+by=d
    def build_line_models(self, point1, point2):
        if point2[0] != point1[0]:
            m = (point2[1] - point1[1])/(point2[0] - point1[0])
        else:
            m = 0
        c = point1[1] - m*point1[0]
        a = m / np.sqrt(m**2+1)
        b = -1 / np.sqrt(m**2+1)
        d = -c / np.sqrt(m**2+1)
        return (a, b, d)

    def compute_error(self, points, model, err_func, threshold):
        error = 0
        inliers = []
        outliers = []
        for p in points:
            curErr = err_func(p, model)
            if curErr <= threshold:
                inliers.append(p)
            else:
                outliers.append(p)
            error += curErr
        return error, inliers, outliers

    def least_square_error(self, point, model):
        a, b, d = model[:3]
        dist = np.abs(a*point[0]+b*point[1]-d)
        return dist

    def sample(self, points):
        l = []
        random.shuffle(points)
        for i in range(0, len(points)):
            for j in range(i+1, len(points)):
                l.append((points[i], points[j]))
        return l

    def fit(self, points, e=0.5, p=0.99, threshold=3):
        best_e = e
        best_inliers = []
        best_outliers = []
        best_model=[0,0,0]

        samples = self.sample(points)
        N = len(samples)+1
        sample_count = 0
        visited = [0] * len(samples)
        max_inlier_ratio = 0

        while N > sample_count:
            # 1, Choose a sample
            while True:
                index = random.randint(0, len(samples)-1)
                if visited[index] == 0:
                    break
            sample = samples[index]
            visited[index] = 1
            # 2, Choose a model and error function
            if self.model == 'line':
                model = self.build_line_models(sample[0], sample[1])
                print(sample[0], sample[1])
                if self.err_func == 'lse':
                    dist, inliers, outliers = self.compute_error(points, model, self.least_square_error, threshold)
                else:
                    print('Error: No such error function yet')
            else:
                print('Error: No such modules yet')
            # 3, Count the number of inliers
            if len(inliers)/len(points) > max_inlier_ratio:
                max_inlier_ratio = len(inliers)/len(points)
                print('The inlier ratio is:', max_inlier_ratio)
                e = 1 - max_inlier_ratio
                best_e = e
                best_model=model
                best_inliers = inliers
                best_outliers = outliers
            # N is in the loop or not???
            N = np.log(1-p)/np.log(1-(1-e)**self.subsets)
            sample_count += 1
        #self.plot(best_model, 3, 'r--')
        # 4, refit
        #refit_model, refit_inliers, refit_outliers = self.refit(best_inliers, points, best_e, p, threshold)
        #self.plot(refit_model, 3, 'b--')
        #return refit_model
        return best_model

    def refit(self, points, data, e, p, threshold=3):
        samples = self.sample(points)
        max_inlier_ratio = 1 - e
        best_e = e
        best_model = [0,0,0]
        best_inliers = []
        best_outliers = []
        dist = 0
        inliers = []
        outliers = []
        for pair in samples:
            if self.model == 'line':
                model = self.build_line_models(pair[0], pair[1])
                if self.err_func == 'lse':
                    dist, inliers, outliers = self.compute_error(data, model, self.least_square_error, threshold)
                else:
                    print('Error: No such error function yet and refit should have same one')
            else:
                print('Error: No such modules yet and refit should have same one')
            if len(inliers)/len(data) > max_inlier_ratio:
                print('Refit works')
                max_inlier_ratio = len(inliers)/len(points)
                best_e = 1 - max_inlier_ratio
                best_inliers = inliers
                best_outliers = outliers
                best_model = model
        return best_model, best_inliers, best_outliers

    def plot(self, model, threshold, mark):
        a, b, d = model[:3]
        if b == 0:
            x1 = d/a
            y1 = -10
            x2 = d/a
            y2 = 10
        else:
            x1=0
            y1 = (d-a*x1)/b
            x2=10
            y2 = (d-a*x2)/b

        theta = np.arctan(-(a/b))
        x11 = x1-threshold*np.sin(theta)
        y11 = y1-threshold*np.cos(theta)
        x12 = x1+threshold*np.sin(theta)
        y12 = y1+threshold*np.cos(theta)

        x21 = x2-threshold*np.sin(theta)
        y21 = y2-threshold*np.cos(theta)
        x22 = x2+threshold*np.sin(theta)
        y22 = y2+threshold*np.cos(theta)


        plt.plot([x1, x2], [y1, y2])
        plt.plot([x11, x21], [y11, y21], mark)
        plt.plot([x12, x22], [y12, y22], mark)


if __name__ == '__main__':
    plt.figure()

    r = RANSAC()
    points = [(1,1), (2.1,2.2), (3.2,3), (4,4),(3,4), (5, 5), (6,6),(5,8), (3,7), (3,8), (6, 7), (6.5,6), (4.3, 4.8)]
    plt.plot(*zip(*points), marker='o', color='r', ls='')
    axes = plt.gca()
    axes.set_xlim([-10,20])
    axes.set_ylim([-10,20])
    #plt.figure()
    #plt.plot(*zip(*points), marker='o', color='r', ls='')
    #plt.show()
    model = r.fit(points)

    plt.show()
'''
    print(model)
    a, b, d = model[:3]
    x1=0
    y1 = (d-a*x1)/b
    x2=10
    y2 = (d-a*x2)/b
    plt.figure()
    plt.plot([x1, x2], [y1, y2])
    plt.plot(*zip(*points), marker='o', color='r', ls='')
    plt.show()
'''

