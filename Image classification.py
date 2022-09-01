#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy.linalg as lg
import math
from sklearn.metrics import confusion_matrix


df_ls = pd.read_csv("Linearly Seperable/trian.txt", delimiter=(","), header = None)
df_ls_dev = pd.read_csv("Linearly Seperable/dev.txt", delimiter=(","), header = None)

df_nls = pd.read_csv("Non Linearly Seperable/trian.txt", delimiter=(","), header = None)
df_nls_dev = pd.read_csv("Non Linearly Seperable/dev.txt", delimiter=(","), header = None)

df_real = pd.read_csv("Real Data/trian.txt", delimiter=(","), header = None)
df_real_dev = pd.read_csv("Real Data/dev.txt", delimiter=(","), header = None)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Essential Functions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def multivariate_gaussian(x,mu,C):
    N = x.size
    
    return((2*np.pi)**(-N/2)*(lg.det(C)**(-1/2))*(np.exp(-.5*(x-mu).T@lg.inv(C)@(x-mu))))

def grid_distribution(X1,X2,mu,C):
    Z = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i,j],X2[i,j]]).reshape(2,1)
            Z[i,j] = multivariate_gaussian(x, mu, C)
    return Z

def mle_estimates(x,y):
    
    classes = sorted(list(np.unique(y)))
    
    pi = []
    mu = []
    var = []
    
    for c in classes:
        
        indices = np.where(np.isin(y,c))
        x_slice = x[indices]
        
        n = float(len(x_slice))
        
        pi.append(n/float(len(x)))
        
        mu_temp = (np.sum(x_slice,axis=0) / n).reshape(-1,1)
        
        mu.append(mu_temp)
        
        def compute_cov(row,mean):

            return(row.reshape(-1,1) - mean).dot((row.reshape(-1,1) - mean).T)

		# do a list comprehension to sum over individual variances
		# to get a variance vector 
        var_temp = (1./(len(x_slice) - len(classes))) * (sum([compute_cov(row,mu_temp) for row in x_slice]))

        var.append(var_temp)        
    
    return (classes, pi, mu, var)

def get_colors(Z1,Z2,Z3):
    colors = np.zeros_like(Z1, dtype=object)

    for i in range(len(Z1)):
      for j in range(len(Z1)):
        if Z1[i,j] != 0:
            colors[i,j] = 'Red'
        elif Z2[i,j] != 0:
            colors[i,j] = 'Blue'
        elif Z3[i,j] != 0:
            colors[i,j] = 'Green'
        else:
            colors[i,j] = 'Black'

    return colors

def plot_distribution(m_estimates, elevation = 20, azim = -15, title = 'Multivariate Distribution', zlow = -0.1, zhigh = 0.1, x1low = -10, x1high = 20, x2low = -10, x2high = 20, zbound = 0.0005):
    x1 = np.linspace(x1low,x1high,100)
    x2 = np.linspace(x2low,x2high,100)
    X1,X2 = np.meshgrid(x1,x2)
        
    Z1 = grid_distribution(X1,X2,m_estimates[2][0],m_estimates[3][0])
    Z2 = grid_distribution(X1,X2,m_estimates[2][1],m_estimates[3][1])
    Z3 = grid_distribution(X1,X2,m_estimates[2][2],m_estimates[3][2])
    Z1[Z1<zbound] = 0
    Z2[Z2<zbound] = 0
    Z3[Z3<zbound] = 0


    fig = plt.figure(figsize = (10,8))
    fig.tight_layout()
    ax = plt.axes(projection="3d")
    ax.set_zlim(zlow,zhigh)
    ax.view_init(elevation, azim)

    ax.plot_surface(X1, X2, Z1+Z2+Z3, facecolors = get_colors(Z1,Z2,Z3), rstride=3, cstride=3, linewidth=1)

    ax.contour(X1,X2,Z1,zdir = 'z', offset = zlow)
    ax.contour(X1,X2,Z2,zdir = 'z', offset = zlow)
    ax.contour(X1,X2,Z3,zdir = 'z', offset = zlow)
    
    ax.set_xlabel("X1")
    ax.set_ylabel('X2')
    ax.set_zlabel("Probability")
    ax.zaxis.labelpad = 10
    ax.set_title(title)
    plt.show()
    

def plot_eigen(m_estimates, title = 'Multivariate Distribution', x1low = -10, x1high = 20, x2low = -10, x2high = 20):
    x1 = np.linspace(x1low,x1high,100)
    x2 = np.linspace(x2low,x2high,100)
    X1,X2 = np.meshgrid(x1,x2)
        
    Z1 = grid_distribution(X1,X2,m_estimates[2][0],m_estimates[3][0])
    Z2 = grid_distribution(X1,X2,m_estimates[2][1],m_estimates[3][1])
    Z3 = grid_distribution(X1,X2,m_estimates[2][2],m_estimates[3][2])


    fig = plt.figure(figsize = (10,8))
    fig.tight_layout()
    ax = plt.axes()
    
    eigen_values, eigen_vectors = lg.eig(m_estimates[3][0])
    eig_vec1 = eigen_vectors[:,0]
    eig_vec2 = eigen_vectors[:,1]
    origin = m_estimates[2][0]
    ax.contour(X1,X2,Z1)
    ax.quiver(*origin, *eig_vec1, color=['r'], scale = 5)
    ax.quiver(*origin, *eig_vec2, color=['b'], scale = 5)
    
    eigen_values, eigen_vectors = lg.eig(m_estimates[3][1])
    eig_vec1 = eigen_vectors[:,0]
    eig_vec2 = eigen_vectors[:,1]
    origin = m_estimates[2][1]
    ax.contour(X1,X2,Z2)
    ax.quiver(*origin, *eig_vec1, color=['g'], scale = 5)
    ax.quiver(*origin, *eig_vec2, color=['c'], scale = 5)
    
    
    eigen_values, eigen_vectors = lg.eig(m_estimates[3][2])
    eig_vec1 = eigen_vectors[:,0]
    eig_vec2 = eigen_vectors[:,1]
    origin = m_estimates[2][2]
    ax.contour(X1,X2,Z3)
    ax.quiver(*origin, *eig_vec1, color=['b'], scale = 5)
    ax.quiver(*origin, *eig_vec2, color=['m'], scale = 5)
    
    
    ax.set_xlabel("Dim 1")
    ax.set_ylabel('Dim 2')
    ax.set_title(title)
    plt.show()
    


def m_estimate_computer(x,y, case = 2):
    
    if case == 2:
        return mle_estimates(x.to_numpy(),y.to_numpy())
    
    if case == 1:
        temp_estimates = mle_estimates(x.to_numpy(),y.to_numpy())        
        temp_var = np.cov(x, rowvar=0)
        temp_estimates = list(temp_estimates)
        temp_estimates.append(temp_var)
        temp_estimates = tuple(temp_estimates)
        
        return temp_estimates
    
    if case == 3:
        temp_estimates = mle_estimates(x.to_numpy(),y.to_numpy())  
        temp_var = np.sum(np.diag(np.cov(x,rowvar=0)))/len(temp_estimates[0])
        temp_estimates = list(temp_estimates)
        temp_estimates.append(temp_var)
        temp_estimates = tuple(temp_estimates)
        return temp_estimates
    
    if case == 4:
        temp_estimates = mle_estimates(x.to_numpy(),y.to_numpy())        
        temp_var = np.diag(np.diag(np.cov(x, rowvar=0)))
        temp_estimates = list(temp_estimates)
        temp_estimates.append(temp_var)
        temp_estimates = tuple(temp_estimates)
        
        return temp_estimates
    
    if case == 5:
        temp_estimates = mle_estimates(x.to_numpy(),y.to_numpy())        
        
        for i in range(len(temp_estimates[0])):
            temp_estimates[3][i] = np.diag(np.diag(temp_estimates[3][i]))
        return temp_estimates
        
    
def bayes_prediction_case_1(X, m_estimates):
    
    bayes_probabilities = []
    
    sigma = m_estimates[4] 
    
    sigma_inv = lg.inv(sigma)
    
    for i in range(len(m_estimates[0])):
        
        bayes_probs = []
        
        pi = m_estimates[1][i]
        
        mu = m_estimates[2][i]
        
        for x in X:
            
            x = x.reshape(-1,1)
            
            wi = np.dot(sigma_inv, mu)
            
            wio = -(.5*lg.multi_dot([mu.T, sigma_inv, mu])) + np.log(pi)
            
            bayes_prob = np.dot(wi.T, x) + wio
        
            bayes_probs.append(bayes_prob)
        
        bayes_probabilities.append(np.array(bayes_probs).reshape(-1,1))
    
    
    bayes_probabilities = np.concatenate(bayes_probabilities,axis=1)
    
    prediction_class_indices = np.argmax(bayes_probabilities, axis = 1)
    
    predictions = np.array([m_estimates[0][i] for i in prediction_class_indices])
    
    return (bayes_probabilities, predictions)


def bayes_prediction_case_2(X, m_estimates):
    
    bayes_probabilities = []
    
    for i in range(len(m_estimates[0])):
        
        bayes_probs = []
        
        pi = m_estimates[1][i]
        
        mu = m_estimates[2][i]
        
        sigma = m_estimates[3][i]
        
        sigma_inv = lg.inv(sigma)
        
        for x in X:
            
            x = x.reshape(-1,1)
            
            bayes_prob = (-.5 * lg.multi_dot([(x-mu).T,(sigma_inv),(x-mu)])[0][0]) - (.5 * np.log(lg.det(sigma))) + np.log(pi)
        
            bayes_probs.append(bayes_prob)
        
        bayes_probabilities.append(np.array(bayes_probs).reshape(-1,1))
    
    bayes_probabilities = np.concatenate(bayes_probabilities,axis=1)
    
    prediction_class_indices = np.argmax(bayes_probabilities, axis = 1)
    
    predictions = np.array([m_estimates[0][i] for i in prediction_class_indices])
    
    return (bayes_probabilities, predictions)


def bayes_prediction_case_3(X, m_estimates):
    
    bayes_probabilities = []
    
    var = m_estimates[4]
    
    for i in range(len(m_estimates[0])):
        
        bayes_probs = []
        
        pi = m_estimates[1][i]
        
        mu = m_estimates[2][i]
    
        
        for x in X:
            
            x = x.reshape(-1,1)
            
            wi = mu/var
            
            bayes_prob = np.dot(wi.T, x) + np.log(pi) + (-1/(2*var))*np.dot(mu.T,mu)
        
            bayes_probs.append(bayes_prob)
        
        bayes_probabilities.append(np.array(bayes_probs).reshape(-1,1))
    
    bayes_probabilities = np.concatenate(bayes_probabilities,axis=1)
    
    prediction_class_indices = np.argmax(bayes_probabilities, axis = 1)
    
    predictions = np.array([m_estimates[0][i] for i in prediction_class_indices])
    
    return (bayes_probabilities, predictions)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Experimentation and Results
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''
Preperation of Data
'''''''''''''''''

x_ls = df_ls.loc[:,:1]
y_ls = df_ls.loc[:, 2]

x_ls_dev = df_ls_dev.loc[:,:1]
y_ls_dev = df_ls_dev.loc[:, 2]

x_nls = df_nls.loc[:,:1]
y_nls = df_nls.loc[:, 2]

x_nls_dev = df_nls_dev.loc[:,:1]
y_nls_dev = df_nls_dev.loc[:, 2]


x_real = df_real.loc[:,:1]
y_real = df_real.loc[:, 2]

x_real_dev = df_real_dev.loc[:,:1]
y_real_dev = df_real_dev.loc[:, 2]


'''''''''''''''''
Multivariate Plots for each data set (Only Train Sets)
'''''''''''''''''

m_estimates_ls = m_estimate_computer(x_ls,y_ls, case = 2)
m_estimates_nls = m_estimate_computer(x_nls,y_nls, case = 2)
m_estimates_real = m_estimate_computer(x_real,y_real, case = 2)

plot_distribution(m_estimates_ls, title = 'Multivariate Distribution for Linearly seperable Data')
plot_distribution(m_estimates_nls, title = 'Multivariate Distribution for Linearly Non seperable Data', x1low = math.floor(x_nls.min()[0]), x1high = math.ceil(x_nls.max()[0]), x2low = math.floor(x_nls.min()[1]), x2high = math.ceil(x_nls.max()[1]))
plot_distribution(m_estimates_real, title = 'Multivariate Distribution for Real Data', x1low = math.floor(x_real.min()[0]), x1high = math.ceil(x_real.max()[0]), x2low = math.floor(x_real.min()[1]), x2high = math.ceil(x_real.max()[1]), zbound = 1e-7, zlow = -1e-5, zhigh = 1e-5)


plot_eigen(m_estimates_ls, title = 'Constant Density Curve and Eigen Vectors for Linearly seperable Data', x1low = -5, x1high = 18, x2low = -2, x2high = 20)
plot_eigen(m_estimates_nls, title = 'Constant Density Curve and Eigen Vectors for Linearly Non seperable Data', x1low = 15, x1high = 50, x2low = math.floor(x_nls.min()[1]), x2high = math.ceil(x_nls.max()[1]))
plot_eigen(m_estimates_real, title = 'Constant Density Curve and Eigen Vectors for Real Data', x1low = 150, x1high = 950, x2low = 350, x2high = 2500)




'''''''''''''''''
Decision Surfaces and Confusion Matrices
'''''''''''''''''

'''
Case I: Bayes with Covariance Same for all Classes
'''


def plot_decision_boundary_local(df,x,y, title = 'Case 1'):
    X1 = np.linspace(math.floor(x.min()[0]), math.ceil(x.max()[0]), 100)
    X2 = np.linspace(math.floor(x.min()[1]), math.ceil(x.max()[1]), 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    
    g = sns.FacetGrid(df, hue=2 , height=8, palette = 'colorblind').map(plt.scatter,0,1).add_legend()
    
    g._legend.set_title("Class")
    
    ax = g.ax
    
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_temp = np.array([[X1[i,j],X2[i,j]]])
            Z[i,j] = bayes_prediction_case_1(x_temp,  m_estimate_computer(x,y, case = 1))[1][0]
            
    
            
    ax.contourf( X1, X2, Z, 2, alpha = .1, colors = ('blue','green','red'))
    ax.contour( X1, X2, Z, 2, alpha = 1, colors = ('blue','green','red'))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    plt.show()    
    
plot_decision_boundary_local(df_ls, x_ls, y_ls, title = "Decision Surface for Case 1 with Linearly Seperable Data")
plot_decision_boundary_local(df_nls, x_nls, y_nls, title = "Decision Surface for Case 1 with Non Linearly Seperable Data")
plot_decision_boundary_local(df_real, x_real, y_real, title = "Decision Surface for Case 1 with Real Data")

def plot_confusion_matrix_local(x_dev,y_dev,x_train,y_train, title = 'Confusion Matrix Case 1 with Linearly Seperable Data'):
    predictions = bayes_prediction_case_1(x_dev.to_numpy(),  m_estimate_computer(x_train,y_train, case = 1))
    
    y_pred = predictions[1]
    
    cf_matrix = confusion_matrix(y_dev.to_numpy(), y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt = '', cmap='Blues', cbar = False)

    ax.set_title(title);
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class');

    ax.xaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    ax.yaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    plt.show()
    
plot_confusion_matrix_local(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'Confusion Matrix Case 1 with Linearly Seperable Data')
plot_confusion_matrix_local(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'Confusion Matrix Case 1 with Non Linearly Seperable Data')
plot_confusion_matrix_local(x_real_dev, y_real_dev, x_real, y_real, title = 'Confusion Matrix Case 1 with Real Data')



'''
Case II: Bayes with Covariance different for all Classes
'''


def plot_decision_boundary_local(df,x,y, title = 'Case 2'):
    X1 = np.linspace(math.floor(x.min()[0]), math.ceil(x.max()[0]), 100)
    X2 = np.linspace(math.floor(x.min()[1]), math.ceil(x.max()[1]), 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    
    g = sns.FacetGrid(df, hue=2 , height=8, palette = 'colorblind').map(plt.scatter,0,1).add_legend()
    
    g._legend.set_title("Class")
    
    ax = g.ax
    
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_temp = np.array([[X1[i,j],X2[i,j]]])
            Z[i,j] = bayes_prediction_case_2(x_temp,  m_estimate_computer(x,y, case = 2))[1][0]
            
    
            
    ax.contourf( X1, X2, Z, 2, alpha = .1, colors = ('blue','green','red'))
    ax.contour( X1, X2, Z, 2, alpha = 1, colors = ('blue','green','red'))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    plt.show()    
    
plot_decision_boundary_local(df_ls, x_ls, y_ls, title = "Decision Surface for Case 2 with Linearly Seperable Data")
plot_decision_boundary_local(df_nls, x_nls, y_nls, title = "Decision Surface for Case 2 with Non Linearly Seperable Data")
plot_decision_boundary_local(df_real, x_real, y_real, title = "Decision Surface for Case 2 with Real Data")

def plot_confusion_matrix_local(x_dev,y_dev,x_train,y_train, title = 'Confusion Matrix Case 2 with Linearly Seperable Data'):
    predictions = bayes_prediction_case_2(x_dev.to_numpy(),  m_estimate_computer(x_train,y_train, case = 2))
    
    y_pred = predictions[1]
    
    cf_matrix = confusion_matrix(y_dev.to_numpy(), y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt = '', cmap='Blues', cbar = False)

    ax.set_title(title);
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class');

    ax.xaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    ax.yaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    plt.show()
    
plot_confusion_matrix_local(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'Confusion Matrix Case 2 with Linearly Seperable Data')
plot_confusion_matrix_local(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'Confusion Matrix Case 2 with Non Linearly Seperable Data')
plot_confusion_matrix_local(x_real_dev, y_real_dev, x_real, y_real, title = 'Confusion Matrix Case 2 with Real Data')


'''
Case III: Bayes with Covariance sigma^2I
'''


def plot_decision_boundary_local(df,x,y, title = 'Case 3'):
    X1 = np.linspace(math.floor(x.min()[0]), math.ceil(x.max()[0]), 100)
    X2 = np.linspace(math.floor(x.min()[1]), math.ceil(x.max()[1]), 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    
    g = sns.FacetGrid(df, hue=2 , height=8, palette = 'colorblind').map(plt.scatter,0,1).add_legend()
    
    g._legend.set_title("Class")
    
    ax = g.ax
    
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_temp = np.array([[X1[i,j],X2[i,j]]])
            Z[i,j] = bayes_prediction_case_3(x_temp,  m_estimate_computer(x,y, case = 3))[1][0]
            
    
            
    ax.contourf( X1, X2, Z, 2, alpha = .1, colors = ('blue','green','red'))
    ax.contour( X1, X2, Z, 2, alpha = 1, colors = ('blue','green','red'))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    plt.show()    
    
plot_decision_boundary_local(df_ls, x_ls, y_ls, title = "Decision Surface for Case 3 with Linearly Seperable Data")
plot_decision_boundary_local(df_nls, x_nls, y_nls, title = "Decision Surface for Case 3 with Non Linearly Seperable Data")
plot_decision_boundary_local(df_real, x_real, y_real, title = "Decision Surface for Case 3 with Real Data")

def plot_confusion_matrix_local(x_dev,y_dev,x_train,y_train, title = 'Confusion Matrix Case 3 with Linearly Seperable Data'):
    predictions = bayes_prediction_case_3(x_dev.to_numpy(),  m_estimate_computer(x_train,y_train, case = 3))
    
    y_pred = predictions[1]
    
    cf_matrix = confusion_matrix(y_dev.to_numpy(), y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt = '', cmap='Blues', cbar = False)

    ax.set_title(title);
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class');

    ax.xaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    ax.yaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    plt.show()
    
plot_confusion_matrix_local(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'Confusion Matrix Case 3 with Linearly Seperable Data')
plot_confusion_matrix_local(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'Confusion Matrix Case 3 with Non Linearly Seperable Data')
plot_confusion_matrix_local(x_real_dev, y_real_dev, x_real, y_real, title = 'Confusion Matrix Case 3 with Real Data')
 

'''
Case IV: Naive Bayes with Covariance same for all Classes
'''


def plot_decision_boundary_local(df,x,y, title = 'Case 4'):
    X1 = np.linspace(math.floor(x.min()[0]), math.ceil(x.max()[0]), 100)
    X2 = np.linspace(math.floor(x.min()[1]), math.ceil(x.max()[1]), 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    
    g = sns.FacetGrid(df, hue=2 , height=8, palette = 'colorblind').map(plt.scatter,0,1).add_legend()
    
    g._legend.set_title("Class")
    
    ax = g.ax
    
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_temp = np.array([[X1[i,j],X2[i,j]]])
            Z[i,j] = bayes_prediction_case_1(x_temp,  m_estimate_computer(x,y, case = 4))[1][0]
            
    
            
    ax.contourf( X1, X2, Z, 2, alpha = .1, colors = ('blue','green','red'))
    ax.contour( X1, X2, Z, 2, alpha = 1, colors = ('blue','green','red'))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    plt.show()    
    
plot_decision_boundary_local(df_ls, x_ls, y_ls, title = "Decision Surface for Case 4 with Linearly Seperable Data")
plot_decision_boundary_local(df_nls, x_nls, y_nls, title = "Decision Surface for Case 4 with Non Linearly Seperable Data")
plot_decision_boundary_local(df_real, x_real, y_real, title = "Decision Surface for Case 4 with Real Data")

def plot_confusion_matrix_local(x_dev,y_dev,x_train,y_train, title = 'Confusion Matrix Case 4 with Linearly Seperable Data'):
    predictions = bayes_prediction_case_1(x_dev.to_numpy(),  m_estimate_computer(x_train,y_train, case = 4))
    
    y_pred = predictions[1]
    
    cf_matrix = confusion_matrix(y_dev.to_numpy(), y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt = '', cmap='Blues', cbar = False)

    ax.set_title(title);
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class');

    ax.xaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    ax.yaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    plt.show()
    
plot_confusion_matrix_local(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'Confusion Matrix Case 4 with Linearly Seperable Data')
plot_confusion_matrix_local(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'Confusion Matrix Case 4 with Non Linearly Seperable Data')
plot_confusion_matrix_local(x_real_dev, y_real_dev, x_real, y_real, title = 'Confusion Matrix Case 4 with Real Data')
 

'''
Case V: Naive Bayes with Covariance different for all Classes
'''


def plot_decision_boundary_local(df,x,y, title = 'Case 5'):
    X1 = np.linspace(math.floor(x.min()[0]), math.ceil(x.max()[0]), 100)
    X2 = np.linspace(math.floor(x.min()[1]), math.ceil(x.max()[1]), 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    
    g = sns.FacetGrid(df, hue=2 , height=8, palette = 'colorblind').map(plt.scatter,0,1).add_legend()
    
    g._legend.set_title("Class")
    
    ax = g.ax
    
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_temp = np.array([[X1[i,j],X2[i,j]]])
            Z[i,j] = bayes_prediction_case_2(x_temp,  m_estimate_computer(x,y, case = 5))[1][0]
            
    
            
    ax.contourf( X1, X2, Z, 2, alpha = .1, colors = ('blue','green','red'))
    ax.contour( X1, X2, Z, 2, alpha = 1, colors = ('blue','green','red'))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    plt.show()    
    
plot_decision_boundary_local(df_ls, x_ls, y_ls, title = "Decision Surface for Case 5 with Linearly Seperable Data")
plot_decision_boundary_local(df_nls, x_nls, y_nls, title = "Decision Surface for Case 5 with Non Linearly Seperable Data")
plot_decision_boundary_local(df_real, x_real, y_real, title = "Decision Surface for Case 5 with Real Data")

def plot_confusion_matrix_local(x_dev,y_dev,x_train,y_train, title = 'Confusion Matrix Case 5 with Linearly Seperable Data'):
    predictions = bayes_prediction_case_2(x_dev.to_numpy(),  m_estimate_computer(x_train,y_train, case = 5))
    
    y_pred = predictions[1]
    
    cf_matrix = confusion_matrix(y_dev.to_numpy(), y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt = '', cmap='Blues', cbar = False)

    ax.set_title(title);
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class');

    ax.xaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    ax.yaxis.set_ticklabels(['Clas 1','Class 2', 'Class 3'])
    plt.show()
    
plot_confusion_matrix_local(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'Confusion Matrix Case 5 with Linearly Seperable Data')
plot_confusion_matrix_local(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'Confusion Matrix Case 5 with Non Linearly Seperable Data')
plot_confusion_matrix_local(x_real_dev, y_real_dev, x_real, y_real, title = 'Confusion Matrix Case 5 with Real Data')
 

'''''''''''''''''
ROC and DET Curves
'''''''''''''''''
def roc_rates(scores, y_actual):
    TPR = []
    FPR = []
    FNR = []
    
    for threshold in np.linspace(scores.max(), scores.min(), 100):
        TP = FP = TN = FN = 0
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i][j] >= threshold:
                    if y_actual[i] == (j + 1):
                        TP = TP + 1
                    else: 
                        FP = FP + 1
                else:
                    if y_actual[i] == (j + 1):
                        FN = FN + 1
                    else: 
                        TN = TN + 1
        FNR.append(FN/(FN+TP))
        TPR.append(TP/(TP + FN))
        FPR.append(FP/(FP + TN))
    
    return (TPR,FPR,FNR)

def roc_plotter(x_dev,y_dev,x_train,y_train,title = 'ROC Plot for Linearly Seperable Data'):
    scores_1 = bayes_prediction_case_1(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 1))[0]
    scores_2 = bayes_prediction_case_2(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 2))[0]
    scores_3 = bayes_prediction_case_3(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 3))[0]
    scores_4 = bayes_prediction_case_1(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 4))[0]
    scores_5 = bayes_prediction_case_2(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 5))[0]
    
    TPR1,FPR1,FNR1 = roc_rates(scores_1, y_dev)
    plt.plot(FPR1,TPR1, label = 'Case 1')
    TPR2,FPR2,FNR2 = roc_rates(scores_2, y_dev)
    plt.plot(FPR2,TPR2, label = 'Case 2')
    TPR3,FPR3,FNR3 = roc_rates(scores_3, y_dev)
    plt.plot(FPR3,TPR3, label = 'Case 3')
    TPR4,FPR4,FNR4 = roc_rates(scores_4, y_dev)
    plt.plot(FPR4,TPR4, label = 'Case 4')
    TPR5,FPR5,FNR5 = roc_rates(scores_5, y_dev)
    plt.plot(FPR5,TPR5, label = 'Case 5')
    plt.title(title)
    plt.xlabel("False Positivity Rate (FPR)")
    plt.ylabel("True Positivity Rate (TPR)")
    plt.legend()
    plt.show()

roc_plotter(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'ROC Plot for Linearly Seperable Data')
roc_plotter(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'ROC Plot for Non Linearly Seperable Data')
roc_plotter(x_real_dev, y_real_dev, x_real, y_real, title = 'ROC Plot for Real Data')
            
def det_plotter(x_dev,y_dev,x_train,y_train,title = 'DET Plot for Linearly Seperable Data'):
    scores_1 = bayes_prediction_case_1(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 1))[0]
    scores_2 = bayes_prediction_case_2(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 2))[0]
    scores_3 = bayes_prediction_case_3(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 3))[0]
    scores_4 = bayes_prediction_case_1(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 4))[0]
    scores_5 = bayes_prediction_case_2(x_dev.to_numpy(), m_estimate_computer(x_train, y_train, case = 5))[0]
    
    TPR1,FPR1,FNR1 = roc_rates(scores_1, y_dev)
    plt.plot(FPR1,FNR1, label = 'Case 1')
    TPR2,FPR2,FNR2 = roc_rates(scores_2, y_dev)
    plt.plot(FPR2,FNR2, label = 'Case 2')
    TPR3,FPR3,FNR3 = roc_rates(scores_3, y_dev)
    plt.plot(FPR3,FNR3, label = 'Case 3')
    TPR4,FPR4,FNR4 = roc_rates(scores_4, y_dev)
    plt.plot(FPR4,FNR4, label = 'Case 4')
    TPR5,FPR5,FNR5 = roc_rates(scores_5, y_dev)
    plt.plot(FPR5,FNR5, label = 'Case 5')
    plt.title(title)
    plt.xlabel("False Positivity Rate (FPR)")
    plt.ylabel("False Negativity Rate (FNR)")
    plt.legend()
    plt.show()
    
det_plotter(x_ls_dev, y_ls_dev, x_ls, y_ls, title = 'DET Plot for Linearly Seperable Data')
det_plotter(x_nls_dev, y_nls_dev, x_nls, y_nls, title = 'DET Plot for Non Linearly Seperable Data')
det_plotter(x_real_dev, y_real_dev, x_real, y_real, title = 'DET Plot for Real Data')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
--------------------------------------THE END--------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

