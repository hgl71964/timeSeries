import matplotlib.pyplot as plt
import numpy as np
import torch as tr
import os

class rosenbrock_plot:

    @staticmethod
    def plot(
            x: np.ndarray, # the range of the variable
            y: np.ndarray, # the range of the variable
            data: list,  # a list of dict, which contains data
            ):

        if isinstance(x, tr.Tensor):
            x, y = x.numpy(), y.numpy()
            
        X, Y = np.meshgrid(x, y)
        f = lambda x,y: (1-x) ** 2 + 100*((y-x**2)**2)
        Z = f(X,Y)

        fig, ax = plt.subplots(figsize=(12,8))

        # ax.set_aspect('equal')
        cf = ax.contourf(X,Y,Z,np.arange(0, 10000, 1000),extend='both')
        ax.set_xlabel('x',fontsize = 16)
        ax.set_ylabel('y',fontsize = 16)
        # ax.set_title('Gradient descent',fontsize = 18)
        # fig.colorbar(cf, ax=ax)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=14) 

        for i in range(len(data)):
            x = data[i]["x"]
            name = data[i]["name"]
            ax.scatter(x[:,0], x[:,1], marker='o', label=f'{name}')

        ax.scatter([1],[1],marker='o',color='white',label='global minimum')
        ax.legend(prop={'size':12})
        plt.show()


class convergent_plot:

    @staticmethod
    def plot(
            data: list,  # a list of dict, which contains data
            ):

        fig, ax = plt.subplots(figsize=(12,8))

        for i in range(len(data)):
            x = data[i]["x"]
            y = data[i]["y"]
            name = data[i]["name"]
            ax.plot(y, label=f'{name}')

        ax.legend(prop={'size':12})
        plt.show()

    @staticmethod
    def plot_all(folder: str, 
                n=None):


        files = os.listdir(folder)

        fig, ax = plt.subplots(figsize=(12,8))

        for f in files:
            
            obj = tr.load(os.path.join(folder, f))

            x, y, name = obj["x"], obj["y"], obj["name"]

            if n is None:
                ax.plot(y, label=f"{name}")
            else:
                ax.plot(y[:n], label=f"{name}")


        ax.set_title("Convergent plots", fontsize=14)
        ax.set_xlabel("Number of iteration", fontsize = 12)
        ax.set_ylabel("Rewards", fontsize = 12)
        ax.legend(prop={'size':12})
        plt.show()

        

