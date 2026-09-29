import plotting as plt
import networkBuild as nb


if __name__ == "__main__":

    net = nb.Network()
    net.networkNodes()      # Create the nodes of the spiderweb network
    net.networkThreads()    # Create the threads of the spiderweb network

    print("Number of nodes:", len(net.X))
    print("Number of threads:", len(net.threads))

    # Plot the network
    plt.plotNetwork(net)