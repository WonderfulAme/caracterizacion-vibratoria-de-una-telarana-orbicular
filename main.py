import plotting as plt
import networkBuild as nb


if __name__ == "__main__":

    net = nb.Network()
    net.networkNodes()      # Create the nodes of the spiderweb network
    net.networkThreads()    # Create the threads of the spiderweb network
    net.threadProperties()  # Calculate the properties of the threads
    net.nodalMasses()       # Calculate the nodal masses of the spiderweb network

    print("Number of nodes:", len(net.X))
    print("Number of threads:", len(net.threads))

    net.degreesOfFreedom()  # Calculate the degrees of freedom of the spiderweb network

    print("Number of degrees of freedom:", net.ndof)

    net.globalStiffnessMatrix()  # Assemble the stiffness matrix of the spiderweb network

    net.globalMassMatrix()

    net.verification()  # Verify the properties of the spiderweb network

    # Plot the network
    plt.plotNetwork(net)    


    