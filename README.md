> # Design of controller for biohybrid actuators of catheters

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) and Hypercube-based Neuroevolution of Augmenting Topologies (HyperNEAT) to design controllers of biohybird actuators (BHAs) focused on catheters. In order to represent and evaluate the controllers generated, BHA morphologies and the effect induced by the controllers are simulated in a physics engine: Voxelyze, which can be found in the following GitHub repository: 

https://github.com/skriegman/reconfigurable_organisms.

To adapt the physics engine to the dynamics of BHAs, two modifications to the source code were performed. The modifications can be found in:

https://github.com/Antisthenis/reconfigurable_organisms/commit/80ae5d9af6f381d565fa4885ba1672f2813a8a28

and

https://github.com/Antisthenis/reconfigurable_organisms/commit/dfcf21dcd0670a0f77d94e826563d09ed82a3786#diff-eed874d9aea4ad5f133265b296188a268bcc85f6f610b9c03ca3ada556cf88f5


> **Architecture**

Since the evolutionary process implies a simulation task, the runtime takes significant time. This software has been designed to reduce the time spent finding suitable controllers for BHAs. It uses concurrency and was designed under a client-server architecture. Generally, the genetic algorithm (GA) is executed on the client side, whereas the fitness function is executed on the server side.

This software was written in Python 3.11 on the client side and Python 3.10 on the server side.

> **Repository Structure**

The source code of this repository is split into two:

* _client_: code related to client side.
* _server_: code related to the server side.

The packages used for the client and server sides are listed in the file called "requirements.md".

> **Important Notice**

The code provided in this repository was used as part of an academic research documented in ..._coming soon_. 
