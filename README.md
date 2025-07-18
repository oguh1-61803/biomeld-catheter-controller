> # Design of controllers for soft actuator morphologies of catheters

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) and Hypercube-based Neuroevolution of Augmenting Topologies (HyperNEAT) to design controllers of soft actuator morphologies (SAMs) - sometimes called biohybrid actuators (BHAs) - focused on catheters. In order to represent and evaluate the controllers generated, SAMs and the effect induced by the controllers are simulated in a physics engine: Voxelyze, which can be found in the following GitHub repository: 

https://github.com/skriegman/reconfigurable_organisms

To adapt the physics engine to the dynamics of SAMs, two modifications to the source code were performed. The modifications can be found in:

* https://github.com/Antisthenis/reconfigurable_organisms/commit/80ae5d9af6f381d565fa4885ba1672f2813a8a28

* https://github.com/Antisthenis/reconfigurable_organisms/commit/dfcf21dcd0670a0f77d94e826563d09ed82a3786#diff-eed874d9aea4ad5f133265b296188a268bcc85f6f610b9c03ca3ada556cf88f5

For benchmark purposes, a standard genetic algorithm (SGA) is utilised as a baseline. The source code of the SGA is also included.

> **Architecture**

Since the evolutionary process implies a simulation task, the runtime takes significant time. This software has been designed to reduce the time spent finding suitable controllers for SAMs. It uses concurrency and was designed under a client-server architecture. Generally, the genetic algorithm (GA) is executed on the client side, whereas the the core of the fitness function (Voxelyze) is executed on the server side. 

This software was written in Python 3.11 on the client side and Python 3.10 on the server side.

> **Repository Structure**

The source code of this repository is split into two:

* _client_: code related to client side. It contains the implementation related to NEAT, HyperNEAT, and SGA.
* _server_: code related to the server side. It contains the implementation related to Voxelyze.

The packages used for the client and server sides are listed in the file called "requirements.md".

> **Important Notice**

The code provided in this repository was used as part of an academic research documented in:

* https://www.scitepress.org/Link.aspx?doi=10.5220/0012919300003837
* https:

Furthermore, this project has received funding from the European Union’s Horizon Europe Research and Innovation programme under grant agreement No. 101070328.UWE researchers were funded by the UK Researchand Innovation grant No. 10044516. 
