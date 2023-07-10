# **OpenQGO**

Contributions made by Anish Jahagirdar of Lakewood High School for TerraNexum.

## About TerraNexum

TerraNexum is a company located in Evergreen, Colorado  focused on combatting environmental sustainability issues throughout the world. They host many sustainability projects including researching and developing carbon capture hardware, energy technologies(solar canopies, batteries, etc), and the use of quantum computing to solve complex optimization problems that can lead to better energy and sustainability planning.

![TN Logo](https://media.licdn.com/dms/image/D560BAQEHjvadbsgQvQ/company-logo_200_200/0/1688195130686?e=2147483647&v=beta&t=Lc05yWlBt-qerpFeBYdWTH70P2uuJvdV396e_Nn0iXg)

## My Responsibilities

At TerraNexum, my main responsibilities were related to the OpenQGO(Global Quantum Optimizer) project which seeks to provide users with a global map interface that has data layers that the user can toggle on and off in order to tell the optimization algorithm what features to take into consideration when optimizing for maximum and efficient energy flow. This product could be used to see how renewable energy sources should replace fossil fuels over time in the most efficient manner by providing the best results to businesses and homes and minimizing environmental damage. 

## Challenges and Successes

During the development processes, there were a fair share of challenges that had to be dealt with. The project primarily uses QAOA(Quantum Approximate Optimization Algorithm), a specific algorithm to quantum that attempts to minimize the Hamiltonian(think energy cost of a solution for quantum). Originally, the Microsoft Quantum language Q# was researched to see if QAOA could be implemented there but it was quickly realized that there was not sufficient documentation on the programming language to make it easy to learn from scratch. After this, Python was adopted as the primary language behind QGO and it streamlined the process as there were many quantum libraries available to pull code from. 

Another challenge was the actual implementation of QAOA itself. The steps of the project included locating two GeoJSON files(files that have points on a map e.g. schools in the U.S.), capturing one feature to link the two files by, and then optimizing for that. 

