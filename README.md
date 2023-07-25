# **OpenQGO**

Contributions made by Anish Jahagirdar of Lakewood High School for TerraNexum.

## About TerraNexum

TerraNexum is a company located in Evergreen, Colorado  focused on combatting environmental sustainability issues throughout the world. They host many sustainability projects including researching and developing carbon capture hardware, energy technologies(solar canopies, batteries, etc), and the use of quantum computing to solve complex optimization problems that can lead to better energy and sustainability planning.

![TN Logo](https://media.licdn.com/dms/image/D560BAQEHjvadbsgQvQ/company-logo_200_200/0/1688195130686?e=2147483647&v=beta&t=Lc05yWlBt-qerpFeBYdWTH70P2uuJvdV396e_Nn0iXg)

## My Responsibilities

At TerraNexum, my main responsibilities were related to the OpenQGO(Global Quantum Optimizer) project which seeks to provide users with a global map interface that has data layers that the user can toggle on and off in order to tell the optimization algorithm what features to take into consideration when optimizing for maximum and efficient energy flow. This product could be used to see how renewable energy sources should replace fossil fuels over time in the most efficient manner by providing the best results to businesses and homes and minimizing environmental damage. 

## Challenges and Successes

During the development processes, there were a fair share of challenges that had to be dealt with. The project primarily uses QAOA(Quantum Approximate Optimization Algorithm), a specific algorithm to quantum that attempts to minimize the Hamiltonian(think energy cost of a solution for quantum). Originally, the Microsoft Quantum language Q# was researched to see if QAOA could be implemented there but it was quickly realized that there was not sufficient documentation on the programming language to make it easy to learn from scratch. After this, Python was adopted as the primary language behind QGO and it streamlined the process as there were many quantum libraries available to pull code from. 

Another challenge was the actual implementation of QAOA itself. The steps of the project included locating two GeoJSON files(files that have points on a map e.g. schools in the U.S.), capturing one feature to link the two files, and then optimizing for that. Finding two GeoJSON files given our resources presented a challenge but using GIS software, we were able to use .shape files and convert them to GeoJSON so that they could be used. The next challenge related to getting a feature to link the two files. Distance seemed simple to implement but provided a challenge in that a pairwise binary distance matrix had to be created between both sets of points from separate GeoJSON files. This took a good amount of research to find a library that could handle this and eventually scikit seemed to provide a good implementation. 

The hardest challenge of the QAOA process was representing the MaxFlow problem as a QUBO(Quadratic Unconstrained Binary Optimization) which basically entails creating binary variables that can be manipulated to represent the problem. Many libraries provided optimization problems as QUBOs with the exception of the MaxFlow problem providing a unique challenge in either finding a library or using another method to represent MaxFlow. Eventually, a way to model MaxFlow using CPlex and DocPlex along with optimizers provided by Qiskit was used in order to both represent MaxFlow and solve the problem. 

## My Learning

During the internship, I really improved my research and debugging skills when it came to programming. Using Python, a language I had previously not used deeply, I was forced to read lots of documentation and adapt to the programming language gaining a greater control of Python's syntax. Also, learning how to use cloud computing platforms required reading documentation and forums like StackOverflow to help solve specific problems. Along with the programming language barrier, understanding quantum concepts like QAOA or quantum annealing was aided by AI research tools like ChatGPT and ChatSonic. Using other AI research tools like Elicit helped when figuring out how to model the MaxFlow problem as a QUBO so that it could be run on a quantum optimizer. 

While writing the program for the QAOA process, lots of code needed to be debugged since many errors were prone to showing up. It was during this process that I learned the value of organizing my code into easy-to-read functions and variables with names that make sense so that not only would it make it easier for me to see what each component of my program did, but also for others as well. Organizing my code really aided in the debugging process because I could stop a function from calling completely if I wanted and isolate a particular part of the program that I thought might be hindering progress. For example, while optimizing for the MaxFlow problem using the quantum optimizer, results would show up as "infeasible" meaning that a solution could not be found even though results were actually correct. During the debugging process, I isolated the program into its different stages with data loading itself, the modeling functions(the functions that modeled the MaxFlow problem), and the actual optimization code. Through this process, we were able to detect that the data loading and modeling components were correct but the optimization stage for whatever reason showed as "infeasible." This taught me a lot of important skills in debugging code. 



## Future Career and Academic Plans

As of now, my future career and academic plans have stayed the same as they were as I want to attent a 4-year university studying machine learning and something in the realm of computer science. After that, I would like to atatin a job at a cutting-edge technology company working to improve the conditions of the world in either sustainability or any issue of great global significiance. This internship definitely set my mind more toward this goal because I realized the power that technology has in solving deep-rooted sustainabiity issues that our planet is plagued by and how it can be used as a massive force of good even by kids. I also got my eyes opened up to the power that quantum computing has and will continue to have in the future so I definitely want to pursue something in that technology space when I am looking for a career as it has the power to solve complex optimization issues. 

## Thank You!

Dahl Winters was my mentor at TerraNexum and she was the greatest mentor that I think I could have had for an internship like this. Dahl was extremely willing to cater to me and let me choose the work that I was interested in doing and learn in. She was also very open with me about the business side of TerraNexum and that provided so much value to me seeing the inner workings of a business like this. She provided me with access to cloud computing platforms, projects and challenges I didn't know existed, and connections that will be very useful in my future when dealing with sustainability problems. I would like to thank Dahl for establishing even more interest in this field than I had before the internship and I definitely owe her a lot in the future for giving me this opportunity. Thank you Dahl!

## The Process

![Colorado Service Utilities Servicing](https://drive.google.com/uc?export=download&id=1IgJTXIJn_QAsr7vs9vlDAc7qrZ6SRUaJ)

