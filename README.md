# OpenQGO
A generalized, open-source version of our Quantum Global Optimizer platform to solve the problem of getting anything where it needs to go.

If the following work is of value to you, please consider supporting our work or joining as a Core Contributor through [Open Collective](https://opencollective.com/terranexum/projects/openqgo).

## Project Scope
![TerraNexum - Business Flow Model - OpenQGO](https://github.com/terranexum/OpenQGO/assets/20586685/41712ee2-b33e-4acd-919c-6f98f8609e0b)

## Overview

The OpenQGO project aims to open-source some of our previous work behind our Quantum Global Optimizer (at https://qgo-dev.terranexum.com) and modify it to create a general engine, able to use quantum optimization if the problem size is large enough, to identify locations where technologies can be co-located based on their inputs and outputs to optimize a metric defined using those inputs and outputs.

The output being aimed for: a set of labeled nodes and connections describing what technologies should go where, given an initial state (today) and a final desired state (tomorrow). OpenQGO will calculate the pipes needed to optimize the flows of these inputs and outputs over time to reach the goal. These are pipes in time, not in space; on the ground, all we will see are appearances of new projects using co-located technologies, and where they should be.  Projects that fail to materialize for whatever reason are fine - the optimization problem gets recalculated to find the next best solution with the most up-to-date information. 

The result: something like this diagram, from the Kalundborg Symbiosis in Denmark (https://www.symbiosis.dk/en/). Folks there understood the importance of what they were building a long time ago, but the information they had has failed to get to many places it needed to go. 

![Industrial Symbiosis](https://github.com/terranexum/OpenQGO/blob/main/Technology-Co-Location-Symbiosis.png)

When you want zero waste and maximum use of the available energy for growth, living things are kind of a good place to look to for basing designs on. Most have evolved circulatory systems as well, once they are large and complex enough that free diffusion-based systems become too inefficient.

