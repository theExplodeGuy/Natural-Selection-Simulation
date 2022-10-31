# Natural-Selection-Simulation
#### Video Demo:  <URL HERE>

#### Description:
Natural Selection implies that self-replication, inheritance, mutation and selection. The project attempts to emulate the main characteristics of natural selection. Organisms self-replicate to propagate their genes and mutation occurs at random. Selection takes place by killing organisms unable to secure a food source to replenish their energy.
The project aims to emulate the behaviour of simple organisms in an environment of changing parameters. Organisms may progress through the simulation by ensuring a food source, fighting between them, cooperating, and reproducing.

Organisms behave based on randomly generated attributes assigned at the beginning of the simulation. Attributes consist of:
- power
- strength
- aggressiveness
- leadership
- team spirit
- sense
- size
- speed
- energy consumption

The starting parameters of the environment are: 
    ```POPULATION = 10
       MAX_BOUNDARY = 5
       MIN_BOUNDARY = 0
       FOOD_SOURCE = 20
       ENERGY = 2000000```
The values can be varied to study the effect of changing food source, energy and population

#### Methodology:
The simulation is written in python using matplotlib to animate the organisms.
The behaviour of the organisms is determined by a series of equations:

- ENERGY CONSUMPTION
```self.consumption = 200 * self.radius + (1000 / 41) * self.speed + (2000 / 14) * self.strength```
This equation determines the life-span of the organism given all the attributes that consume energy.

- TO DECIDE FIGHT OR FLIGHT
```4 * self.aggressiveness + 2 * self.strength + 280 * self.radius) - 7 * (1400 / 41) * self.speed```
This equation decides whether an organism should engage in fighting based on whether it will be beneficial for it or not. The equation takes into account attributes such as aggressiveness, strength, and speed. Attributes that are usually decisive in a fight.

- DETERMINE FIGHT RESULT
```self.power = 2 * self.strength + 280 * self.radius```
This equation decides the winner of the fight between two organisms.

#### Results:
The simulation appears to favor aggression and power over cooperation. Weaker organisms that survive tend to cooperate in order to gain additional strength in numbers.
![alt text](https://github.com/theExplodeGuy/Natural-Selection-Simulation/blob/main/Figures/Average_Aggression.png?raw=true)
Average Aggression
![alt text](https://github.com/theExplodeGuy/Natural-Selection-Simulation/blob/main/Figures/Average_Aggression.png?raw=true)



