# RESuM: Rare Event Surrogate Model for  Physics Detector Design

The experimental discovery of neutrinoless double-beta decay (NLDBD) would answer one of the most important questions in physics: Why is there more matter than antimatter in our universe? To maximize the chances of detection, NLDBD experiments must optimize their detector designs to minimize the probability of background events contaminating the detector. Given that this probability is inherently low, design optimization either requires extremely costly simulations to generate sufficient background counts or contending with significant variance. In this work, we formalize this dilemma as a Rare Event Design (RED) problem: identifying optimal design parameters when the design metric to be minimized is inherently small. We then designed the Rare Event Surrogate Model (RESuM) for physics detector design optimization under RED conditions. RESuM uses a pretrained Conditional Neural Process (CNP) model to incorporate additional prior knowledges into a Multi-Fidelity Gaussian Process model. We applied RESuM to optimize neutron moderator designs for the LEGEND NLDBD experiment, identifying an optimal design that reduces neutron background by ($66.5\pm3.5$)\% while using only 3.3\% of the computational resources compared to traditional methods. Given the prevalence of RED problems in other fields of physical sciences, the RESuM algorithm has broad potential for simulation-intensive applications.

See [arxiv:2410.03873](http://arxiv.org/abs/2410.03873)

![alt text](https://github.com/annkasch/legend-multi-fidelity-surrogate-model/blob/main/MF-GP_concept.png)

### Visualization of the LEGEND neutron moderator

[Link to visualization tool](https://annkasch.github.io/legend-multi-fidelity-surrogate-model/)

<img src="https://github.com/annkasch/legend-multi-fidelity-surrogate-model/blob/main/utilities/vis.png" width="600">

