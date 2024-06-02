# Simulations for Finite Difference Time Domain Computation of Electromagnetic Fields

By: Mansi Bezbaruah, Ajay George, Matt Giardinelli, Maisha Marzan, Dinh-Quan Tran.

Part of the 2024 Undergraduate Summer School in Modeling and Simulation with PDEs at Texas A&M University.

## Introduction

This is the code for the simulation of electromagnetic waves on a finite domain. The waves here obeys Maxwell's Equations, and reflects under Dirichlet boundary conditions. To simulate free space, we implemented Berenger's Perfectly Matched Layer for absorption of the waves.

We also simulated the waves under the influence of a graphene sheet. Waves from far-away sources passes through the sheet, while waves from close sources form the surface plasmon polariton.

## Usage

Each file is a complete script for the corresponding boundary conditions, and can be run individually. There are naive (nested loops) approach and matrix approach for runtime comparisons. One can also create animations with this code using additional help from [FFmpeg](https://ffmpeg.org/).
