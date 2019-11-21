# TODO: easy distribution

## requirements
For distributing the project and allowing others to use and expand it as easy as possible, we have the following requirements:

 * Anyone should be able to download the scripts and data and retrain it with one or more own categories
 * Anyone should be able to deploy the model on a raspberry pi.

We would like to make both options as easy as possible. If it were only about deploying the model on a raspberry pi, we could use things like docker, or even package the whole python project + dependencies as a linux executable.
But, since a typical user will also want to retrain the model, they will need a python environment with all dependences on the computer, so they need to know how to install this anyway.

## Proposed method

 * Make a python package of the project, with train and run cli commands [4]
 * write a howto on how to install the environment on Linux, Windows and MacOS for training [4]
 * Write a howto on how to install the environment on a raspberry pi (using berryconda) and how to use a self-trained model [4]

## Installation on Raspberry pi

 * Install [berryconda](https://github.com/jjhelmus/berryconda)
 * copy distribute_to_pi files
 * 
