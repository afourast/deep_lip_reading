#!/usr/bin/env bash

mkdir models && cd models

wget http://www.robots.ox.ac.uk/~vgg/research/deep_lip_reading/models/lrs2_lip_model.zip && \
unzip lrs2_lip_model.zip && \
rm lrs2_lip_model.zip

wget http://www.robots.ox.ac.uk/~vgg/research/deep_lip_reading/models/lrs2_language_model.zip && \
unzip lrs2_language_model.zip && \
rm lrs2_language_model.zip
