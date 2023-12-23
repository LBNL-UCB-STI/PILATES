#!/bin/sh

echo "Deleting beam outputs"
for d in pilates/beam/beam_output/*/ ; do
    echo "deleting $d";
    sudo rm -rf "$d"*
done
sudo rm pilates/beam/beam_output/*.omx

echo "Deleting activitysim output"

sudo rm pilates/activitysim/output/*
sudo rm -r pilates/activitysim/output/year*

echo "Deleting interim activitysim inputs"
sudo rm pilates/activitysim/data/*

echo "Deleting interim urbansim data"
sudo rm pilates/urbansim/data/input*
sudo rm pilates/urbansim/data/model*

echo "Deleting inexus output data"
sudo rm pilates/postprocessing/output/*
sudo rm pilates/postprocessing/MEP/*

