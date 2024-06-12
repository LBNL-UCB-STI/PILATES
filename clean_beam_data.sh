#!/bin/sh

rm current_stage.yaml

echo "Deleting beam outputs"
for d in pilates/beam/beam_output/*/ ; do
    echo "deleting $d";
    rm -rf "$d"*
done
rm pilates/beam/beam_output/*.omx

echo "Deleting activitysim output"

rm pilates/activitysim/output/*
rm -r pilates/activitysim/output/year*

echo "Deleting interim activitysim inputs"
rm pilates/activitysim/data/*

echo "Deleting interim urbansim data"
rm pilates/urbansim/data/input*
rm pilates/urbansim/data/model*

echo "Deleting inexus output data"
rm pilates/postprocessing/output/*
rm pilates/postprocessing/MEP/*
