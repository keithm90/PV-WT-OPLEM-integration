# PV-WT-OPLEM Integration

This project extends the Open Platform for Local Energy Markets (OPLEM) by integrating renewable energy models for Photovoltaic (PV) systems and Wind turbines (WT).

## Overview
The work focuses on developing simplified, accurate and computationally efficient models for renewable generation using meteorological data from Renewables.ninja.

## Contents
- Assets.py  
  Contains the implementation of PVAsset and WTAsset classes integrated into the OPLEM framework.

- PV_test.py  
  Validation script for the PV model using irradiance and temperature data.

- WT_test.py  
  Validation script for the wind turbine model using wind speed data.

## Methodology
- PV model based on irradiance scaling with temperature correction.
- WT model based on a piecewise cubic power curve (cut-in, rated, cut-out).

## Validation
Models were validated against Renewables.ninja reference data using:
- RMSE
- MAE
- R²
- Energy error

## Purpose
This repository supports the ENGG30051 Individual Engineering Project and ensures reproducibility of the results presented my academic paper.
