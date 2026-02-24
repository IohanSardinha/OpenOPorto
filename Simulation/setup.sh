#!/bin/bash

if [ "$1" = "clear" ] || [ "$2" = "clear" ] ; then
    echo "Clearing input and output folders..."
    rm -rf ./input/* ./output/*
fi

echo "Setting up MATSim simulation environment..."

mkdir -p ./input
mkdir -p ./output

if [ -f "../PhysicalNetwork/network.xml" ] ; then
    echo "Copying network.xml from ../PhysicalNetwork."
    cp ../PhysicalNetwork/network.xml ./input/
else    
    echo "No network.xml found in ../PhysicalNetwork."
    
    read -p "Do you want to run generate_network.py to create a network? (y/n) " choice
    if [ "$choice" = "y" ]; then
        cd ../PhysicalNetwork
        python generate_network.py
        cd ../Simulation
        cp ../PhysicalNetwork/network.xml ./input/
    else
        echo "Please provide a network.xml file in ./input/ to proceed."
    fi
fi

if [ -f "../PhysicalNetwork/schedule.xml" ] ; then
    echo "Copying schedule.xml from ../PhysicalNetwork."
    cp ../PhysicalNetwork/schedule.xml ./input/
fi
if [ -f "../PhysicalNetwork/vehicles.xml" ] ; then
    echo "Copying vehicles.xml from ../PhysicalNetwork."    
    cp ../PhysicalNetwork/vehicles.xml ./input/
fi

if [ -f "../Population/plans.xml" ] ; then
    echo "Copying plans.xml from ../Population."
    cp ../Population/plans.xml ./input/
else    
    echo "No plans.xml found in ../Population."
    
    read -p "Do you want to run generate_population.py to create a population? (y/n) " choice
    if [ "$choice" = "y" ]; then
        cd ../Population
        python generate_population.py
        cd ../Simulation
        cp ../Population/plans.xml ./input/
    else
        echo "Please provide a plans.xml file in ./input/ to proceed."
    fi
fi

if [ ! -f "./input/config.xml" ] ; then
    echo "Creating default config.xml..."
    python helpers/create_config.py ./input/config.xml
fi

if [ ! -d "matsim-example-project" ] ; then
    echo "Cloning MATSim example project..."
    git clone https://github.com/matsim-org/matsim-example-project.git  
    sed -i 's/gui.MATSimGUI/project.RunMatsim/g' matsim-example-project/pom.xml
else
    echo "MATSim example found!."
fi

if [ ! -f "./matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar" ] ; then
    cd matsim-example-project
    echo "Building MATSim example project..."
    ./mvnw clean package
    cd ..
fi

echo "Setup complete. You can now run the MATSim simulation."


if [ "$1" = "run" ] || [ "$2" = "run" ] ; then
    echo "Running MATSim simulation..."
    java -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml
else
    echo "To run the simulation, execute: "
    echo "java -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml"
fi

