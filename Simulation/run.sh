if [ "$1" = "--memory" ]; then
    #check if an argument was provided for memory allocation
    if [ -z "$2" ]; then
        echo "No value provided for memory allocation"
        exit 1
    fi
    MEMORY=$2
    echo "Running MATSim with memory allocation: $MEMORY"
    echo "java -Xmx"$MEMORY" -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml"
    java -Xmx"$MEMORY" -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml
else
    echo "Running MATSim with default memory allocation"
    echo "java -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml"
    java -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar input/config.xml
fi