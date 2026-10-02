if [ "$1" = "--memory" ]; then
    #check if an argument was provided for memory allocation
    if [ -z "$2" ]; then
        echo "No maximum memory value provided for memory allocation"
        exit 1
    fi
    if [ -z "$3" ]; then
        echo "No minimum memory value provided for memory allocation"
        exit 1
    fi
    MEMORY_MAX=$2
    MEMORY_MIN=$3
    echo "Running MATSim with memory allocation: $MEMORY_MIN - $MEMORY_MAX"
    echo "java -Xms"$MEMORY_MIN" -Xmx"$MEMORY_MAX" -XX:+UseG1GC -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar --config=input/config.xml"
    java -Xms"$MEMORY_MIN" -Xmx"$MEMORY_MAX" -XX:+UseG1GC -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar run  --config=input/config.xml
else
    echo "Running MATSim with default memory allocation"
    echo "java -XX:+UseG1GC -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar --config=input/config.xml"
    java -XX:+UseG1GC -jar matsim-example-project/matsim-example-project-0.0.1-SNAPSHOT.jar run --config=input/config.xml
fi
