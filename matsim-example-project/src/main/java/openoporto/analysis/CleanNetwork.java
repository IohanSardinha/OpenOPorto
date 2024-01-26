package openoporto.analysis;

import com.graphhopper.reader.osm.OSMReader;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.io.MatsimNetworkReader;
import org.matsim.core.network.io.NetworkWriter;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.network.algorithms.NetworkCleaner;

public class CleanNetwork {
    public static void main(String[] args) {
        String inputFile = "/home/iohan/Documentos/FEUP/MAMS/OpenOPorto/input/network.xml";
        String outputFile = "/home/iohan/Documentos/FEUP/MAMS/OpenOPorto/input/network_clean.xml";
        Network network = NetworkUtils.readNetwork(inputFile);
        NetworkCleaner networkCleaner = new NetworkCleaner();
        networkCleaner.run(network);

        

        NetworkUtils.writeNetwork(network, outputFile);
    }
}
