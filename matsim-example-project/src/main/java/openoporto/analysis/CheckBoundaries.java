package openoporto.analysis;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.utils.geometry.CoordinateTransformation;
import org.matsim.core.utils.geometry.transformations.TransformationFactory;
import org.matsim.run.NetworkCleaner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CheckBoundaries {
    public static void main(String[] args) {
        Network network = NetworkUtils.readNetwork("/home/iohan/Documentos/FEUP/MAMS/OpenOPorto/input/testNetwork2.xml");

        Population population = PopulationUtils.readPopulation("/home/iohan/Documentos/FEUP/MAMS/OpenOPorto/input/plans.xml");

        CoordinateTransformation ct = TransformationFactory.getCoordinateTransformation(TransformationFactory.WGS84, "EPSG:3857");

        List<Double> Xs = new ArrayList<>();
        List<Double> Ys = new ArrayList<>();

        for(Node node :network.getNodes().values()){
            Coord coord = node.getCoord();
            Xs.add(coord.getX());
            Ys.add(coord.getY());
        }

        double minX = Collections.min(Xs);
        double maxX = Collections.max(Xs);
        double minY = Collections.min(Ys);
        double maxY = Collections.max(Ys);

        System.out.println("Min x:"+minX+", Max x:"+maxX+", Min y:"+minY+", Max y:"+maxY);
        System.out.println("Top left: x = "+minX+" y = "+maxY);
        System.out.println("Top right: x = "+maxX+" y = "+maxY);
        System.out.println("Bottom right: x = "+maxX+" y = "+minY);
        System.out.println("Bottom left: x = "+minX+" y = "+minY);

        for(Person p: population.getPersons().values()){
            for(Plan plan: p.getPlans()){
                for(PlanElement planElement: plan.getPlanElements()){
                    if(planElement instanceof Activity a){
                        Coord coord = a.getCoord();
                        coord = ct.transform(coord);
                        double x = coord.getX();
                        double y = coord.getY();
                        System.out.println(coord);
                        System.out.println(x >= minX && x <= maxX && y >= minY && y <= maxY);
                    }
                }
            }
        }

    }
}
