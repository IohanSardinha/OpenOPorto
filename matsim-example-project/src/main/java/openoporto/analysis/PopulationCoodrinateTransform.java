package openoporto.analysis;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.population.*;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.utils.geometry.CoordinateTransformation;
import org.matsim.core.utils.geometry.transformations.TransformationFactory;

public class PopulationCoodrinateTransform {
    public static void main(String[] args) {

        String filename = "/home/iohan/Documentos/FEUP/MAMS/OpenOPorto/input/plans.xml";

        Population population = PopulationUtils.readPopulation(filename);

        CoordinateTransformation ct = TransformationFactory.getCoordinateTransformation(TransformationFactory.WGS84, "EPSG:3857");

        for(Person p: population.getPersons().values()){
            for(Plan plan: p.getPlans()){
                for(PlanElement planElement: plan.getPlanElements()){
                    if(planElement instanceof Activity a){
                        try {
                            a.setCoord(ct.transform(a.getCoord()));
                        }catch (Exception e){
                            System.out.println("Err");
                        }
                    }
                }
            }
        }

        PopulationUtils.writePopulation(population, filename);
    }
}
