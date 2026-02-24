def _activity_params(param):
    return\
f"""
		<parameterset type="activityParams" >
			<param name="activityType"            value="{param["type"]}" /> <!-- {param["type"]} -->
			<param name="priority"        value="1" />
			<param name="typicalDuration" value="{param["typicalDuration"]}" />
		</parameterset>
"""

def _mode_params(mode):
    return\
f"""
		<parameterset type= "modeParams" >
			<param name= "mode" value= "{mode}" />
			<param name= "monetaryDistanceRate" value= "-0.0002 " />
		</parameterset>
"""

DEFAULT_CONFIG = lambda config:\
f"""<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>

	<module name="global">
		<param name="randomSeed" value="4711" />
		<param name="coordinateSystem" value="Atlantis" />
	</module>

	<module name="network">
		<param name="inputNetworkFile" value="{config.get("inputNetworkFile", "network.xml")}" />
	</module>

	<module name="plans">
		<param name="inputPlansFile" value="{config.get("inputPlansFile", "plans.xml")}" />
	</module>

	<module name="transit">
		<param name="transitScheduleFile" value="{config.get("transitScheduleFile", "schedule.xml")}" />
		<param name="vehiclesFile" value="{config.get("vehiclesFile", "vehicles.xml")}" />
		<param name="transitModes" value="{','.join(['pt'] + config.get("transitModes", []))}" />
		<param name="useTransit" value="{'true' if len(config.get("transitModes",[])) > 0 else 'false'}" />
	</module>

	<module name="controller">
		<param name= "routingAlgorithmType" value= "AStarLandmarks" />
		<param name="outputDirectory" value="{config.get("outputDirectory", "./output")}" />
		<param name="firstIteration" value="{config.get("firstIteration", "0")}" />
		<param name="lastIteration" value="{config.get("lastIteration", "10")}" />
	</module>

	<module name="qsim">
		<!-- "start/endTime" of MobSim (00:00:00 == take earliest activity time/ run as long as active vehicles exist) -->
		<param name="startTime" value="00:00:00" />
		<param name="endTime" value="23:59:59" />
		<param name="flowCapacityFactor" value="0.1" />
		<param name="mainMode" value="{','.join(config.get("transitModes", [])+['car'])}" />
		<param name = "snapshotperiod"	value = "00:00:00"/> <!-- 00:00:00 means NO snapshot writing -->
	</module>

	<module name="scoring">
		<param name="learningRate" value="1.0" />
		<param name="brainExpBeta" value="2.0" />

		<param name="lateArrival" value="-18" />
		<param name="earlyDeparture" value="-0" />
		<param name="performing" value="+6" />
		<param name="waiting" value="-0" />
{"".join([_activity_params(param) for param in config.get("activityParams", [])])}


	<parameterset type= "modeParams" >
		<param name= "mode" value= "pt" />
		<param name= "monetaryDistanceRate" value= "-0.0002 " />
	</parameterset>
	</module>


	<module name="routing">
		<param name="accessEgressType" value="accessEgressModeToLink" />
		<param name= "networkModes" value= "car"/>
		<param name="networkRouteConsistencyCheck" value="disable" />
    </module>

	<module name="changeMode">
  		<param name="modes" value="{','.join(config.get("activityModes", []))}" /> 
	</module>

	<module name="replanning">
		<param name="maxAgentPlanMemorySize" value="0" /> <!-- 0 means unlimited -->

		<param name="ModuleProbability_1" value="0.6" />
		<param name="Module_1" value="BestScore" />

		<param name="ModuleProbability_2" value="0.1" />
		<param name="Module_2" value="ReRoute" />

		<param name="ModuleProbability_3" value="0.3" />
	    <param name="Module_3" value="ChangeTripMode" />

	</module>

</config>"""