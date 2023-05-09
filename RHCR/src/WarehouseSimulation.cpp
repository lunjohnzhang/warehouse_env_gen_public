// Main API for python to run simulation

#include <pybind11/pybind11.h>
#include "KivaSystem.h"
#include "SortingSystem.h"
#include "OnlineSystem.h"
#include "BeeSystem.h"
#include "ID.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace py = pybind11;


void set_parameters(BasicSystem& system, const py::kwargs& kwargs)
{
	system.outfile = kwargs["output"].cast<std::string>();
	system.screen = kwargs["screen"].cast<int>();
	system.log = kwargs["log"].cast<bool>();
	system.num_of_drives = kwargs["agentNum"].cast<int>();
	system.time_limit = kwargs["cutoffTime"].cast<int>();
	system.simulation_window = kwargs["simulation_window"].cast<int>();
	system.planning_window = kwargs["planning_window"].cast<int>();
	system.travel_time_window = kwargs["travel_time_window"].cast<int>();
	system.consider_rotation = kwargs["rotation"].cast<bool>();
	system.k_robust = kwargs["robust"].cast<int>();
	system.hold_endpoints = kwargs["hold_endpoints"].cast<bool>();
	system.useDummyPaths = kwargs["dummy_paths"].cast<bool>();
    system.save_result = kwargs["save_result"].cast<bool>();
    system.save_solver = kwargs["save_solver"].cast<bool>();
    system.stop_at_traffic_jam = kwargs["stop_at_traffic_jam"].cast<bool>();
	if (kwargs.contains("seed"))
		system.seed = kwargs["seed"].cast<int>();
	else
		system.seed = (int)time(0);
	srand(system.seed);
}


MAPFSolver* set_solver(const BasicGraph& G, const py::kwargs& kwargs)
{
	string solver_name = kwargs["single_agent_solver"].cast<string>();
	SingleAgentSolver* path_planner;
	MAPFSolver* mapf_solver;
	if (solver_name == "ASTAR")
	{
		path_planner = new StateTimeAStar();
	}
	else if (solver_name == "SIPP")
	{
		path_planner = new SIPP();
	}
	else
	{
		cout << "Single-agent solver " << solver_name << "does not exist!" << endl;
		exit(-1);
	}

	solver_name = kwargs["solver"].cast<string>();
	if (solver_name == "ECBS")
	{
		ECBS* ecbs = new ECBS(G, *path_planner);
		ecbs->potential_function = kwargs["potential_function"].cast<string>();
		ecbs->potential_threshold = kwargs["potential_threshold"].cast<double>();
		ecbs->suboptimal_bound = kwargs["suboptimal_bound"].cast<double>();
		mapf_solver = ecbs;
	}
	else if (solver_name == "PBS")
	{
		PBS* pbs = new PBS(G, *path_planner);
		pbs->lazyPriority = kwargs["lazyP"].cast<bool>();
		pbs->prioritize_start = kwargs["prioritize_start"].cast<bool>();
		pbs->setRT(kwargs["CAT"].cast<bool>(), kwargs["prioritize_start"].cast<bool>());
		mapf_solver = pbs;
	}
	else if (solver_name == "WHCA")
	{
		mapf_solver = new WHCAStar(G, *path_planner);
	}
	else if (solver_name == "LRA")
	{
		mapf_solver = new LRAStar(G, *path_planner);
	}
	else
	{
		cout << "Solver " << solver_name << "does not exist!" << endl;
		exit(-1);
	}

	if (kwargs["id"].cast<bool>())
	{
		return new ID(G, *path_planner, *mapf_solver);
	}
	else
	{
		return mapf_solver;
	}
}

/**
 * Run one round of warehouse simulation
 *
 *
 *
 * @param kwargs py dict with all the params, specifically:
 * 	   scenario       		: scenario (SORTING, KIVA, ONLINE, BEE)
 *     map				 	: input map json string
 *     task                 : input task file
 *     output			    : output folder name
 *     agentNum             : number of drives
 *     cutoffTime           : cutoff time (seconds)
 *     seed                 : random seed
 *     screen               : screen option (0: none; 1: results; 2:all)
 *     solver	            : solver (LRA, PBS, WHCA, ECBS)
 *     id                   : independence detection
 *     single_agent_solver	: single-agent solver (ASTAR, SIPP)
 *     lazyP                : use lazy priority
 *     simulation_time      : run simulation
 *     simulation_window    : call the planner every simulation_window timesteps
 *     travel_time_window   : consider the traffic jams within the given window
 *     planning_window      : the planner outputs plans with first
 *     						: planning_window timesteps collision-free
 *     potential_function	: potential function (NONE, SOC, IC)
 *     potential_threshold  : potential threshold
 *     rotation           	: consider rotation
 *     robust               : k-robust (for now, only work for PBS)
 *     CAT                	: use conflict-avoidance table
 *     hold_endpoints     	: Hold endpoints from Ma et al, AAMAS 2017
 *     dummy_paths        	: Find dummy paths from Liu et al, AAMAS 2019
 *     prioritize_start    	: Prioritize waiting at start locations
 *     suboptimal_bound     : Suboptimal bound for ECBS
 *     log                	: save the search trees (and the priority trees)
 *     force_new_logdir     : force the program to create a new logdir
 *     save_result          : whether save the result (path, etc) to disk
 *     save_solver          : whether save the solver result (solver.csv) to
 *                            disk
 *     save_heuristics_table: whether save the heuristics table to disk
 *     stop_at_traffic_jam  : whether to stop at traffic jam

 * @return result (json string): summerized objective, measures, and any
 * ancillary data
 */
std::string run(const py::kwargs& kwargs) {
	if (kwargs["test"].cast<bool>())
	{
		// For testing
		json result = {
			{"throughput", 10},
			{"tile_usage", {0.0, 1.0, 2.0}},
			{"num_wait", {3.0, 4.0, 5.0}},
			{"finished_task_len", {6.0, 7.0, 8.0}},
			{"tile_usage_mean", 1.0},
			{"tile_usage_std", 0.67},
			{"num_wait_mean", 4.0},
			{"num_wait_std", 0.67},
			{"finished_len_mean", 7.0},
			{"finished_len_std", 0.67}
		};

		return result.dump(4);
	}

    // Default variables
    if (!kwargs.contains("planning_window"))
    {
        kwargs["planning_window"] = INT_MAX / 2;
        // cout << kwargs["planning_window"].cast<int>() << endl;
    }

    namespace po = boost::program_options;
    clock_t start_time = clock();
	json result;
	result["finish"] = false; // status code to false by default

    // check params
    if (kwargs["hold_endpoints"].cast<bool>() or kwargs["dummy_paths"].cast<bool>())
    {
        if (kwargs["hold_endpoints"].cast<bool>() and kwargs["dummy_paths"].cast<bool>())
        {
            std::cerr << "Hold endpoints and dummy paths cannot be used simultaneously" << endl;
            exit(-1);
        }
        if (kwargs["simulation_window"].cast<int>() != 1)
        {
            std::cerr << "Hold endpoints and dummy paths can only work when the simulation window is 1" << endl;
            exit(-1);
        }
        if (kwargs["planning_window"].cast<int>() < INT_MAX / 2)
        {
            std::cerr << "Hold endpoints and dummy paths cannot work with planning windows" << endl;
            exit(-1);
        }
    }

    // make dictionary
    bool force_new_logdir = kwargs["force_new_logdir"].cast<bool>();
	boost::filesystem::path dir(kwargs["output"].cast<std::string>() +"/");

    // Remove previous dir is necessary.
    if (boost::filesystem::exists(dir) && force_new_logdir)
    {
        boost::filesystem::remove_all(dir);
    }

    if (kwargs["log"].cast<bool>() ||
        kwargs["save_heuristics_table"].cast<bool>() ||
        kwargs["save_result"].cast<bool>() ||
        kwargs["save_solver"].cast<bool>())
	{
        boost::filesystem::create_directories(dir);
		boost::filesystem::path dir1(kwargs["output"].cast<std::string>() + "/goal_nodes/");
		boost::filesystem::path dir2(kwargs["output"].cast<std::string>() + "/search_trees/");
		boost::filesystem::create_directories(dir1);
		boost::filesystem::create_directories(dir2);
	}


	if (kwargs["scenario"].cast<string>() == "KIVA")
	{
		KivaGrid G;
		G.screen = kwargs["screen"].cast<int>();
        G.hold_endpoints = kwargs["hold_endpoints"].cast<bool>();
	    G.useDummyPaths = kwargs["dummy_paths"].cast<bool>();
        G._save_heuristics_table = kwargs["save_heuristics_table"].cast<bool>();
		if (!G.load_map_from_jsonstr(kwargs["map"].cast<std::string>()))
			return result.dump(4);
		MAPFSolver* solver = set_solver(G, kwargs);
		KivaSystem system(G, *solver);
		set_parameters(system, kwargs);
		G.preprocessing(system.consider_rotation,
						kwargs["output"].cast<std::string>());
		result = system.simulate(kwargs["simulation_time"].cast<int>());
		result["finish"] = true; // Change status code
        double runtime = (double)(clock() - start_time)/ CLOCKS_PER_SEC;
		cout << "Overall runtime: " << runtime << " seconds." << endl;
        result["cpu_runtime"] = runtime;
		return result.dump(4); // Dump to string with indent 4
	}
	// *********** Note: currrently unsupported, TODO for future **********
	// else if (kwargs["scenario"].cast<string>() == "SORTING")
	// {
	// 	 SortingGrid G;
	// 	 if (!G.load_map(kwargs["map"].cast<std::string>()))
	// 		 return py::make_tuple(-1);
	// 	 MAPFSolver* solver = set_solver(G, kwargs);
	// 	 SortingSystem system(G, *solver);
	// 	 assert(!system.hold_endpoints);
	// 	 assert(!system.useDummyPaths);
	// 	 set_parameters(system, kwargs);
	// 	 G.preprocessing(system.consider_rotation);
	// 	 system.simulate(kwargs["simulation_time"].cast<int>());
	// 	 return py::make_tuple(0);
	// }
	// else if (kwargs["scenario"].cast<string>() == "ONLINE")
	// {
	// 	OnlineGrid G;
	// 	if (!G.load_map(kwargs["map"].cast<std::string>()))
	// 		return py::make_tuple(-1);
	// 	MAPFSolver* solver = set_solver(G, kwargs);
	// 	OnlineSystem system(G, *solver);
	// 	assert(!system.hold_endpoints);
	// 	assert(!system.useDummyPaths);
	// 	set_parameters(system, kwargs);
	// 	G.preprocessing(system.consider_rotation);
	// 	system.simulate(kwargs["simulation_time"].cast<int>());
	// 	return py::make_tuple(0);
	// }
	// else if (kwargs["scenario"].cast<string>() == "BEE")
	// {
	// 	BeeGraph G;
	// 	if (!G.load_map(kwargs["map"].cast<std::string>()))
	// 		return py::make_tuple(-1);
	// 	MAPFSolver* solver = set_solver(G, kwargs);
	// 	BeeSystem system(G, *solver);
	// 	assert(!system.hold_endpoints);
	// 	assert(!system.useDummyPaths);
	// 	set_parameters(system, kwargs);
	// 	G.preprocessing(kwargs["tcastk"].cast<std::string>(), system.consider_rotation);
	// 	system.load_task_assignments(kwargs["tcastk"].cast<std::string>());
	// 	system.simulate();
	// 	double runtime = (double)(clock() - start_time)/ CLOCKS_PER_SEC;
	// 	cout << "Overall runtime:			" << runtime << " seconds." << endl;
	// 	// cout << "	Reading from file:		" << G.loading_time + system.loading_time << " seconds." << endl;
	// 	// cout << "	Preprocessing:			" << G.preprocessing_time << " seconds." << endl;
	// 	// cout << "	Writing to file:		" << system.saving_time << " seconds." << endl;
	// 	cout << "Makespan:		" << system.get_makespan() << " timesteps." << endl;
	// 	cout << "Flowtime:		" << system.get_flowtime() << " timesteps." << endl;
	// 	cout << "Flowtime lowerbound:	" << system.get_flowtime_lowerbound() << " timesteps." << endl;
	// 	auto flower_ids = system.get_missed_flower_ids();
	// 	cout << "Missed tcastks:";
	// 	for (auto id : flower_ids)
	// 		cout << " " << id;
	// 	cout << endl;
	// 	// cout << "Remaining tcastks: " << system.get_num_of_remaining_tcastks() << endl;
	// 	cout << "Objective: " << system.get_objective() << endl;
	// 	std::ofstream output;
	// 	output.open(kwargs["output"].cast<std::string>() + "/MAPF_results.txt", std::ios::out);
	// 	output << "Overall runtime: " << runtime << " seconds." << endl;;
	// 	output << "Makespan: " << system.get_makespan() << " timesteps." << endl;
	// 	output << "Flowtime: " << system.get_flowtime() << " timesteps." << endl;
	// 	output << "Flowtime lowerbound: " << system.get_flowtime_lowerbound() << " timesteps." << endl;
	// 	output << "Missed tcastks:";
	// 	for (auto id : flower_ids)
	// 		output << " " << id;
	// 	output << endl;
	// 	output << "Objective: " << system.get_objective() << endl;
	// 	output.close();
    //     return py::make_tuple(0);
	// }
	// *************************************************************
	else
	{
		cout << "Scenario " << kwargs["scenario"].cast<string>() << "does not exist!" << endl;
		return result.dump(4);
	}

    return result.dump(4);
}

string playground(){
	std::string json_string = R"(
	{
		"pi": 3.141,
		"happy": true
	}
	)";
	json ex1 = json::parse(json_string);

	cout << ex1["pi"] << endl;

	return ex1.dump();
}


PYBIND11_MODULE(warehouse_sim, m) {
	// optional module docstring
    // m.doc() = ;

    m.def("playground", &playground, "Playground function to test everything");
    // m.def("add", &add, py::arg("i")=0, py::arg("j")=1);
    m.def("run", &run, "Function to run warehouse simulation");
}